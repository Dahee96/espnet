"""Speech Cleaner Vocoder: HiFi-GAN with Snake activation.

Placed under espnet2/enh/decoder/ alongside the existing conv_decoder.py
and stft_decoder.py.  The existing decoders are NOT modified.

Key differences from standard HiFi-GAN already in ESPnet (gan_tts/hifigan/):
  - Input  : SSL feature vectors [B, T_ssl, ssl_dim=1024], NOT mel-spectrogram
  - Activation : Snake  (DAC / BigVGAN style) instead of LeakyReLU
  - Upsampling : [8, 5, 4, 3, 2] → 960× (50 Hz SSL → 48 kHz waveform)
  - No pretrained weights assumed; trained from scratch in two stages

The module is an AbsDecoder so it slots into the standard ESPnet enh pipeline.
SSL features are passed via the `additional` dict (key "pred_ssl_feat") that
the ESPnet model forwards from the feature-predictor's `others` output.

References
----------
Kong et al., "HiFi-GAN," NeurIPS 2020.
Kumar et al., "High-Fidelity Audio Compression with Improved RVQGAN (DAC),"
    NeurIPS 2023.  (Snake activation)
Nakata et al., "Sidon," arXiv:2509.17052, 2026.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from espnet2.enh.decoder.abs_decoder import AbsDecoder


# ---------------------------------------------------------------------------
# Snake activation  (x + sin²(αx) / α)
# ---------------------------------------------------------------------------

class Snake(nn.Module):
    """Per-channel Snake activation.  α is a learnable parameter (init=1).

    Args:
        channels : number of feature channels C in a [B, C, T] tensor
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha.abs() + 1e-8
        return x + torch.sin(a * x) ** 2 / a


# ---------------------------------------------------------------------------
# HiFi-GAN ResBlock with Snake
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1   = nn.ModuleList()
        self.convs2   = nn.ModuleList()
        self.snakes1  = nn.ModuleList()
        self.snakes2  = nn.ModuleList()
        for d in dilation:
            p = d * (kernel_size - 1) // 2
            self.convs1.append(weight_norm(
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=p)
            ))
            self.convs2.append(weight_norm(
                nn.Conv1d(channels, channels, kernel_size, dilation=1,
                          padding=(kernel_size - 1) // 2)
            ))
            self.snakes1.append(Snake(channels))
            self.snakes2.append(Snake(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for s1, c1, s2, c2 in zip(
            self.snakes1, self.convs1, self.snakes2, self.convs2
        ):
            x = x + c2(s2(c1(s1(x))))
        return x

    def remove_weight_norm(self):
        for c in (*self.convs1, *self.convs2):
            remove_weight_norm(c)


# ---------------------------------------------------------------------------
# Vocoder (AbsDecoder)
# ---------------------------------------------------------------------------

class SpeechCleanerVocoder(AbsDecoder):
    """HiFi-GAN + Snake vocoder: SSL features → 48 kHz waveform.

    Upsampling stack : [8, 5, 4, 3, 2]  →  960× total
    SSL frame rate   : 50 Hz
    Output SR        : 48 000 Hz

    Args:
        input_dim                  : SSL feature dim (1024 for w2v-BERT 2.0)
        upsample_initial_channels  : channels after pre-conv (default 512)
        upsample_rates             : list of transpose-conv strides
        upsample_kernel_sizes      : must satisfy k >= r for each (k,r) pair
        resblock_kernel_sizes      : MRF kernel sizes
        resblock_dilation_sizes    : MRF dilation patterns
    """

    def __init__(
        self,
        input_dim: int = 1024,
        upsample_initial_channels: int = 512,
        upsample_rates: List[int] = [8, 5, 4, 3, 2],
        upsample_kernel_sizes: List[int] = [16, 10, 8, 6, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [
            [1, 3, 5], [1, 3, 5], [1, 3, 5]
        ],
    ):
        super().__init__()

        self.num_kernels  = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        ch = upsample_initial_channels
        self.conv_pre = weight_norm(
            nn.Conv1d(input_dim, ch, kernel_size=7, padding=3)
        )

        self.ups        = nn.ModuleList()
        self.snakes_up  = nn.ModuleList()
        self.resblocks  = nn.ModuleList()

        for u, k in zip(upsample_rates, upsample_kernel_sizes):
            self.snakes_up.append(Snake(ch))
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(
                    ch, ch // 2,
                    kernel_size=k, stride=u,
                    padding=(k - u) // 2,
                )
            ))
            ch //= 2
            for kr, dr in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(_ResBlock(ch, kr, tuple(dr)))

        self.snake_post = Snake(ch)
        self.conv_post  = weight_norm(
            nn.Conv1d(ch, 1, kernel_size=7, padding=3)
        )

    # ------------------------------------------------------------------
    # AbsDecoder interface
    # ------------------------------------------------------------------

    def forward(
        self,
        input: torch.Tensor,           # unused; kept for AbsDecoder compat
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synthesise waveform from SSL features in `additional`.

        The ESPnet enh pipeline passes the separator's `others` dict as
        `additional`.  We read `additional["pred_ssl_feat"]`.

        Returns:
            wav   : [B, T_wav]
            olens : [B] (estimated output lengths)
        """
        if additional is None or "pred_ssl_feat" not in additional:
            raise ValueError(
                "SpeechCleanerVocoder requires 'pred_ssl_feat' in additional. "
                "Ensure SpeechCleanerFeaturePredictor is used upstream."
            )
        ssl_feat = additional["pred_ssl_feat"]   # [B, T_ssl, ssl_dim]
        wav      = self.generate(ssl_feat)        # [B, T_wav]

        # Proportional output length estimate
        up_factor = 1
        for m in self.ups:
            up_factor *= m.stride[0]
        olens = (ilens.float() * (ssl_feat.shape[1] * up_factor) /
                 ilens.float().max()).long().clamp(max=wav.shape[-1])

        return wav, olens

    def generate(self, ssl_feat: torch.Tensor) -> torch.Tensor:
        """Core generation.  Also called directly from inference / GAN model.

        Args:
            ssl_feat : [B, T_ssl, ssl_dim]
        Returns:
            wav      : [B, T_wav]
        """
        x = ssl_feat.transpose(1, 2)   # [B, ssl_dim, T_ssl]
        x = self.conv_pre(x)

        n_res = self.num_kernels
        for i, (snake, up) in enumerate(zip(self.snakes_up, self.ups)):
            x  = up(snake(x))
            xs = None
            for j in range(n_res):
                out = self.resblocks[i * n_res + j](x)
                xs  = out if xs is None else xs + out
            x = xs / n_res

        return torch.tanh(self.conv_post(self.snake_post(x))).squeeze(1)

    def remove_weight_norm(self):
        """Call before inference to remove weight norm for speed."""
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for up in self.ups:
            remove_weight_norm(up)
        for rb in self.resblocks:
            rb.remove_weight_norm()


# =============================================================================
# DAC Decoder wrapper (Sidon-original)
# =============================================================================
class DACVocoder(AbsDecoder):
    """DAC Decoder: SSL features → 48 kHz waveform.

    This is the vocoder used in the original Sidon implementation.
    Sidon config: input_channel=1024, channels=1536, rates=[8,5,4,3,2]

    Requires: pip install descript-audio-codec
    """

    def __init__(
        self,
        input_dim: int = 1024,
        channels: int = 1536,
        rates: List[int] = [8, 5, 4, 3, 2],
        d_out: int = 1,
    ):
        super().__init__()
        try:
            import dac
            self.decoder = dac.model.dac.Decoder(
                input_channel=input_dim,
                channels=channels,
                rates=rates,
                d_out=d_out,
            )
        except ImportError:
            raise ImportError(
                "descript-audio-codec required: "
                "pip install descript-audio-codec"
            )

    def generate(self, ssl_feat: torch.Tensor) -> torch.Tensor:
        """ssl_feat: [B, T_ssl, C] → wav: [B, T_wav]"""
        x = ssl_feat.transpose(1, 2)          # [B, C, T_ssl]
        wav = self.decoder(x)                 # [B, 1, T_wav]
        return wav.squeeze(1)                 # [B, T_wav]

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = (additional or {}).get("pred_ssl_feat", input)
        wav  = self.generate(feat)
        return wav, ilens
