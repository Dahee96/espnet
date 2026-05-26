"""Speech Cleaner: unified speech restoration model for ESPnet2.

Sidon-faithful implementation.

Key design decisions
--------------------
1. W2vBert2Encoder:
   - teacher: frozen, num_hidden_layers=8, no LoRA, eval mode
   - student: num_hidden_layers=8, LoRA via peft.inject_adapter_in_model
     (target_modules=["output_dense"], bias="lora_only", r=64, alpha=16)
   - forward()               : student(**noisy_ssl_inputs).last_hidden_state
   - extract_clean_features() : teacher(**clean_ssl_inputs).last_hidden_state
   - ssl_inputs dict built in collate_fn (40-sample padding, Sidon-faithful)
     → no CPU numpy round-trip at forward time

2. max_duration crop: done in _FPCollateFn/_GANCollateFn BEFORE padding.

3. SpeechCleanerFPModel: MSELoss identical to Sidon. No in-forward cropping.

4. SpeechCleanerVocoderModel: No in-forward cropping.

References
----------
Nakata et al., "Sidon," arXiv:2509.17052, 2026.
Kong et al., "HiFi-GAN," NeurIPS 2020.
Kumar et al., "DAC," NeurIPS 2023.
"""

from __future__ import annotations

import abc
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF

try:
    import audiotools
    from audiotools import AudioSignal
    from dac.nn.loss import MelSpectrogramLoss, MultiScaleSTFTLoss
    from dac.nn.loss import L1Loss as DACWavLoss
    _DAC_AVAILABLE = True
except ImportError:
    _DAC_AVAILABLE = False

from torch.nn.utils import remove_weight_norm, weight_norm
from typeguard import typechecked
from espnet2.torch_utils.device_funcs import force_gatherable

from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel

logger = logging.getLogger(__name__)


# =============================================================================
# 1. SSL Encoder abstraction
# =============================================================================

class AbsSSLEncoder(nn.Module, abc.ABC):
    """Abstract base for SSL encoders used in Speech Cleaner.

    forward() and extract_clean_features() both receive a preprocessed
    ssl_inputs dict (built in collate_fn), NOT raw waveforms.
    For W2vBert2Encoder: {"input_features": ..., "attention_mask": ...}
    For XeusEncoder:     {"waveform": ..., "ilens": ...}
    """

    @property
    @abc.abstractmethod
    def ssl_dim(self) -> int: ...

    @property
    @abc.abstractmethod
    def num_layers(self) -> int: ...

    @abc.abstractmethod
    def forward(
        self,
        ssl_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, OrderedDict]: ...

    @torch.no_grad()
    @abc.abstractmethod
    def extract_clean_features(
        self,
        ssl_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Concrete SSL encoder: w2v-BERT 2.0  (peft LoRA, Sidon-faithful)
# ---------------------------------------------------------------------------

class W2vBert2Encoder(AbsSSLEncoder):
    """w2v-BERT 2.0 — teacher/student pair with peft LoRA.

    Exactly mirrors Sidon FeaturePredictorLightningModule:

        student = Wav2Vec2BertModel(num_hidden_layers=8, layerdrop=0.0).train()
        teacher = Wav2Vec2BertModel(num_hidden_layers=8).eval()
        adapter = LoraConfig(r=64, alpha=16, dropout=0.1,
                             bias="lora_only",
                             target_modules=["output_dense"])
        student = inject_adapter_in_model(adapter, student)
        teacher.requires_grad_(False)

        teacher_feat = teacher(**clean_ssl_inputs).last_hidden_state  # no_grad
        student_feat = student(**noisy_ssl_inputs).last_hidden_state
        loss = MSELoss(student_feat, teacher_feat)
    """

    def __init__(
        self,
        model_tag: str = "facebook/w2v-bert-2.0",
        target_layer: int = 8,
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        input_sr: int = 16000,
        freeze_base: bool = True,
        use_flash_attention: bool = False,
        use_bf16: bool = False,
    ):
        super().__init__()
        try:
            from transformers import Wav2Vec2BertModel
        except ImportError as e:
            raise ImportError("transformers >= 4.40 required") from e

        try:
            from peft import LoraConfig, inject_adapter_in_model
        except ImportError as e:
            raise ImportError(
                "peft required: pip install 'peft==0.11.0'"
            ) from e

        self._target_layer = target_layer
        self.input_sr      = input_sr

        hf_kwargs = dict(
            num_hidden_layers=target_layer,
            layerdrop=0.0,
            attn_implementation="eager",
        )
        if use_bf16:
            hf_kwargs["torch_dtype"] = torch.bfloat16

        # ── Teacher: frozen, no LoRA ──────────────────────────────────────
        self.teacher = Wav2Vec2BertModel.from_pretrained(model_tag, **hf_kwargs)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # ── Student: same arch + peft LoRA ───────────────────────────────
        self.student = Wav2Vec2BertModel.from_pretrained(model_tag, **hf_kwargs)

        if freeze_base:
            for p in self.student.parameters():
                p.requires_grad = False

        adapter_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_rank,
            bias="lora_only",                  # Sidon: bias="lora_only"
            target_modules=["output_dense"],   # Sidon: ffn1+ffn2 output_dense
        )
        self.student = inject_adapter_in_model(adapter_config, self.student)

        hidden_dim       = self.student.config.hidden_size   # 1024
        self._ssl_dim    = hidden_dim
        self._num_layers = len(self.student.encoder.layers)  # 8

        logger.info(
            "W2vBert2Encoder: teacher/student loaded (num_hidden_layers=%d), "
            "peft LoRA r=%d alpha=%d bias=lora_only on output_dense",
            target_layer, lora_rank, lora_alpha,
        )

    @property
    def ssl_dim(self) -> int:
        return self._ssl_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def forward(
        self,
        ssl_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, OrderedDict]:
        """Student forward on noisy ssl_inputs."""
        feat   = self.student(**ssl_inputs).last_hidden_state
        others = OrderedDict(pred_ssl_feat=feat)
        return feat, others

    @torch.no_grad()
    def extract_clean_features(
        self,
        ssl_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Teacher forward on clean ssl_inputs (no gradient)."""
        return self.teacher(**ssl_inputs).last_hidden_state


# ---------------------------------------------------------------------------
# Concrete SSL encoder: XEUS
# ---------------------------------------------------------------------------
class XeusEncoder(AbsSSLEncoder):
    """XEUS (ESPnet E-Branchformer) teacher/student with peft LoRA.
    Takes raw waveform: ssl_inputs = {"waveform": [B,T], "ilens": [B]}
    """
    def __init__(
        self,
        model_tag: str = "xeus_checkpoint",
        target_layer: int = 10,
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        input_sr: int = 16000,
        freeze_base: bool = True,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self._target_layer = target_layer
        self.input_sr      = input_sr

        try:
            from espnet2.tasks.ssl import SSLTask
        except ImportError as e:
            raise ImportError("ESPnet SSLTask required for XeusEncoder") from e

        import os
        ckpt_path   = os.path.join(model_tag, "model", "xeus_checkpoint_new.pth")
        config_path = os.path.join(model_tag, "model", "config.yaml")
        assert os.path.exists(ckpt_path),   f"XEUS checkpoint not found: {ckpt_path}"
        assert os.path.exists(config_path), f"XEUS config not found: {config_path}"

        xeus_t, _ = SSLTask.build_model_from_file(config_path, ckpt_path, "cpu")
        xeus_s, _ = SSLTask.build_model_from_file(config_path, ckpt_path, "cpu")
        self.teacher = xeus_t
        self.student = xeus_s

        # Teacher: fully frozen
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # Student base: freeze all first
        if freeze_base:
            for p in self.student.parameters():
                p.requires_grad = False

        # Apply LoRA ONLY to encoders[:target_layer]
        # encoders[target_layer:] remain frozen with no LoRA adapters
        try:
            from peft import LoraConfig, inject_adapter_in_model
            adapter_config = LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                r=lora_rank,
                bias="lora_only",
                target_modules=["w_2"],
            )
            total_layers = len(self.student.encoder.encoders)
            for i, layer in enumerate(self.student.encoder.encoders):
                if i < target_layer:
                    inject_adapter_in_model(adapter_config, layer)
                else:
                    # Ensure layers beyond target_layer are fully frozen
                    for p in layer.parameters():
                        p.requires_grad = False

            trainable = sum(
                p.numel() for p in self.student.parameters() if p.requires_grad
            )
            logger.info(
                "XeusEncoder: LoRA applied to encoders[0:%d] of %d total layers, "
                "trainable=%.2fM (r=%d alpha=%d)",
                target_layer, total_layers, trainable / 1e6, lora_rank, lora_alpha,
            )
        except ImportError:
            logger.warning("peft not available; XeusEncoder has no LoRA adapters")

        self._ssl_dim    = 1024
        self._num_layers = target_layer  # only target_layer layers are used

    @property
    def ssl_dim(self) -> int:
        return self._ssl_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def _run(
        self,
        model,
        ssl_inputs: Dict[str, torch.Tensor],
        use_multilayer: bool = False,
    ):
        wav   = ssl_inputs["waveform"]
        ilens = ssl_inputs["ilens"]
        model = model.to(wav.device)

        # inference_encode returns (final_out, layer_list, lengths)
        final_out, layer_list, _ = model.inference_encode(
            wav, ilens, use_mask=False, use_final_output=True,
        )
        if use_multilayer:
            # Return only the first target_layer outputs
            return layer_list[:self._target_layer]
        else:
            # Use the target_layer-th output (0-indexed → index target_layer-1)
            # layer_list[i] is the output of encoders[i]
            return layer_list[self._target_layer - 1]

    def forward(
        self,
        ssl_inputs: Dict[str, torch.Tensor],
        use_multilayer: bool = False,
    ) -> Tuple[torch.Tensor, OrderedDict]:
        result = self._run(self.student, ssl_inputs, use_multilayer)
        if use_multilayer:
            last_feat = result[-1]
            others = OrderedDict(pred_ssl_feat=last_feat, all_feats=result)
        else:
            last_feat = result
            others = OrderedDict(pred_ssl_feat=last_feat)
        return last_feat, others

    @torch.no_grad()
    def extract_clean_features(
        self,
        ssl_inputs: Dict[str, torch.Tensor],
        use_multilayer: bool = False,
    ):
        return self._run(self.teacher, ssl_inputs, use_multilayer)

class WavLMEncoder(AbsSSLEncoder):
    """WavLM — teacher/student + peft LoRA on output_dense."""

    def __init__(
        self,
        model_tag: str = "microsoft/wavlm-large",
        target_layer: int = 6,
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        input_sr: int = 16000,
        freeze_base: bool = True,
    ):
        super().__init__()
        try:
            from transformers import WavLMModel
        except ImportError as e:
            raise ImportError("transformers required") from e

        self._target_layer = target_layer
        self.input_sr      = input_sr

        hf_kwargs = dict(num_hidden_layers=target_layer)

        self.teacher = WavLMModel.from_pretrained(model_tag, **hf_kwargs)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.student = WavLMModel.from_pretrained(model_tag, **hf_kwargs)
        if freeze_base:
            for p in self.student.parameters():
                p.requires_grad = False

        try:
            from peft import LoraConfig, inject_adapter_in_model
            adapter_config = LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                r=lora_rank,
                bias="lora_only",
                target_modules=["output_dense"],
            )
            self.student = inject_adapter_in_model(adapter_config, self.student)
        except ImportError:
            logger.warning("peft not available; WavLMEncoder has no LoRA adapters")

        self._ssl_dim    = self.student.config.hidden_size
        self._num_layers = len(self.student.encoder.layers)

    @property
    def ssl_dim(self) -> int:
        return self._ssl_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def forward(
        self,
        ssl_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, OrderedDict]:
        feat   = self.student(**ssl_inputs).last_hidden_state
        others = OrderedDict(pred_ssl_feat=feat)
        return feat, others

    @torch.no_grad()
    def extract_clean_features(
        self,
        ssl_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.teacher(**ssl_inputs).last_hidden_state


# =============================================================================
# 2. Vocoder (HiFi-GAN + Snake)
# =============================================================================

class Snake(nn.Module):
    """Snake activation: x + sin²(α·x) / α  (DAC / BigVGAN style)."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha.abs() + 1e-8
        return x + torch.sin(a * x) ** 2 / a


class _ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: Tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs1  = nn.ModuleList()
        self.convs2  = nn.ModuleList()
        self.snakes1 = nn.ModuleList()
        self.snakes2 = nn.ModuleList()
        for d in dilation:
            p = d * (kernel_size - 1) // 2
            self.convs1.append(weight_norm(
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=p)
            ))
            self.convs2.append(weight_norm(
                nn.Conv1d(channels, channels, kernel_size,
                          dilation=1, padding=(kernel_size - 1) // 2)
            ))
            self.snakes1.append(Snake(channels))
            self.snakes2.append(Snake(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for s1, c1, s2, c2 in zip(self.snakes1, self.convs1,
                                    self.snakes2, self.convs2):
            x = x + c2(s2(c1(s1(x))))
        return x

    def remove_weight_norm(self):
        for c in (*self.convs1, *self.convs2):
            remove_weight_norm(c)


class SpeechCleanerVocoder(nn.Module):
    """HiFi-GAN + Snake: SSL features → 48 kHz waveform.
    Upsampling: [8, 5, 4, 3, 2] → 960× (50 Hz SSL → 48 kHz)
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
        self.num_kernels    = len(resblock_kernel_sizes)
        self.num_upsamples  = len(upsample_rates)
        self.total_upsample = 1
        for r in upsample_rates:
            self.total_upsample *= r

        ch = upsample_initial_channels
        self.conv_pre  = weight_norm(nn.Conv1d(input_dim, ch, 7, padding=3))
        self.ups       = nn.ModuleList()
        self.snakes_up = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for u, k in zip(upsample_rates, upsample_kernel_sizes):
            self.snakes_up.append(Snake(ch))
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(ch, ch // 2, k, u, padding=(k - u) // 2)
            ))
            ch //= 2
            for kr, dr in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(_ResBlock(ch, kr, tuple(dr)))

        self.snake_post = Snake(ch)
        self.conv_post  = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

    def generate(self, ssl_feat: torch.Tensor) -> torch.Tensor:
        """[B, T_ssl, D] → [B, T_wav]"""
        x = ssl_feat.transpose(1, 2)
        x = self.conv_pre(x)
        n = self.num_kernels
        for i, (snake, up) in enumerate(zip(self.snakes_up, self.ups)):
            x  = up(snake(x))
            xs = None
            for j in range(n):
                out = self.resblocks[i * n + j](x)
                xs  = out if xs is None else xs + out
            x = xs / n
        return torch.tanh(self.conv_post(self.snake_post(x))).squeeze(1)

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for up in self.ups:
            remove_weight_norm(up)
        for rb in self.resblocks:
            rb.remove_weight_norm()


# =============================================================================
# 3. Loss functions
# =============================================================================

class SpeechCleanerSSLFeatureLoss(nn.Module):
    """MSE — identical to Sidon's nn.MSELoss(reduction='mean').
    Trims to shorter T to handle ±1 frame differences.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        T = min(pred.shape[1], target.shape[1])
        return F.mse_loss(pred[:, :T], target[:, :T], reduction=self.reduction)


class SpeechCleanerMelLoss(nn.Module):
    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self._sr = sample_rate
        if _DAC_AVAILABLE:
            self._stft_loss = MultiScaleSTFTLoss(window_lengths=[2048, 512])
            self._mel_loss  = MelSpectrogramLoss(
                n_mels=[5, 10, 20, 40, 80, 160, 320],
                window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
                mel_fmin=[0]*7, mel_fmax=[None]*7,
                pow=1.0, clamp_eps=1e-5, mag_weight=0.0,
            )
            self._wav_loss = DACWavLoss()
        else:
            self._fallback_mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=2048, hop_length=480,
                win_length=2048, n_mels=128, power=1.0,
                norm="slaney", mel_scale="slaney",
            )

    def _to_signal(self, wav: torch.Tensor):
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        return AudioSignal(wav, self._sr)

    def forward(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
        T = min(pred_wav.shape[-1], target_wav.shape[-1])
        pred_wav   = pred_wav[..., :T]
        target_wav = target_wav[..., :T]
        if _DAC_AVAILABLE:
            pred_sig   = self._to_signal(pred_wav)
            target_sig = self._to_signal(target_wav)
            return (self._stft_loss(pred_sig, target_sig)
                    + self._mel_loss(pred_sig, target_sig)
                    + self._wav_loss(pred_sig, target_sig))
        else:
            fb = self._fallback_mel.to(pred_wav.device)
            pm = torch.log(fb(pred_wav).clamp(1e-5))
            tm = torch.log(fb(target_wav).clamp(1e-5))
            Tm = min(pm.shape[-1], tm.shape[-1])
            return F.l1_loss(pm[..., :Tm], tm[..., :Tm])


class SpeechCleanerGeneratorLoss(nn.Module):
    def forward(self, fake_outs: list) -> torch.Tensor:
        return sum(torch.mean((fo - 1.0) ** 2) for fo in fake_outs) / max(len(fake_outs), 1)


class SpeechCleanerDiscriminatorLoss(nn.Module):
    def forward(self, real_outs: list, fake_outs: list) -> torch.Tensor:
        loss = sum(
            torch.mean((ro - 1.0) ** 2) + torch.mean(fo ** 2)
            for ro, fo in zip(real_outs, fake_outs)
        )
        return loss / max(len(real_outs), 1)


class SpeechCleanerFeatureMatchLoss(nn.Module):
    def __init__(self, weight: float = 2.0):
        super().__init__()
        self.weight = weight

    def forward(self, real_fmaps: list, fake_fmaps: list) -> torch.Tensor:
        loss, n = 0.0, 0
        for r_fmap, f_fmap in zip(real_fmaps, fake_fmaps):
            for rf, ff in zip(r_fmap, f_fmap):
                loss += F.l1_loss(ff, rf.detach())
                n    += 1
        return self.weight * loss / max(n, 1)


# =============================================================================
# 4. ESPnet model — Stage 1 (Feature Predictor)
# =============================================================================

class SpeechCleanerFPModel(AbsESPnetModel):
    """Stage 1: predict clean SSL features from noisy waveform.

    Batch receives:
        noisy_speech          : [B, T] padded waveform (for ESPnet compatibility)
        noisy_speech_lengths  : [B]
        speech_ref1           : [B, T] padded waveform
        speech_ref1_lengths   : [B]
        noisy_speech_ssl      : dict {input_features, attention_mask}  ← built in collate_fn
        speech_ref1_ssl       : dict {input_features, attention_mask}  ← built in collate_fn

    Sidon-faithful:
        loss = MSELoss(student(**noisy_ssl), teacher(**clean_ssl))
    """

    @typechecked
    def __init__(self, ssl_encoder: AbsSSLEncoder):
        super().__init__()
        self.ssl_encoder = ssl_encoder
        self.loss_fn     = SpeechCleanerSSLFeatureLoss()

    def forward(
        self,
        noisy_speech: torch.Tensor,
        noisy_speech_lengths: torch.Tensor,
        speech_ref1: torch.Tensor,
        speech_ref1_lengths: torch.Tensor,
        noisy_speech_ssl: Optional[Dict[str, torch.Tensor]] = None,
        speech_ref1_ssl: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:

        if noisy_speech_ssl is None or speech_ref1_ssl is None:
            raise ValueError(
                "noisy_speech_ssl and speech_ref1_ssl must be provided. "
                "Ensure _FPCollateFn is adding ssl_inputs to the batch."
            )

        B = noisy_speech.shape[0]

        # Move ssl dicts to the same device as the waveform
        device = noisy_speech.device
        noisy_ssl = {k: v.to(device) for k, v in noisy_speech_ssl.items()}
        clean_ssl = {k: v.to(device) for k, v in speech_ref1_ssl.items()}

        pred_feat, _ = self.ssl_encoder(noisy_ssl)
        target_feat  = self.ssl_encoder.extract_clean_features(clean_ssl)

        loss  = self.loss_fn(pred_feat, target_feat)
        # print(f"[DEBUG] noisy zeros: {(noisy_speech==0).float().mean():.3f}")
        # print(f"[DEBUG] pred_feat mean: {pred_feat.mean():.6f}")
        # print(f"[DEBUG] target_feat mean: {target_feat.mean():.6f}")
        # print(f"[DEBUG] loss: {loss.item():.8f}")
        stats = dict(loss=loss.detach(), loss_ssl=loss.detach())
        loss, stats, weight = force_gatherable((loss, stats, B), loss.device)
        return loss, stats, weight

    def collect_feats(self, **kwargs) -> Dict:
        return {}


# =============================================================================
# 5. ESPnet model — Stage 2/3 (Vocoder GAN)
# =============================================================================

class SpeechCleanerVocoderModel(AbsGANESPnetModel):
    """Stage 2/3: vocoder GAN (GANTrainer, two optimizers).

    Uses same ssl_inputs dict pattern as FPModel.
    speech_ref1 is the 48 kHz reference for reconstruction loss.
    """

    @typechecked
    def __init__(
        self,
        ssl_encoder: AbsSSLEncoder,
        vocoder: nn.Module,
        use_predicted_feat: bool = False,
        mel_loss_weight: float = 15.0,
        adv_loss_weight: float = 2.0,
        fm_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.ssl_encoder        = ssl_encoder
        self.vocoder            = vocoder
        self.use_predicted_feat = use_predicted_feat

        from espnet2.gan_tts.hifigan_speech_cleaner.discriminators import (
            MultiPeriodDiscriminator,
            MultiScaleDiscriminator,
        )
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

        self.mel_loss = SpeechCleanerMelLoss()
        self.adv_G    = SpeechCleanerGeneratorLoss()
        self.adv_D    = SpeechCleanerDiscriminatorLoss()
        self.fm_loss  = SpeechCleanerFeatureMatchLoss(weight=fm_loss_weight)

        self.mel_w = mel_loss_weight
        self.adv_w = adv_loss_weight

    def forward(
        self,
        noisy_speech: torch.Tensor,
        noisy_speech_lengths: torch.Tensor,
        speech_ref1: torch.Tensor,
        speech_ref1_lengths: torch.Tensor,
        noisy_speech_ssl: Optional[Dict[str, torch.Tensor]] = None,
        speech_ref1_ssl: Optional[Dict[str, torch.Tensor]] = None,
        forward_generator: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        device = noisy_speech.device
        noisy_ssl = {k: v.to(device) for k, v in noisy_speech_ssl.items()} \
            if noisy_speech_ssl else None
        clean_ssl = {k: v.to(device) for k, v in speech_ref1_ssl.items()} \
            if speech_ref1_ssl else None

        if forward_generator:
            return self._forward_G(noisy_ssl, clean_ssl, speech_ref1)
        return self._forward_D(noisy_ssl, clean_ssl, speech_ref1)

    def _get_ssl_feat(self, noisy_ssl, clean_ssl) -> torch.Tensor:
        if self.use_predicted_feat:
            feat, _ = self.ssl_encoder(noisy_ssl)
        else:
            feat = self.ssl_encoder.extract_clean_features(clean_ssl)
        return feat

    @staticmethod
    def _trim(a: torch.Tensor, b: torch.Tensor):
        T = min(a.shape[-1], b.shape[-1])
        return a[..., :T], b[..., :T]

    def _forward_G(self, noisy_ssl, clean_ssl, speech_ref1) -> Dict:
        ssl_feat           = self._get_ssl_feat(noisy_ssl, clean_ssl)
        fake_wav           = self.vocoder.generate(ssl_feat)
        real_wav, fake_wav = self._trim(speech_ref1, fake_wav)

        r_outs_mpd, f_outs_mpd, r_fmaps_mpd, f_fmaps_mpd = self.mpd(
            real_wav.detach(), fake_wav
        )
        r_outs_msd, f_outs_msd, r_fmaps_msd, f_fmaps_msd = self.msd(
            real_wav.detach(), fake_wav
        )

        l_mel  = self.mel_loss(fake_wav, real_wav)
        l_adv  = self.adv_G(f_outs_mpd + f_outs_msd)
        l_fm   = (
            self.fm_loss(r_fmaps_mpd, f_fmaps_mpd)
            + self.fm_loss(r_fmaps_msd, f_fmaps_msd)
        )
        loss_G = self.mel_w * l_mel + self.adv_w * l_adv + l_fm
        B = speech_ref1.shape[0]
        loss_G, stats_G, weight = force_gatherable(
            (loss_G,
             dict(loss_G=loss_G.detach(), loss_mel=l_mel.detach(),
                  loss_adv_G=l_adv.detach(), loss_fm=l_fm.detach()),
             B),
            loss_G.device,
        )
        return dict(loss=loss_G, stats=stats_G, weight=weight, optim_idx=0)

    def _forward_D(self, noisy_ssl, clean_ssl, speech_ref1) -> Dict:
        with torch.no_grad():
            ssl_feat = self._get_ssl_feat(noisy_ssl, clean_ssl)
            fake_wav = self.vocoder.generate(ssl_feat)

        real_wav, fake_wav = self._trim(speech_ref1, fake_wav)

        r_outs_mpd, f_outs_mpd, _, _ = self.mpd(real_wav, fake_wav.detach())
        r_outs_msd, f_outs_msd, _, _ = self.msd(real_wav, fake_wav.detach())

        loss_D = (
            self.adv_D(r_outs_mpd, f_outs_mpd)
            + self.adv_D(r_outs_msd, f_outs_msd)
        )
        B = speech_ref1.shape[0]
        loss_D, stats_D, weight = force_gatherable(
            (loss_D, dict(loss_D=loss_D.detach()), B),
            loss_D.device,
        )
        return dict(loss=loss_D, stats=stats_D, weight=weight, optim_idx=1)

    def collect_feats(self, **kwargs) -> Dict:
        return {}