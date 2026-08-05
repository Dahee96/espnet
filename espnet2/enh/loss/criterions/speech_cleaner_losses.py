"""Speech Cleaner loss functions.

Stage 1 – Feature Predictor
    SpeechCleanerSSLFeatureLoss  :  MSE on SSL feature space

Stage 2/3 – Vocoder GAN
    SpeechCleanerMelLoss         :  log-mel L1 (48 kHz)
    SpeechCleanerGeneratorLoss   :  least-squares GAN generator
    SpeechCleanerDiscriminatorLoss: least-squares GAN discriminator
    SpeechCleanerFeatureMatchLoss:  L1 on discriminator feature maps

Design rule
-----------
NO existing espnet2/enh/loss/criterions/*.py is modified.
TimeDomainMSE and FrequencyDomainL1 already exist; we add NEW classes
that operate in the SSL feature domain (float tensors, any shape) and
at a fixed 48 kHz mel configuration matching the speech cleaner vocoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ---------------------------------------------------------------------------
# Stage 1 — SSL feature MSE
# ---------------------------------------------------------------------------

class SpeechCleanerSSLFeatureLoss(nn.Module):
    """MSE between predicted and target SSL feature tensors.

    Both inputs are [B, T_ssl, D].  If T_ssl differs by ±1 frame due to
    padding, we trim to the shorter dimension before computing the loss.

    Args:
        reduction : 'mean' (default) or 'sum'
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,    # [B, T_ssl, D]
        target: torch.Tensor,  # [B, T_ssl', D]
    ) -> torch.Tensor:
        T = min(pred.shape[1], target.shape[1])
        return F.mse_loss(pred[:, :T], target[:, :T], reduction=self.reduction)


# ---------------------------------------------------------------------------
# Stage 2/3 — Mel spectrogram L1 loss (48 kHz)
# ---------------------------------------------------------------------------

class SpeechCleanerMelLoss(nn.Module):
    """Log-mel L1 loss tuned for 48 kHz vocoder output.

    Args:
        sample_rate : output sample rate (default 48000)
        n_fft       : FFT size (default 2048)
        hop_length  : hop length (default 480 = 10 ms at 48 kHz)
        win_length  : window length (default 2048)
        n_mels      : mel bands (default 128)
        fmin        : min frequency (default 0)
        fmax        : max frequency (default None → sr/2)
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 480,
        win_length: int = 2048,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: float = None,
    ):
        super().__init__()
        self._mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=1.0,
            norm="slaney",
            mel_scale="slaney",
        )

    def _mel(self, wav: torch.Tensor) -> torch.Tensor:
        if self._mel_fn.mel_scale.fb.device != wav.device:
            self._mel_fn = self._mel_fn.to(wav.device)
        return torch.log(self._mel_fn(wav).clamp(min=1e-5))

    def forward(
        self,
        pred_wav: torch.Tensor,   # [B, T_wav]
        target_wav: torch.Tensor, # [B, T_wav]
    ) -> torch.Tensor:
        T   = min(pred_wav.shape[-1], target_wav.shape[-1])
        p_m = self._mel(pred_wav[..., :T])
        t_m = self._mel(target_wav[..., :T])
        Tm  = min(p_m.shape[-1], t_m.shape[-1])
        return F.l1_loss(p_m[..., :Tm], t_m[..., :Tm])


# ---------------------------------------------------------------------------
# Stage 2/3 — GAN losses
# ---------------------------------------------------------------------------

class SpeechCleanerGeneratorLoss(nn.Module):
    """Least-squares GAN loss for the generator.

    loss = E[ (D(G(x)) − 1)² ]
    """

    def forward(self, fake_outs: list) -> torch.Tensor:
        loss = sum(torch.mean((fo - 1.0) ** 2) for fo in fake_outs)
        return loss / max(len(fake_outs), 1)


class SpeechCleanerDiscriminatorLoss(nn.Module):
    """Least-squares GAN loss for the discriminator.

    loss = E[(D(real)−1)²] + E[D(fake)²]
    """

    def forward(self, real_outs: list, fake_outs: list) -> torch.Tensor:
        loss = sum(
            torch.mean((ro - 1.0) ** 2) + torch.mean(fo ** 2)
            for ro, fo in zip(real_outs, fake_outs)
        )
        return loss / max(len(real_outs), 1)


class SpeechCleanerFeatureMatchLoss(nn.Module):
    """L1 feature matching loss on discriminator intermediate activations.

    Args:
        weight : scalar multiplier (default 2.0, following HiFi-GAN paper)
    """

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
