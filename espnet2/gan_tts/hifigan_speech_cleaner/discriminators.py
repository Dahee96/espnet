"""HiFi-GAN discriminators for Speech Cleaner vocoder training.

Placed under espnet2/gan_tts/hifigan_speech_cleaner/ so that the
existing espnet2/gan_tts/hifigan/ code is NEVER modified.

Contains:
  MultiPeriodDiscriminator  (MPD)
  MultiScaleDiscriminator   (MSD)

Used only by SpeechCleanerVocoderModel (espnet_model_speech_cleaner_vocoder.py)
during Stage 2/3 GAN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm


class _PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1,    32,   (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32,   128,  (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128,  512,  (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512,  1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1,           padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        fmap = []
        B, T = x.shape
        if T % self.period != 0:
            x = F.pad(x.unsqueeze(1),
                      (0, self.period - T % self.period),
                      mode="reflect").squeeze(1)
        x = x.view(B, 1, -1, self.period)
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    """MPD with periods [2, 3, 5, 7, 11]."""

    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [_PeriodDiscriminator(p) for p in periods]
        )

    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        r_outs, f_outs, r_fmaps, f_fmaps = [], [], [], []
        for d in self.discriminators:
            ro, rf = d(real);  fo, ff = d(fake)
            r_outs.append(ro); f_outs.append(fo)
            r_fmaps.append(rf); f_fmaps.append(ff)
        return r_outs, f_outs, r_fmaps, f_fmaps


class _ScaleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1,    128,  15, 1,  padding=7)),
            norm(nn.Conv1d(128,  128,  41, 2,  groups=4,  padding=20)),
            norm(nn.Conv1d(128,  256,  41, 2,  groups=16, padding=20)),
            norm(nn.Conv1d(256,  512,  41, 4,  groups=16, padding=20)),
            norm(nn.Conv1d(512,  1024, 41, 4,  groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1,  groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5,  1,  padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor):
        fmap = []
        x = x.unsqueeze(1)
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1);  fmap.append(x)
        x = self.conv_post(x);  fmap.append(x)
        return x.squeeze(1), fmap


class MultiScaleDiscriminator(nn.Module):
    """MSD at 3 scales (original, ×2 pooled, ×4 pooled)."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            _ScaleDiscriminator(use_spectral_norm=False),
            _ScaleDiscriminator(),
            _ScaleDiscriminator(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        r_outs, f_outs, r_fmaps, f_fmaps = [], [], [], []
        r, f = real, fake
        for i, d in enumerate(self.discriminators):
            if i > 0:
                r = self.meanpools[i - 1](r.unsqueeze(1)).squeeze(1)
                f = self.meanpools[i - 1](f.unsqueeze(1)).squeeze(1)
            ro, rf = d(r);  fo, ff = d(f)
            r_outs.append(ro); f_outs.append(fo)
            r_fmaps.append(rf); f_fmaps.append(ff)
        return r_outs, f_outs, r_fmaps, f_fmaps
