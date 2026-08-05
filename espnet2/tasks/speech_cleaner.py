"""ESPnet2 task definitions for Speech Cleaner (speech restoration).

On-the-fly degradation pipeline
---------------------------------
Data flow (FP stage):
  data/train_fp/wav.scp  →  clean audio (any SR, loaded by ESPnet loader)
  collate_fn:
    1. resample clean to input_sr (16kHz) if needed
    2. apply degrade_waveform() → noisy (16k)
    3. random crop both to max_duration
    4. pad batch
    5. build ssl_inputs dict

Data flow (VOC stage):
  data/train_voc/wav.scp  →  clean 48kHz audio
  collate_fn:
    1. resample 48k→16k → apply degrade → noisy_speech (16k)
    2. keep speech_ref1 at 48k (vocoder reconstruction target)
    3. random crop both (aligned)
    4. pad batch
    5. build ssl_inputs dict

Safety checks in degrade_waveform:
  - packet_loss scales num_chunks to total duration (short utterances → fewer chunks)
  - minimum chunk size enforced (no chunk longer than utterance itself)
  - NaN/Inf replaced, output clamped to [-1, 1]
  - empty output (all-zeros) triggers warning

Worker efficiency:
  - noise_files / rir_files loaded once at collate_fn __init__
  - each DataLoader worker has its own copy (no shared state)
  - torchaudio AudioEffector re-used across calls (pre-instantiated)
  - all operations on CPU tensors → no GPU/CPU sync in workers
"""

import argparse
import math
import os
import random
from io import BytesIO
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF_

from espnet2.tasks.abs_task import AbsTask
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.trainer import Trainer
from espnet2.train.gan_trainer import GANTrainer
from espnet2.utils.types import str2bool, str_or_none

from espnet2.enh.speech_cleaner_model import (
    W2vBert2Encoder, XeusEncoder, WavLMEncoder,
    SpeechCleanerFPModel, SpeechCleanerVocoderModel,
)
from espnet2.enh.decoder.speech_cleaner_vocoder import (
    DACVocoder, SpeechCleanerVocoder,
)

import logging
logger = logging.getLogger(__name__)

SSL_ENCODER_CLASSES = dict(
    w2v_bert2=W2vBert2Encoder,
    xeus=XeusEncoder,
    wavlm=WavLMEncoder,
)
_SEAMLESS_ENCODERS = {"w2v_bert2", "wavlm"}
_WAVEFORM_ENCODERS = {"xeus"}


# =============================================================================
# On-the-fly degradation
# =============================================================================

_SR_CANDIDATES = [8000, 16000, 22050, 24000, 44100, 48000]

# Pre-instantiate codec effectors (MP3, qscale 1=best, 10=worst)
# Lazy init to avoid issues when imported in non-training contexts
_CODEC_EFFECTORS: Optional[List] = None

def _get_codec_effectors() -> List:
    global _CODEC_EFFECTORS
    if _CODEC_EFFECTORS is None:
        _CODEC_EFFECTORS = [
            torchaudio.io.AudioEffector(
                format="mp3",
                codec_config=torchaudio.io.CodecConfig(qscale=q),
            )
            for q in range(1, 11)
        ]
    return _CODEC_EFFECTORS


def _load_file_list(pool_dir: str) -> List[str]:
    if not pool_dir or not os.path.isdir(pool_dir):
        return []
    files = sorted(
        os.path.join(pool_dir, f)
        for f in os.listdir(pool_dir)
        if f.endswith(".wav") or f.endswith(".flac")
    )
    return files


# def _apply_reverb(wav: torch.Tensor, sr: int, rir_files: List[str]) -> torch.Tensor:
#     """Convolve with a randomly selected pre-generated RIR."""
#     if not rir_files:
#         return wav
#     try:
#         rir, rir_sr = torchaudio.load(random.choice(rir_files))
#         rir = rir[0].unsqueeze(0)
#         if rir_sr != sr:
#             rir = AF_.resample(rir, rir_sr, sr)
#         peak = rir.abs().max()
#         if peak > 1e-8:
#             rir = rir / peak
#         # fftconvolve: wav [1,T], rir [1,T'] → [1, T+T'-1], trim to T
#         result = AF_.fftconvolve(wav.unsqueeze(0), rir)[0, :wav.shape[-1]]
#         return result
#     except Exception:
#         return wav

def _apply_reverb(wav: torch.Tensor, sr: int, rir_files: List[str]) -> torch.Tensor:
    """Convolve with a randomly selected pre-generated RIR.

    FIX: cap the RIR's effective length to at most half of the input
    signal's length. Without this, an RIR with RT60 close to its max
    (e.g. 2.0s) convolved onto a short crop (e.g. 2.0s for VOC stage)
    produces a training example almost ENTIRELY dominated by reverberant
    tail, regardless of whether cropping happens before or after
    degradation -- the RT60 distribution itself is simply too long
    relative to short crops. Capping RIR length relative to signal length
    guarantees every training example retains a meaningful dry/clean
    portion, while leaving long-crop (FP, 20s) behavior essentially
    unchanged since the cap (10s) is far longer than any RIR ever drawn.
    """
    if not rir_files:
        return wav
    try:
        rir, rir_sr = torchaudio.load(random.choice(rir_files))
        rir = rir[0].unsqueeze(0)
        if rir_sr != sr:
            rir = AF_.resample(rir, rir_sr, sr)

        # ── FIX: cap RIR tail to at most 50% of the signal length ──────
        max_rir_len = max(int(wav.shape[-1] * 0.5), 1)
        if rir.shape[-1] > max_rir_len:
            rir = rir[..., :max_rir_len]
        # ──────────────────────────────────────────────────────────────

        peak = rir.abs().max()
        if peak > 1e-8:
            rir = rir / peak
        # fftconvolve: wav [1,T], rir [1,T'] → [1, T+T'-1], trim to T
        result = AF_.fftconvolve(wav.unsqueeze(0), rir)[0, :wav.shape[-1]]
        return result
    except Exception:
        return wav

def _apply_noise(wav: torch.Tensor, sr: int, noise_files: List[str]) -> torch.Tensor:
    """Add noise at SNR ~ U(-5, 20) dB."""
    if not noise_files:
        return wav
    try:
        noise, noise_sr = torchaudio.load(random.choice(noise_files))
        noise = noise.mean(dim=0, keepdim=True)   # mono [1, T_noise]
        if noise_sr != sr:
            noise = AF_.resample(noise, noise_sr, sr)
        if noise.shape[-1] == 0:
            return wav
        # Loop noise to match wav length
        reps  = math.ceil(wav.shape[-1] / noise.shape[-1]) + 1
        noise = noise.repeat(1, reps)[..., :wav.shape[-1]]
        snr   = torch.tensor([random.uniform(-5.0, 20.0)])
        return AF_.add_noise(wav.unsqueeze(0), noise, snr)[0]
    except Exception:
        return wav


def _apply_band_limit(wav: torch.Tensor, sr: int) -> torch.Tensor:
    target_sr = random.choice(_SR_CANDIDATES)
    if target_sr == sr:
        return wav
    try:
        down = AF_.resample(wav.unsqueeze(0), sr, target_sr)
        up   = AF_.resample(down, target_sr, sr)
        return up[0, :wav.shape[-1]]
    except Exception:
        return wav


def _apply_clipping(wav: torch.Tensor) -> torch.Tensor:
    lo_q = random.uniform(0.0,  0.10)
    hi_q = random.uniform(0.90, 1.00)
    lo_v = float(torch.quantile(wav, lo_q))
    hi_v = float(torch.quantile(wav, hi_q))
    if lo_v >= hi_v:
        return wav
    return wav.clamp(lo_v, hi_v)


# def _apply_codec(wav: torch.Tensor, sr: int) -> torch.Tensor:
#     """MP3 codec at random quality (qscale 1–10)."""
#     effectors = _get_codec_effectors()
#     effector  = random.choice(effectors)
#     T = wav.shape[-1]
#     try:
#         # AudioEffector expects [T, C]
#         out = effector.apply(wav.unsqueeze(-1), sr)   # [T', 1]
#         out = out.squeeze(-1)                          # [T']
#         if out.shape[-1] >= T:
#             return out[:T]
#         return torch.nn.functional.pad(out, (0, T - out.shape[-1]))
#     except Exception:
#         return wav
def _align_codec_waveform(original: torch.Tensor, codec_applied: torch.Tensor) -> torch.Tensor:
    T_orig = original.shape[-1]
    T_codec = codec_applied.shape[-1]
    if T_orig == T_codec:
        return codec_applied
    if T_orig < T_codec:
        
        best_i, best_mse = 0, float("inf")
        max_shift = min(T_codec - T_orig, 4096)  # 
        for i in range(0, max_shift + 1):
            mse = torch.mean((codec_applied[i:i+T_orig] - original) ** 2).item()
            if mse < best_mse:
                best_mse, best_i = mse, i
        return codec_applied[best_i:best_i+T_orig]
    else:
        pad = T_orig - T_codec
        return F.pad(codec_applied, (0, pad))


def _apply_codec(wav: torch.Tensor, sr: int) -> torch.Tensor:
    effectors = _get_codec_effectors()
    effector  = random.choice(effectors)
    T = wav.shape[-1]
    try:
        out = effector.apply(wav.unsqueeze(-1), sr).squeeze(-1)
        return _align_codec_waveform(wav, out)
    except Exception:
        return wav

def _apply_packet_loss(
    wav: torch.Tensor,
    sr: int,
    loss_rate: float = 0.09,
    min_ms: float = 20.0,
    max_ms: float = 200.0,
) -> torch.Tensor:
    """Zero out random segments (Sidon Section 3.4).

    Safety: num_chunks is scaled to actual duration, and each chunk is
    clamped so it cannot exceed the utterance length.
    """
    T = wav.shape[-1]
    total_s = T / sr
    if total_s <= 0:
        return wav

    # Scale chunk duration to not exceed half the utterance
    effective_max_ms = min(max_ms, total_s * 500)   # max 50% of utterance
    if effective_max_ms < min_ms:
        return wav   # utterance too short for packet loss

    target_zeroed = total_s * loss_rate
    mean_chunk_s  = (min_ms + effective_max_ms) / 2 / 1000.0
    num_chunks    = max(1, round(target_zeroed / mean_chunk_s))

    result = wav.clone()
    for _ in range(num_chunks):
        chunk_s   = random.uniform(min_ms / 1000.0, effective_max_ms / 1000.0)
        max_start = max(0.0, total_s - chunk_s)
        start_s   = random.uniform(0.0, max_start)
        s = int(start_s * sr)
        e = min(T, int((start_s + chunk_s) * sr))
        result[..., s:e] = 0.0
    return result


def degrade_waveform(
    wav: torch.Tensor,
    sr: int,
    noise_files: List[str],
    rir_files: List[str],
    apply_prob: float = 0.5,
) -> torch.Tensor:
    """Apply full Sidon degradation pipeline to a 1D float32 tensor.

    Order (Sidon paper 3.4): reverb → noise → band_limit → clip → codec → packet_loss
    Each applied independently with probability apply_prob (default 0.5).

    Safety guarantees:
      - Input must be 1D, float32, normalized to ~[-1, 1]
      - Output: NaN/Inf replaced with 0.0/±1.0, clamped to [-1, 1]
      - Empty (all-zero) output after degradation logs a warning
    """
    assert wav.dim() == 1, f"degrade_waveform expects 1D tensor, got {wav.shape}"
    w = wav.float().clone()

    fns = [
        lambda x: _apply_reverb(x, sr, rir_files),
        lambda x: _apply_noise(x, sr, noise_files),
        lambda x: _apply_band_limit(x, sr),
        lambda x: _apply_clipping(x),
        lambda x: _apply_codec(x, sr),
        lambda x: _apply_packet_loss(x, sr),
    ]
    for fn in fns:
        if random.random() < apply_prob:
            w = fn(w)
            # Early safety check after each degradation
            if not torch.isfinite(w).all():
                w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=-1.0)

    w = w.clamp(-1.0, 1.0)

    if w.abs().max() < 1e-8:
        logger.debug("degrade_waveform: output is near-silent — returning original")
        return wav.clamp(-1.0, 1.0)

    return w


# =============================================================================
# SSL encoder / task arg helpers
# =============================================================================

def _build_ssl_encoder(args):
    name = getattr(args, "ssl_encoder", "w2v_bert2")
    conf = dict(getattr(args, "ssl_encoder_conf", None) or {})
    conf.setdefault("target_layer",       args.target_layer)
    conf.setdefault("lora_rank",          args.lora_rank)
    conf.setdefault("lora_alpha",         args.lora_alpha)
    conf.setdefault("lora_dropout",       args.lora_dropout)
    conf.setdefault("input_sr",           args.input_sr)
    conf.setdefault("use_flash_attention",
                    getattr(args, "use_flash_attention", True))
    conf.setdefault("multilayer_mode",
                    getattr(args, "multilayer_mode", "low"))
    use_ml = getattr(args, "use_multilayer_loss",
                     getattr(args, "use_multilayer_feat", False))
    conf.setdefault("use_multilayer_loss", use_ml)
    return SSL_ENCODER_CLASSES[name](**conf)


def _add_ssl_args(parser):
    g = parser.add_argument_group("Speech Cleaner — SSL encoder")
    g.add_argument("--ssl_encoder", type=str, default="w2v_bert2",
                   choices=list(SSL_ENCODER_CLASSES))
    g.add_argument("--ssl_encoder_conf", action=_NestedDictAction, default=None)
    g.add_argument("--target_layer",  type=int,   default=8)
    g.add_argument("--lora_rank",     type=int,   default=64)
    g.add_argument("--lora_alpha",    type=int,   default=16)
    g.add_argument("--lora_dropout",  type=float, default=0.1)
    g.add_argument("--max_duration",  type=float, default=20.0)
    g.add_argument("--input_sr",      type=int,   default=16000)
    g.add_argument("--warmup_steps",  type=int,   default=2000)
    g.add_argument("--use_flash_attention", type=str2bool, default=True,
                   help="SDPA patch for w2v-BERT; flash_attn for XEUS.")
    g.add_argument("--use_multilayer_loss", type=str2bool, default=False)
    g.add_argument("--multilayer_mode", type=str, default="low",
                   choices=["low", "up", "all"])
    # On-the-fly degradation
    g.add_argument("--noise_dir", type=str, default="data/noise_pool",
                   help="Directory with noise wav/flac files.")
    g.add_argument("--rir_dir",   type=str, default="data/rir_pool",
                   help="Directory with pre-generated RIR wav files.")
    g.add_argument("--degrade_prob", type=float, default=0.5,
                   help="Per-degradation apply probability (Sidon: 0.5).")
    g.add_argument("--online_degradation", type=str2bool, default=True,
                   help="Generate noisy speech on-the-fly. "
                        "Set False to use pre-degraded paired wav.scps.")
    g.add_argument("--layer_weighting", type=str_or_none, default=None,
                help="uniform | global_learnable | utterance_dynamic | frame_dynamic "
                        "(only used when use_multilayer_feat=True)")


class _NestedDictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, dict):
            setattr(namespace, self.dest, values); return
        d = {}
        for item in (values if isinstance(values, list) else [values]):
            if "=" in item:
                k, v = item.split("=", 1)
                if v.lower() == "true":    v = True
                elif v.lower() == "false": v = False
                else:
                    try:    v = int(v)
                    except:
                        try: v = float(v)
                        except: pass
                d[k] = v
        setattr(namespace, self.dest, d)


# =============================================================================
# Collate functions
# =============================================================================

class _FPCollateFn:
    """Stage 1 collate: on-the-fly degradation → crop → pad → ssl_inputs.

    On-the-fly mode (online_degradation=True):
      wav.scp = clean audio paths.
      Each worker generates: noisy = degrade(clean) at input_sr.
      speech_ref1 = clean (16kHz).

    Pre-degraded mode (online_degradation=False):
      wav.scp = already-paired (noisy, clean).
      Pass through unchanged.

    Sidon repeat: each clean utterance is degraded once per epoch step.
    The 4× repetition in Sidon is achieved by 4 epochs (or 4 workers each
    seeing different random seeds) — NOT by storing 4 copies on disk.
    """

    def __init__(
        self,
        max_samples: int,
        ssl_encoder: str = "w2v_bert2",
        input_sr: int = 16000,
        processor_tag: str = "facebook/w2v-bert-2.0",
        noise_dir: str = "data/noise_pool",
        rir_dir: str = "data/rir_pool",
        degrade_prob: float = 0.5,
        online_degradation: bool = True,
    ):
        self.max_samples        = max_samples
        self.ssl_encoder        = ssl_encoder
        self.input_sr           = input_sr
        self.degrade_prob       = degrade_prob
        self.online_degradation = online_degradation
        self._base = CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

        if ssl_encoder in _SEAMLESS_ENCODERS:
            from transformers import SeamlessM4TFeatureExtractor
            self.processor = SeamlessM4TFeatureExtractor.from_pretrained(processor_tag)
        else:
            self.processor = None

        self.noise_files = _load_file_list(noise_dir)
        self.rir_files   = _load_file_list(rir_dir)

        if online_degradation:
            if not self.noise_files:
                logger.warning("_FPCollateFn: noise_dir=%s has no files — "
                               "noise augmentation skipped", noise_dir)
            if not self.rir_files:
                logger.warning("_FPCollateFn: rir_dir=%s has no files — "
                               "reverberation augmentation skipped", rir_dir)

    def __call__(self, data):
        processed = []
        for key, d in data:
            new_d = dict(d)

            if self.online_degradation:
                # speech_ref1 is the clean signal (loaded from clean wav.scp)
                clean = d.get("speech_ref1", d.get("noisy_speech"))
                if clean is not None:
                    # numpy → torch → degrade → numpy (CommonCollateFn expects numpy)
                    if isinstance(clean, np.ndarray):
                        clean_t = torch.from_numpy(clean.copy()).float()
                    else:
                        clean_t = clean.float()
                    try:
                        noisy_t = degrade_waveform(
                            clean_t, self.input_sr,
                            self.noise_files, self.rir_files,
                            self.degrade_prob,
                        )
                    except Exception:
                        noisy_t = clean_t.clone()
                    # Convert back to numpy for CommonCollateFn
                    new_d["noisy_speech"] = noisy_t.numpy()
                    new_d["speech_ref1"]  = clean_t.numpy()

            # Crop to max_duration (random start for data augmentation)
            noisy = new_d.get("noisy_speech")
            if noisy is not None and isinstance(noisy, np.ndarray):
                if noisy.shape[0] > self.max_samples:
                    T = noisy.shape[0]
                    s = random.randint(0, T - self.max_samples)
                    new_d["noisy_speech"] = noisy[s:s + self.max_samples]
                    if "speech_ref1" in new_d:
                        ref = new_d["speech_ref1"]
                        if ref.shape[0] > self.max_samples:
                            new_d["speech_ref1"] = ref[s:s + self.max_samples]
            elif noisy is not None and isinstance(noisy, torch.Tensor):
                if noisy.shape[0] > self.max_samples:
                    T = noisy.shape[0]
                    s = random.randint(0, T - self.max_samples)
                    new_d["noisy_speech"] = noisy[s:s + self.max_samples].numpy()
                    if "speech_ref1" in new_d:
                        ref = new_d["speech_ref1"]
                        new_d["speech_ref1"] = (ref[s:s + self.max_samples].numpy()
                                                if isinstance(ref, torch.Tensor)
                                                else ref[s:s + self.max_samples])
            processed.append((key, new_d))

        keys, batch = self._base(processed)

        # Build ssl_inputs
        if self.ssl_encoder in _SEAMLESS_ENCODERS:
            noisy_np = self._to_np(batch["noisy_speech"],
                                   batch["noisy_speech_lengths"], pad40=True)
            clean_np = self._to_np(batch["speech_ref1"],
                                   batch["speech_ref1_lengths"], pad40=True)
            batch["noisy_speech_ssl"] = dict(self.processor(
                noisy_np, sampling_rate=self.input_sr,
                return_tensors="pt", padding=True))
            batch["speech_ref1_ssl"] = dict(self.processor(
                clean_np, sampling_rate=self.input_sr,
                return_tensors="pt", padding=True))
        else:
            batch["noisy_speech_ssl"] = {
                "waveform": batch["noisy_speech"],
                "ilens":    batch["noisy_speech_lengths"],
            }
            batch["speech_ref1_ssl"] = {
                "waveform": batch["speech_ref1"],
                "ilens":    batch["speech_ref1_lengths"],
            }
        return keys, batch

    def _to_np(self, wav_tensor, lengths, pad40=False):
        result = []
        for i in range(wav_tensor.shape[0]):
            w = wav_tensor[i, :lengths[i]].float().numpy()
            if pad40:
                w = np.pad(w, (40, 40), mode="constant")
            result.append(w)
        return result


class _GANCollateFn:
    """Stage 2/3 collate: crop -> on-the-fly degradation -> pad -> ssl_inputs.

    Alignment: crop indices are sampled from 48k space, then divided by 3
    for the 16k window (maintains temporal alignment).
    """

    def __init__(
        self,
        max_16k: int,
        max_48k: int,
        ssl_encoder: str = "w2v_bert2",
        input_sr: int = 16000,
        processor_tag: str = "facebook/w2v-bert-2.0",
        noise_dir: str = "data/noise_pool",
        rir_dir: str = "data/rir_pool",
        degrade_prob: float = 0.5,
        online_degradation: bool = True,
    ):
        self.max_16k            = max_16k
        self.max_48k            = max_48k
        self.ssl_encoder        = ssl_encoder
        self.input_sr           = input_sr
        self.degrade_prob       = degrade_prob
        self.online_degradation = online_degradation
        self._base = CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

        if ssl_encoder in _SEAMLESS_ENCODERS:
            from transformers import SeamlessM4TFeatureExtractor
            self.processor = SeamlessM4TFeatureExtractor.from_pretrained(processor_tag)
        else:
            self.processor = None

        self.noise_files = _load_file_list(noise_dir)
        self.rir_files   = _load_file_list(rir_dir)

    def __call__(self, data):
        processed = []
        for key, d in data:
            new_d = dict(d)

            ref48 = d.get("speech_ref1")
            if ref48 is None:
                processed.append((key, new_d))
                continue

            if isinstance(ref48, np.ndarray):
                ref48_t = torch.from_numpy(ref48.copy()).float()
            else:
                ref48_t = ref48.float()

            # ── Crop FIRST (on the full-length clean 48k ref) ───────────
            T48 = ref48_t.shape[0]
            if T48 > self.max_48k:
                s48 = random.randint(0, T48 - self.max_48k)
                ref48_crop = ref48_t[s48:s48 + self.max_48k]
            else:
                ref48_crop = ref48_t

            # Resample the CROPPED window (not the full utterance) to 16k
            ref16_crop = AF_.resample(
                ref48_crop.unsqueeze(0), 48000, self.input_sr
            ).squeeze(0)
            if ref16_crop.shape[0] > self.max_16k:
                ref16_crop = ref16_crop[:self.max_16k]

            # ── Degrade SECOND, on the already-cropped short window ─────
            if self.online_degradation:
                try:
                    noisy16_crop = degrade_waveform(
                        ref16_crop, self.input_sr,
                        self.noise_files, self.rir_files,
                        self.degrade_prob,
                    )
                except Exception:
                    noisy16_crop = ref16_crop.clone()
            else:
                noisy16_crop = ref16_crop.clone()

            new_d["speech_ref1"]  = ref48_crop.numpy()
            new_d["noisy_speech"] = noisy16_crop.numpy()

            processed.append((key, new_d))

        keys, batch = self._base(processed)

        if self.ssl_encoder in _SEAMLESS_ENCODERS:
            noisy_np = self._to_np(batch["noisy_speech"],
                                   batch["noisy_speech_lengths"], pad40=True)
            clean_np = self._ref48_to_16k(batch["speech_ref1"],
                                           batch["speech_ref1_lengths"])
            batch["noisy_speech_ssl"] = dict(self.processor(
                noisy_np, sampling_rate=self.input_sr,
                return_tensors="pt", padding=True))
            batch["speech_ref1_ssl"] = dict(self.processor(
                clean_np, sampling_rate=self.input_sr,
                return_tensors="pt", padding=True))
        else:
            ref_16k      = AF_.resample(batch["speech_ref1"], 48000, self.input_sr)
            ref_lens_16k = (batch["speech_ref1_lengths"].float() / 3.0).long()
            batch["noisy_speech_ssl"] = {
                "waveform": batch["noisy_speech"],
                "ilens":    batch["noisy_speech_lengths"],
            }
            batch["speech_ref1_ssl"] = {
                "waveform": ref_16k,
                "ilens":    ref_lens_16k,
            }
        return keys, batch

    def _to_np(self, wav_tensor, lengths, pad40=False):
        result = []
        for i in range(wav_tensor.shape[0]):
            w = wav_tensor[i, :lengths[i]].float().numpy()
            if pad40:
                w = np.pad(w, (40, 40), mode="constant")
            result.append(w)
        return result

    def _ref48_to_16k(self, wav_tensor, lengths):
        result = []
        for i in range(wav_tensor.shape[0]):
            w = wav_tensor[i, :lengths[i]].float()
            w = AF_.resample(w.unsqueeze(0), 48000, self.input_sr).squeeze(0)
            w_np = np.pad(w.numpy(), (40, 40), mode="constant")
            result.append(w_np)
        return result


# =============================================================================
# Stage 1 Task — Feature Predictor
# =============================================================================

class SpeechCleanerFPTask(AbsTask):
    num_optimizers: int = 1

    @classmethod
    def plot_attention(cls, *args, **kwargs): pass

    @classmethod
    def add_task_arguments(cls, parser):
        _add_ssl_args(parser)

    @classmethod
    def build_collate_fn(cls, args, train):
        return _FPCollateFn(
            max_samples=int(getattr(args, "max_duration", 20.0) * 16000),
            ssl_encoder=getattr(args, "ssl_encoder", "w2v_bert2"),
            input_sr=getattr(args, "input_sr", 16000),
            processor_tag="facebook/w2v-bert-2.0",
            noise_dir=getattr(args, "noise_dir", "data/noise_pool"),
            rir_dir=getattr(args, "rir_dir", "data/rir_pool"),
            degrade_prob=getattr(args, "degrade_prob", 0.5),
            online_degradation=getattr(args, "online_degradation", True),
        )

    @classmethod
    def build_preprocess_fn(cls, args, train): return None

    @classmethod
    def required_data_names(cls, train=True, inference=False):
        if inference:
            return ("noisy_speech",)
        # online_degradation=True: only clean (speech_ref1) needed
        # online_degradation=False: both needed
        # We declare speech_ref1 as required; noisy_speech optional
        return ("speech_ref1",)

    @classmethod
    def optional_data_names(cls, train=True, inference=False):
        return ("noisy_speech",)

    @classmethod
    def build_model(cls, args):
        return SpeechCleanerFPModel(
            ssl_encoder=_build_ssl_encoder(args),
            use_multilayer_loss=getattr(args, "use_multilayer_loss", False),
        )

    @classmethod
    def build_optimizers(cls, args, model):
        conf  = dict(getattr(args, "optim_conf", None) or {})
        betas = tuple(conf.get("betas", [0.8, 0.98]))
        return [torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=conf.get("lr", 1e-4), betas=betas,
            weight_decay=conf.get("weight_decay", 0.01),
        )]

    @classmethod
    def build_scheduler(cls, args, optimizers):
        warmup = getattr(args, "warmup_steps", 2000)
        try:
            from transformers import get_constant_schedule_with_warmup
            return [get_constant_schedule_with_warmup(
                optimizers[0], num_warmup_steps=warmup)]
        except ImportError:
            return [torch.optim.lr_scheduler.LambdaLR(
                optimizers[0], lambda s: min(1.0, s / max(warmup, 1)))]

    @classmethod
    def get_trainer(cls): return Trainer


# =============================================================================
# Stage 2/3 Task — Vocoder GAN
# =============================================================================

class SpeechCleanerGANTask(AbsTask):
    num_optimizers: int = 2
    trainer = GANTrainer

    @classmethod
    def plot_attention(cls, *args, **kwargs): pass

    @classmethod
    def add_task_arguments(cls, parser):
        _add_ssl_args(parser)
        g = parser.add_argument_group("Speech Cleaner — Vocoder GAN")
        g.add_argument("--ssl_dim",            type=int,      default=1024)
        g.add_argument("--use_predicted_feat", type=str2bool, default=False)
        g.add_argument("--use_multilayer_feat", type=str2bool, default=False,
                       help="Vocoder uses softmax-weighted sum of SSL layers.")
        g.add_argument("--fp_model_path",      type=str_or_none, default=None)
        g.add_argument("--mel_loss_weight",    type=float,    default=15.0)
        g.add_argument("--adv_loss_weight",    type=float,    default=2.0)
        g.add_argument("--fm_loss_weight",     type=float,    default=1.0)
        g.add_argument("--vocoder_type",       type=str,      default="hifigan",
                       choices=["hifigan", "dac"])

    @classmethod
    def build_collate_fn(cls, args, train):
        max_dur = getattr(args, "max_duration", 20.0)
        return _GANCollateFn(
            max_16k=int(max_dur * 16000),
            max_48k=int(max_dur * 48000),
            ssl_encoder=getattr(args, "ssl_encoder", "w2v_bert2"),
            input_sr=getattr(args, "input_sr", 16000),
            processor_tag="facebook/w2v-bert-2.0",
            noise_dir=getattr(args, "noise_dir", "data/noise_pool"),
            rir_dir=getattr(args, "rir_dir", "data/rir_pool"),
            degrade_prob=getattr(args, "degrade_prob", 0.5),
            online_degradation=getattr(args, "online_degradation", True),
        )

    @classmethod
    def build_preprocess_fn(cls, args, train): return None

    @classmethod
    def required_data_names(cls, train=True, inference=False):
        return ("noisy_speech",) if inference else ("speech_ref1",)

    @classmethod
    def optional_data_names(cls, train=True, inference=False):
        return ("noisy_speech",)

    @classmethod
    def build_model(cls, args):
        ssl_encoder         = _build_ssl_encoder(args)
        fp_path             = getattr(args, "fp_model_path", None)
        use_multilayer_feat = getattr(args, "use_multilayer_feat", False)

        if fp_path is not None:
            state  = torch.load(fp_path, map_location="cpu", weights_only=False)
            sd     = state.get("model", state)
            enc_sd = {k.replace("ssl_encoder.", "", 1): v
                      for k, v in sd.items() if k.startswith("ssl_encoder.")}
            missing, _ = ssl_encoder.load_state_dict(enc_sd, strict=False)
            if missing:
                logger.warning("ssl_encoder missing keys: %s", missing[:5])
            for p in ssl_encoder.parameters():
                p.requires_grad = False
            logger.info("Loaded ssl_encoder from FP checkpoint and froze.")

        ssl_dim = args.ssl_dim if args.ssl_dim > 0 else ssl_encoder.ssl_dim
        vocoder = (DACVocoder(input_dim=ssl_dim)
                   if getattr(args, "vocoder_type", "hifigan") == "dac"
                   else SpeechCleanerVocoder(input_dim=ssl_dim))

        return SpeechCleanerVocoderModel(
            ssl_encoder=ssl_encoder,
            vocoder=vocoder,
            use_predicted_feat=args.use_predicted_feat,
            use_multilayer_feat=use_multilayer_feat,
            layer_weighting=getattr(args, "layer_weighting", None),   
            mel_loss_weight=args.mel_loss_weight,
            adv_loss_weight=args.adv_loss_weight,
            fm_loss_weight=args.fm_loss_weight,
        )

    @classmethod
    def build_optimizers(cls, args, model):
        conf = dict(getattr(args, "optim_conf", None) or {})
        lr   = conf.get("lr", 1e-4)
        lr_d = conf.get("lr_d", lr)   # optional: separate discriminator LR
        wd   = conf.get("weight_decay", 0.01)
        b    = tuple(conf.get("betas", [0.8, 0.98]))
 
        # Generator: vocoder + layer_weights (if multilayer)
        gen_params = list(model.vocoder.parameters())
        if getattr(model, "layer_weights", None) is not None:
            gen_params.append(model.layer_weights)
        if getattr(model, "layer_router", None) is not None:
            gen_params += list(model.layer_router.parameters())   
 
        # Discriminator: DAC Discriminator (replaces MPD + MSD)
        disc_params = list(model.discriminator.parameters())
 
        return [
            torch.optim.AdamW(gen_params,  lr=lr,   betas=b, weight_decay=wd),
            torch.optim.AdamW(disc_params, lr=lr_d,  betas=b, weight_decay=wd),
        ]
 

    @classmethod
    def build_scheduler(cls, args, optimizers):
        warmup     = getattr(args, "warmup_steps", 2000)
        sched_type = getattr(args, "scheduler", "exponentiallr")
        gamma      = (dict(getattr(args, "scheduler_conf", None) or {})).get("gamma", 0.999996)
        schedulers = []
        for opt in optimizers:
            if sched_type == "exponentiallr":
                schedulers.append(torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma))
            else:
                from transformers import get_constant_schedule_with_warmup
                schedulers.append(get_constant_schedule_with_warmup(opt, num_warmup_steps=warmup))
        return schedulers

    @classmethod
    def get_trainer(cls): return GANTrainer