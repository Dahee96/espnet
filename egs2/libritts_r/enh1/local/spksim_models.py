#!/usr/bin/env python3
"""Speaker similarity backends: WavLM (existing), ECAPA-TDNN, RawNet3.

Replicates Samuele's ACTUAL windowed embedding-averaging protocol, as
confirmed from his reference implementation (not the earlier verbal
"0.5s stride" description, which turned out to not match the code):

  - window = 3.0s, hop = 1.5s (NOT 0.5s)
  - segments shorter than the window are ZERO-PADDED to exactly one
    window length (not used as-is)
  - the final window is shifted to end exactly at the segment boundary
    ("flush" window), so the tail of the segment is never under-covered
  - per-window embeddings are mean-pooled THEN L2-normalized once
    (normalizing per-window before averaging silently shrinks the
    averaged vector's norm and biases cosine similarity downward —
    this was an earlier bug here, now fixed to normalize only once,
    after pooling)
  - everything is resampled to 16kHz before embedding extraction

This windowing is applied uniformly regardless of utterance length or
dataset (LibriTTS / AMI / Fisher) — for utterances shorter than the window
size, padding makes the same windowing logic correct everywhere without
special-casing by dataset.

Reference comparison target: noisy/original input vs restored (NOT a
separate "clean GT" file) — confirmed this matches current setup; the
input recording IS the restoration target here, so noisy_scp doubles as
the "clean reference" in Samuele's terminology.

Models
------
wavlm   : microsoft/wavlm-base-plus-sv (existing implementation, kept as-is
          for backward comparability with prior numbers)
ecapa   : speechbrain/spkrec-ecapa-voxceleb (ECAPA-TDNN, SpeechBrain)
rawnet3 : espnet/voxcelebs12_rawnet3 (RawNet3, ESPnet — same model VERSA's
          speaker.py uses internally via espnet2.bin.spk_inference)

Install
-------
ecapa:    pip install speechbrain
rawnet3:  pip install espnet espnet_model_zoo
          (already present if espnet2 is installed for this repo)
wavlm:    pip install transformers  (already required elsewhere in score.py)
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Segment loading (matches Samuele's load_segment_16k() ordering: read the
# exact sample range at native SR first, THEN resample to 16kHz)
# =============================================================================

def load_segment_16k(path: str, start: float, end: float) -> np.ndarray:
    """Random-access load of [start, end] seconds, mono, resampled to 16kHz.

    Matches Samuele's load_segment_16k(): reads ONLY the needed sample range
    directly from disk at the file's native sample rate, THEN resamples to
    16kHz — as opposed to resampling the whole recording first and slicing
    afterward. This ordering avoids per-segment resampling artifacts
    compounding differently than a single whole-file resample, and matters
    for matching exact reported numbers.

    For sox-pipe wav.scp entries, trims directly in the pipe at the source
    rate (same "read only the needed range" principle) before the pipe's
    own resampling. Returns None if the segment is empty after clamping to
    the file's actual length.
    """
    import soundfile as sf
    import torchaudio.functional as AF

    path = path.strip()
    if path.endswith("|"):
        base = path[:-1].strip()
        cmd = f"{base} trim {start:.3f} ={end:.3f} |"
        import subprocess
        proc = subprocess.run(
            cmd[:-1].strip(), shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"sox pipe failed: {proc.stderr.decode()}\ncmd: {cmd}")
        wav = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        if len(wav) == 0:
            return None
        return wav.astype(np.float32)

    info = sf.info(path)
    sr = info.samplerate
    s = max(0, int(round(start * sr)))
    e = min(info.frames, int(round(end * sr)))
    if e <= s:
        return None
    wav, _ = sf.read(path, start=s, stop=e, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # mono
    t = torch.from_numpy(wav)
    if sr != 16000:
        t = AF.resample(t, sr, 16000)
    return t.numpy().astype(np.float32)


# =============================================================================
# Windowing helper (shared by all backends) — matches Samuele's windows()
# =============================================================================

def _make_windows(
    wav: np.ndarray,
    sr: int = 16000,
    win_sec: float = 3.0,
    hop_sec: float = 1.5,
) -> List[np.ndarray]:
    """Slice wav into overlapping windows of win_sec with hop_sec stride.

    Matches Samuele's windows() exactly:
      - if shorter than one window, ZERO-PAD to exactly win_sec (not
        returned as-is at its original length)
      - otherwise, slide with hop_sec stride, and if the last regular
        window doesn't end exactly at the segment boundary, add one more
        "flush" window that ends exactly at n (so the tail is never
        under-covered, even though it overlaps the previous window more
        than hop_sec).

    Default hop_sec=1.5 (not 0.5) — this matches Samuele's actual code,
    not his earlier verbal description.
    """
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    n = len(wav)

    if n < win:
        pad = win - n
        return [np.pad(wav, (0, pad), mode="constant")]

    starts = list(range(0, n - win + 1, hop))
    if starts[-1] != n - win:
        starts.append(n - win)  # flush window ending exactly at segment end

    return [wav[s:s + win] for s in starts]


# =============================================================================
# WavLM (existing backend — unchanged behavior, just refactored to share
# the windowing helper for consistency with the other two backends)
# =============================================================================

class WavLMSpkEmbedder:
    """microsoft/wavlm-base-plus-sv, windowed-mean embedding."""

    def __init__(self, device: str):
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel
        model_id = "microsoft/wavlm-base-plus-sv"
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = WavLMModel.from_pretrained(model_id).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def embed(self, wav_np: np.ndarray, win_sec: float = 3.0,
              hop_sec: float = 1.5) -> torch.Tensor:
        windows = _make_windows(wav_np, 16000, win_sec, hop_sec)
        embs = []
        for w in windows:
            inp = self.extractor(w, sampling_rate=16000,
                                 return_tensors="pt", padding=True).to(self.device)
            out = self.model(**inp)
            e = out.last_hidden_state.mean(dim=1)   # [1, D]
            embs.append(e)
        avg = torch.stack(embs, dim=0).mean(dim=0)   # [1, D] — average RAW embeddings
        return F.normalize(avg, dim=-1)              # normalize ONCE, after averaging


# =============================================================================
# ECAPA-TDNN (SpeechBrain)
# =============================================================================

class EcapaSpkEmbedder:
    """speechbrain/spkrec-ecapa-voxceleb, windowed-mean embedding."""

    def __init__(self, device: str):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as e:
            raise ImportError(
                "ECAPA backend requires speechbrain: pip install speechbrain"
            ) from e
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="versa_cache/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        self.device = device

    @torch.inference_mode()
    def embed(self, wav_np: np.ndarray, win_sec: float = 3.0,
              hop_sec: float = 1.5, max_bs: int = 256) -> torch.Tensor:
        windows = _make_windows(wav_np, 16000, win_sec, hop_sec)
        win_tensors = [torch.from_numpy(w).float() for w in windows]

        embs = []
        for i in range(0, len(win_tensors), max_bs):
            batch = torch.stack(win_tensors[i:i + max_bs]).to(self.device)  # [B, T]
            e = self.model.encode_batch(batch)        # [B, 1, D]
            embs.append(e.squeeze(1).float())           # [B, D]
        avg = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)  # [1, D] — average RAW embeddings
        return F.normalize(avg, dim=-1)                          # normalize ONCE, after averaging


# =============================================================================
# RawNet3 (ESPnet, exact VERSA-style protocol — NO windowing)
# =============================================================================

class RawNet3SpkEmbedder:
    """espnet/voxcelebs12_rawnet3, whole-segment embedding (VERSA protocol).

    Unlike WavLM/ECAPA in this file, RawNet3 does NOT use sliding-window
    averaging here — this matches Samuele's actual VERSA-style
    implementation exactly:

        "RawNet3 handles arbitrary length via its own attentive pooling,
         so (unlike the ECAPA scorer) we feed the whole segment rather
         than 3s windows -- this is what VERSA does."

    The only length handling is a 1-second minimum-length zero-pad (RawNet3
    stability floor; the actual segment-selection floor of >=3s is applied
    upstream when building the eligible segment set, same as for ECAPA).

    win_sec / hop_sec arguments are accepted for interface compatibility
    with the other embedders (so callers can pass the same kwargs to any
    backend) but are IGNORED — RawNet3 always embeds the whole input array
    as a single segment, per VERSA's protocol.
    """

    MIN_SAMPLES = 16000  # 1s floor at 16kHz, matches Samuele's RawNet3 scorer

    def __init__(self, device: str):
        try:
            from espnet2.bin.spk_inference import Speech2Embedding
        except ImportError as e:
            raise ImportError(
                "RawNet3 backend requires espnet2: pip install espnet espnet_model_zoo"
            ) from e
        self.model = Speech2Embedding.from_pretrained(
            model_tag="espnet/voxcelebs12_rawnet3", device=device,
        )
        if hasattr(self.model, "spk_model"):
            self.model.spk_model.eval()
        self.device = device

    @torch.inference_mode()
    def embed(self, wav_np: np.ndarray, win_sec: float = 3.0,
              hop_sec: float = 1.5) -> torch.Tensor:
        t = torch.from_numpy(wav_np).float()
        if t.shape[0] < self.MIN_SAMPLES:
            t = torch.nn.functional.pad(t, (0, self.MIN_SAMPLES - t.shape[0]))

        emb = self.model(t.to(self.device))   # Speech2Embedding __call__
        if isinstance(emb, (list, tuple)):
            emb = emb[0]
        emb = torch.as_tensor(emb).float().reshape(1, -1)  # [1, D]
        return F.normalize(emb, dim=-1)


# =============================================================================
# Unified factory
# =============================================================================

_BACKENDS = {
    "wavlm":   WavLMSpkEmbedder,
    "ecapa":   EcapaSpkEmbedder,
    "rawnet3": RawNet3SpkEmbedder,
}


def get_spk_embedder(backend: str, device: str):
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown spksim backend: {backend!r}. "
                         f"Choose from: {list(_BACKENDS)}")
    return _BACKENDS[backend](device)