#!/usr/bin/env python3
"""Inference with Sidon original published checkpoints (FP + vocoder).

Sidon-faithful preprocessing (verified from Sidon/infer.py):
  1. Resample to 16 kHz
  2. Peak normalize: 0.9 * wav / max_val
  3. 160-sample padding: F.pad(wav, (160, 160))
  4. extract_seamless_m4t_features (kaldi fbank, stride-2)
  5. FP: input_features tensor only (not **ssl_inputs)

Uses:
  feature_extractor_cuda.pt + decoder_cuda.pt
  from sarulab-speech/sidon-v0.1

Usage:
    python3 local/infer_sidon_original.py \
        --sidon_dir /home/samsung/.cache/huggingface/hub/models--sarulab-speech--sidon-v0.1/snapshots/b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52 \
        --datasets  all \
        --device    cuda

    python3 local/infer_sidon_original.py \
        --sidon_dir /home/samsung/.cache/huggingface/hub/models--sarulab-speech--sidon-v0.1/snapshots/b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52 \
        --wav_scp   data/libritts_test-clean/wav.scp \
        --out_dir   exp/restored_sidon_orig_libritts_test-clean \
        --device    cuda
"""

#!/usr/bin/env python3
"""Inference with Sidon original published checkpoints (FP + vocoder).

Supports:
  - File paths and sox pipe commands in wav.scp
  - Kaldi segments file for segment-level inference (ami_sdm_test)
  - Chunk-based inference for long-form audio with overlap-add (ami_sdm_longform)

Usage:
    # Segment-level (ami_sdm_test, with segments file)
    python3 local/infer_sidon_original.py \
        --sidon_dir  /path/to/sidon-v0.1 \
        --wav_scp    data/ami_sdm_test/wav.scp \
        --segments   data/ami_sdm_test/segments \
        --out_dir    exp/restored_sidon_orig_ami_sdm_test \
        --device     cuda

    # Long-form (ami_sdm_longform, no segments file)
CUDA_VISIBLE_DEVICES=2    python3 local/infer_sidon_original.py \
        --sidon_dir /home/samsung/.cache/huggingface/hub/models--sarulab-speech--sidon-v0.1/snapshots/b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52 \
        --wav_scp    data/ami_sdm_longform/wav.scp \
        --out_dir    exp/restored_sidon_orig_ami_sdm_longform \
        --chunk_sec  20.0 \
        --device     cuda

    # LibriTTS (file paths, no segments)
    python3 local/infer_sidon_original.py \
        --sidon_dir  /path/to/sidon-v0.1 \
        --wav_scp    data/libritts_test-clean/wav.scp \
        --out_dir    exp/restored_sidon_orig_libritts_test-clean \
        --device     cuda
"""

import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATASETS = {
    "libritts_test-clean":        "data/libritts_test-clean/wav.scp",
    "libritts_test-other":        "data/libritts_test-other/wav.scp",
    "libritts_test-clean-degrad": "data/libritts_test-clean-degrad/noisy/wav.scp",
    "libritts_test-other-degrad": "data/libritts_test-other-degrad/noisy/wav.scp",
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sidon_dir",   required=True)
    p.add_argument("--wav_scp",     default=None)
    p.add_argument("--segments",    default=None,
                   help="Kaldi segments file. If provided, each segment is "
                        "extracted from the recording and inferred separately.")
    p.add_argument("--out_dir",     default=None)
    p.add_argument("--datasets",    default=None, choices=["all"])
    p.add_argument("--out_base",    default="exp")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--input_sr",    type=int, default=16000)
    p.add_argument("--chunk_sec",   type=float, default=20.0,
                   help="Chunk duration in seconds for long-form inference. "
                        "Only used when --segments is NOT provided.")
    p.add_argument("--overlap_sec", type=float, default=0.5,
                   help="Overlap between chunks in seconds for overlap-add.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def read_wav_scp(path: str) -> Dict[str, str]:
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def read_segments(path: str) -> List[Tuple[str, str, float, float]]:
    """Read Kaldi segments file.
    Returns list of (uttid, recording_id, start_sec, end_sec).
    """
    segs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                uttid   = parts[0]
                recid   = parts[1]
                start   = float(parts[2])
                end     = float(parts[3])
                segs.append((uttid, recid, start, end))
    return segs


def load_wav_from_pipe_or_file(wav_cmd: str, target_sr: int = 16000) -> np.ndarray:
    """Load full recording from sox pipe or file path."""
    wav_cmd = wav_cmd.strip()
    if wav_cmd.endswith("|"):
        cmd  = wav_cmd[:-1].strip()
        proc = subprocess.run(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"sox pipe failed (return {proc.returncode}): "
                f"{proc.stderr.decode()}\ncmd: {cmd}"
            )
        wav = (np.frombuffer(proc.stdout, dtype=np.int16)
               .astype(np.float32) / 32768.0)
    else:
        import librosa
        wav, sr = sf.read(wav_cmd, always_2d=True)
        wav = wav.mean(axis=1).astype(np.float32)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


def load_wav_segment(
    wav_cmd: str,
    start_sec: float,
    end_sec: float,
    target_sr: int = 16000,
) -> np.ndarray:
    """Load a segment from a recording via sox trim.

    Works whether wav_cmd is a file path or a sox pipe (without trim).
    Injects 'trim start_sec =end_sec' into the sox command.
    """
    wav_cmd = wav_cmd.strip()

    if wav_cmd.endswith("|"):
        # Already a sox pipe: inject trim before the trailing '|'
        # e.g. "sox file.wav -r 16000 -c 1 -t wavpcm - |"
        # → "sox file.wav -r 16000 -c 1 -t wavpcm - trim 1.0 =3.5 |"
        base = wav_cmd[:-1].strip()
        cmd  = f"{base} trim {start_sec:.3f} ={end_sec:.3f} |"
    else:
        # Plain file path: build sox command
        cmd = (f"sox {wav_cmd} -r {target_sr} -c 1 -t wavpcm - "
               f"trim {start_sec:.3f} ={end_sec:.3f} |")

    return load_wav_from_pipe_or_file(cmd, target_sr)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(sidon_dir, device):
    suffix = "cuda" if device.startswith("cuda") else "cpu"
    alt    = "cpu" if suffix == "cuda" else "cuda"

    def _load(name):
        path = os.path.join(sidon_dir, f"{name}_{suffix}.pt")
        if not os.path.exists(path):
            path = os.path.join(sidon_dir, f"{name}_{alt}.pt")
            logger.warning("Primary not found, using: %s", path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} checkpoint not found in {sidon_dir}.\n"
                "Download: huggingface-cli download sarulab-speech/sidon-v0.1"
            )
        logger.info("Loading %s", path)
        return torch.jit.load(path, map_location=device).eval().to(device)

    return _load("feature_extractor"), _load("decoder")


# ---------------------------------------------------------------------------
# Preprocessing: Sidon-faithful
# ---------------------------------------------------------------------------

def _extract_fbank_features(wav_t: torch.Tensor, device: str) -> dict:
    import torchaudio
    feat = torchaudio.compliance.kaldi.fbank(
        wav_t.unsqueeze(0),
        sample_frequency=16000,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        preemphasis_coefficient=0.97,
        remove_dc_offset=True,
        window_type="povey",
        use_energy=False,
        energy_floor=1.192092955078125e-07,
    )
    mean = feat.mean(0, keepdim=True)
    var  = feat.var(0, keepdim=True)
    feat = (feat - mean) / torch.sqrt(var + 1e-5)
    T    = (feat.shape[0] // 2) * 2
    feat = feat[:T].reshape(T // 2, 160) 
    return {
        "input_features": feat.unsqueeze(0).to(device),
        "attention_mask": torch.ones(1, T // 2, dtype=torch.int64, device=device),
    }


def preprocess(wav_np: np.ndarray, device: str) -> dict:
    wav_t   = torch.from_numpy(wav_np).float()
    max_val = wav_t.abs().max().clamp_min(1e-6)
    wav_t   = 0.9 * wav_t / max_val
    wav_t   = F.pad(wav_t, (160, 160) 
    return _extract_fbank_features(wav_t, device)


# ---------------------------------------------------------------------------
# Core inference (single chunk)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _infer_chunk(wav_np: np.ndarray, fp, voc, device: str) -> np.ndarray:
    """Inference on a single numpy chunk → numpy output at 48 kHz."""
    ssl_inputs = preprocess(wav_np, device)
    fp_out     = fp(ssl_inputs["input_features"])

    if isinstance(fp_out, dict):
        enhanced = list(fp_out.values())[0]
    elif isinstance(fp_out, (tuple, list)):
        enhanced = fp_out[0]
    else:
        enhanced = fp_out

    restored = voc(enhanced.transpose(1, 2))
    if restored.dim() == 3:
        restored = restored.squeeze(1)
    return restored.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Short utterance inference (≤ max_sec)
# ---------------------------------------------------------------------------

def restore_short(wav_np: np.ndarray, fp, voc, device: str) -> np.ndarray:
    return _infer_chunk(wav_np, fp, voc, device)


# ---------------------------------------------------------------------------
# Long-form inference with overlap-add chunking
# ---------------------------------------------------------------------------

def restore_longform(
    wav_np: np.ndarray,
    fp,
    voc,
    device: str,
    input_sr: int   = 16000,
    output_sr: int  = 48000,
    chunk_sec: float = 20.0,
    overlap_sec: float = 0.5,
) -> np.ndarray:
    """Chunk-based inference with linear overlap-add crossfade.

    Processing:
      1. Split wav into overlapping chunks of chunk_sec with overlap_sec overlap
      2. Infer each chunk independently
      3. Crossfade at boundaries using linear ramp (overlap-add)

    Args:
        wav_np:      input waveform at input_sr
        chunk_sec:   chunk duration in seconds
        overlap_sec: overlap duration in seconds (crossfade region)
    """
    chunk_samples   = int(chunk_sec   * input_sr)
    overlap_samples = int(overlap_sec * input_sr)
    hop_samples     = chunk_samples - overlap_samples
    total_samples   = len(wav_np)

    # Upsample ratio for output alignment
    up_ratio = output_sr / input_sr

    # Split into chunks
    chunks = []
    pos = 0
    while pos < total_samples:
        end   = min(pos + chunk_samples, total_samples)
        chunk = wav_np[pos:end]
        # Skip chunks that are too short to process (< 0.1s)
        if len(chunk) >= int(0.1 * input_sr):
            chunks.append((pos, chunk))
        pos += hop_samples

    logger.debug("  %d chunks (chunk=%.1fs, overlap=%.1fs)",
                 len(chunks), chunk_sec, overlap_sec)

    if len(chunks) == 0:
        return np.array([], dtype=np.float32)

    if len(chunks) == 1:
        return _infer_chunk(chunks[0][1], fp, voc, device)

    # Infer all chunks
    restored_chunks = []
    for i, (pos, chunk) in enumerate(chunks):
        try:
            out = _infer_chunk(chunk, fp, voc, device)
            restored_chunks.append((pos, out))
        except Exception as e:
            logger.warning("  chunk %d failed: %s", i, e)
            # Fill with zeros on failure
            out_len = int(len(chunk) * up_ratio)
            restored_chunks.append((pos, np.zeros(out_len, dtype=np.float32)))

    # Overlap-add with linear crossfade
    output_len = int(total_samples * up_ratio) + int(overlap_sec * output_sr)
    output     = np.zeros(output_len, dtype=np.float32)
    weight     = np.zeros(output_len, dtype=np.float32)

    overlap_out = int(overlap_sec * output_sr)

    for i, (pos, out) in enumerate(restored_chunks):
        out_start = int(pos * up_ratio)
        out_end   = out_start + len(out)

        if out_end > output_len:
            out    = out[:output_len - out_start]
            out_end = output_len

        # Build envelope: fade-in at start, fade-out at end
        env = np.ones(len(out), dtype=np.float32)

        # Fade-in: first chunk has no fade-in; others fade in over overlap region
        if i > 0 and overlap_out > 0:
            fade_len = min(overlap_out, len(out))
            env[:fade_len] = np.linspace(0.0, 1.0, fade_len)

        # Fade-out: last chunk has no fade-out; others fade out over overlap region
        if i < len(restored_chunks) - 1 and overlap_out > 0:
            fade_len = min(overlap_out, len(out))
            env[-fade_len:] = np.minimum(
                env[-fade_len:],
                np.linspace(1.0, 0.0, fade_len),
            )

        output[out_start:out_end] += out * env
        weight[out_start:out_end] += env

    # Normalize by weight to avoid amplitude artifacts in overlap regions
    valid = weight > 1e-6
    output[valid] /= weight[valid]

    # Trim to expected length
    expected_len = int(total_samples * up_ratio)
    return output[:expected_len].astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level restore dispatcher
# ---------------------------------------------------------------------------

def restore_one(
    wav_np: np.ndarray,
    fp,
    voc,
    device: str,
    input_sr: int    = 16000,
    output_sr: int   = 48000,
    chunk_sec: float = 20.0,
    overlap_sec: float = 0.5,
) -> np.ndarray:
    """Restore one waveform. Uses chunking if longer than chunk_sec."""
    dur = len(wav_np) / input_sr
    if dur <= chunk_sec:
        return restore_short(wav_np, fp, voc, device)
    else:
        logger.debug("  long-form (%.1fs): chunked inference", dur)
        return restore_longform(
            wav_np, fp, voc, device,
            input_sr=input_sr, output_sr=output_sr,
            chunk_sec=chunk_sec, overlap_sec=overlap_sec,
        )


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_dataset(
    wav_scp: str,
    out_dir: str,
    fp,
    voc,
    device: str,
    input_sr: int,
    max_samples: int,
    chunk_sec: float,
    overlap_sec: float,
    segments_path: Optional[str] = None,
):
    out_dir = Path(out_dir)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    wav_map = read_wav_scp(wav_scp)

    if segments_path and os.path.exists(segments_path):
        # Segment-level inference: each segment extracted from recording
        logger.info("Segment mode: reading %s", segments_path)
        segs = read_segments(segments_path)
        if max_samples > 0:
            segs = segs[:max_samples]
        entries = segs  # (uttid, recid, start, end)
        mode = "segment"
    else:
        # Recording-level inference (with optional chunking)
        entries = [(recid, recid, None, None) for recid in sorted(wav_map)]
        if max_samples > 0:
            entries = entries[:max_samples]
        mode = "recording"

    logger.info("Mode: %s | %d entries → %s", mode, len(entries), out_dir)

    scp_lines = []
    for i, (uttid, recid, start, end) in enumerate(entries):
        try:
            wav_cmd = wav_map.get(recid)
            if wav_cmd is None:
                logger.warning("Recording %s not in wav.scp, skipping", recid)
                continue

            if mode == "segment":
                wav_np = load_wav_segment(wav_cmd, start, end, input_sr)
            else:
                wav_np = load_wav_from_pipe_or_file(wav_cmd, input_sr)

            restored = restore_one(
                wav_np, fp, voc, device,
                input_sr=input_sr, output_sr=48000,
                chunk_sec=chunk_sec, overlap_sec=overlap_sec,
            )

            out_path = wav_dir / f"{uttid}.wav"
            sf.write(str(out_path), restored, 48000)
            scp_lines.append(f"{uttid} {out_path}")

            if (i + 1) % 200 == 0 or i == 0:
                logger.info("  [%d/%d] %s (%.1fs → %.1fs)",
                            i+1, len(entries), uttid,
                            len(wav_np) / input_sr,
                            len(restored) / 48000)
        except Exception as e:
            logger.error("FAILED %s: %s", uttid, e)

    scp_path = out_dir / "wav.scp"
    with open(scp_path, "w") as f:
        f.write("\n".join(scp_lines) + "\n")
    logger.info("Done: %d/%d written → %s", len(scp_lines), len(entries), scp_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = get_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    fp, voc = load_models(args.sidon_dir, device)

    if args.datasets == "all":
        for name, scp in DEFAULT_DATASETS.items():
            if not os.path.exists(scp):
                logger.warning("Skipping %s (not found: %s)", name, scp)
                continue
            out_dir = os.path.join(args.out_base, f"restored_sidon_orig_{name}")
            logger.info("=" * 60)
            logger.info("Dataset: %s", name)
            process_dataset(
                scp, out_dir, fp, voc, device,
                args.input_sr, args.max_samples,
                args.chunk_sec, args.overlap_sec,
            )
    elif args.wav_scp:
        if not args.out_dir:
            raise ValueError("--out_dir is required when --wav_scp is used")
        process_dataset(
            args.wav_scp, args.out_dir, fp, voc, device,
            args.input_sr, args.max_samples,
            args.chunk_sec, args.overlap_sec,
            segments_path=args.segments,
        )
    else:
        raise ValueError("Provide --wav_scp + --out_dir, or --datasets all")


if __name__ == "__main__":
    main()