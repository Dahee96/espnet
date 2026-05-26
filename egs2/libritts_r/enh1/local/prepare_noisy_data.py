#!/usr/bin/env python3
"""Sidon degradation simulation pipeline.

Generates (noisy, clean) paired data for training.

Degradation order (following Sidon paper Section 3.4):
  1. Reverberation       (pyroomacoustics)
     - RT60: U(0.1, 2.0) seconds
     - Room dimensions: U(2, 20) m (rectangular cuboid)
  2. Background noise    (AudioSet / WHAM! / FSD50K / SC-Wind)
     - SNR: U(-5, 20) dB
  3. Band limitation
     - Target SR: {8, 16, 22.05, 24, 44.1, 48} kHz
  4. Clipping
     - min quantile: U(0, 10th percentile)
     - max quantile: U(90th, 100th percentile)
  5. Codec               (MP3 at random bitrate 65–245 kbps)
  6. Packet loss
     - Random 9% segments replaced with zeros
     - Each segment duration: U(20, 200) milliseconds  [Sidon paper]

Each degradation applied with probability 0.5 (independent).
The full pipeline is applied N times per utterance (default 4).

Output directory structure
---------------------------
out_dir/
  clean/wav.scp       — paths to original clean WAVs (resampled)
  noisy/wav.scp       — paths to generated noisy WAVs
  noisy/wavs/         — actual noisy WAV files
"""

import argparse
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from typing import List, Tuple

import numpy as np
import soundfile as sf

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Candidate sample rates for band-limitation augmentation (Sidon paper)
_SR_CANDIDATES = [8000, 16000, 22050, 24000, 44100, 48000]


# ---------------------------------------------------------------------------
# Individual degradation functions
# ---------------------------------------------------------------------------

def _apply_reverberation(audio: np.ndarray, sr: int) -> np.ndarray:
    """Simulate room reverberation using pyroomacoustics.

    Sidon paper: RT60 ~ U(0.1, 2.0)s, room dims ~ U(2, 20)m
    """
    try:
        import pyroomacoustics as pra
    except ImportError:
        logger.warning("pyroomacoustics not installed; skipping reverberation")
        return audio

    rt60 = random.uniform(0.1, 2.0)
    room_dim = [random.uniform(2.0, 20.0) for _ in range(3)]
    try:
        e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
        e_abs = float(np.clip(e_abs, 1e-4, 0.9999))
        room = pra.ShoeBox(
            room_dim, fs=sr,
            materials=pra.Material(e_abs),
            max_order=max_order,
        )
        src_pos = [d * random.uniform(0.1, 0.9) for d in room_dim]
        mic_pos = [d * random.uniform(0.1, 0.9) for d in room_dim]
        room.add_source(src_pos, signal=audio)
        room.add_microphone(mic_pos)
        room.simulate()
        rev = room.mic_array.signals[0][: len(audio)]
        if np.abs(rev).max() > 1e-8:
            rev = rev / np.abs(rev).max() * np.abs(audio).max()
        return rev.astype(np.float32)
    except Exception as exc:
        logger.debug("Reverberation failed (%s); returning original", exc)
        return audio


def _apply_noise(audio: np.ndarray, sr: int, noise_dir: str) -> np.ndarray:
    """Add background noise at SNR ~ U(-5, 20) dB.

    Sidon paper: noise looped to match utterance duration, SNR ~ U(-5, 20) dB
    """
    noise_files = []
    for root, _, files in os.walk(noise_dir):
        for f in files:
            if f.endswith(".wav") or f.endswith(".flac"):
                noise_files.append(os.path.join(root, f))
    if not noise_files:
        return audio

    noise_path = random.choice(noise_files)
    try:
        noise, noise_sr = sf.read(noise_path, dtype="float32")
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
        if noise_sr != sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, noise_sr)
            noise = resample_poly(noise, sr // g, noise_sr // g)
        # Loop noise to match utterance length exactly
        reps = (len(audio) // max(len(noise), 1)) + 2
        noise = np.tile(noise, reps)[: len(audio)]
        snr_db = random.uniform(-5.0, 20.0)
        signal_power = np.mean(audio ** 2) + 1e-12
        noise_power  = np.mean(noise ** 2) + 1e-12
        scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
        return (audio + scale * noise).astype(np.float32)
    except Exception as exc:
        logger.debug("Noise addition failed (%s)", exc)
        return audio


def _apply_band_limitation(audio: np.ndarray, sr: int) -> np.ndarray:
    """Randomly resample to a lower SR and back.

    Sidon paper: target SR in {8, 16, 22.05, 24, 44.1, 48} kHz
    """
    from math import gcd
    from scipy.signal import resample_poly
    target_sr = random.choice(_SR_CANDIDATES)
    if target_sr == sr:
        return audio
    g1   = gcd(sr, target_sr)
    down = resample_poly(audio, target_sr // g1, sr // g1)
    g2   = gcd(target_sr, sr)
    up   = resample_poly(down, sr // g2, target_sr // g2)
    return up[: len(audio)].astype(np.float32)


def _apply_clipping(audio: np.ndarray) -> np.ndarray:
    """Clip audio at random quantile thresholds.

    Sidon paper:
        min = quantile(U(0th, 10th percentile))
        max = quantile(U(90th, 100th percentile))
    """
    low_q  = random.uniform(0.00, 0.10)
    high_q = random.uniform(0.90, 1.00)
    low_v  = float(np.quantile(audio, low_q))
    high_v = float(np.quantile(audio, high_q))
    return np.clip(audio, low_v, high_v).astype(np.float32)


def _apply_codec(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply MP3 codec compression.

    Sidon paper: random average bitrate in [65, 245] kbps
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        logger.debug("pydub not installed; skipping codec augmentation")
        return audio

    bitrate = random.randint(65, 245)
    try:
        pcm = (audio * 32767).astype(np.int16)
        seg = AudioSegment(pcm.tobytes(), frame_rate=sr, sample_width=2, channels=1)
        buf = BytesIO()
        seg.export(buf, format="mp3", bitrate=f"{bitrate}k")
        buf.seek(0)
        decoded = AudioSegment.from_mp3(buf)
        samples = np.frombuffer(decoded.raw_data, dtype=np.int16).astype(np.float32)
        samples /= 32767.0
        if len(samples) >= len(audio):
            return samples[: len(audio)]
        return np.pad(samples, (0, len(audio) - len(samples)))
    except Exception as exc:
        logger.debug("Codec augmentation failed (%s)", exc)
        return audio


def _apply_packet_loss(
    audio: np.ndarray,
    sr: int,
    loss_rate: float = 0.09,
    min_duration_ms: float = 20.0,
    max_duration_ms: float = 200.0,
) -> np.ndarray:
    """Simulate packet loss by zeroing random segments.

    Sidon paper Section 3.4:
        "Random 9% segments of speech were selected for packet loss.
         For each segment duration sampled from U(20, 200) milliseconds
         were selected to be replaced with zeros to simulate packet loss."

    Implementation:
        total_duration * loss_rate / mean_chunk_duration = expected number of chunks
        Each chunk: duration ~ U(min_duration_ms, max_duration_ms) ms
        Start time: uniformly random within valid range

    Args:
        audio            : input waveform
        sr               : sample rate
        loss_rate        : fraction of total duration to zero out (default 0.09)
        min_duration_ms  : minimum chunk duration in ms (default 20, Sidon paper)
        max_duration_ms  : maximum chunk duration in ms (default 200, Sidon paper)
    """
    result = audio.copy()
    total_duration_s = len(audio) / sr

    # Expected total zeroed duration
    target_zeroed_s = total_duration_s * loss_rate
    # Mean chunk duration
    mean_chunk_s = (min_duration_ms + max_duration_ms) / 2 / 1000.0
    # Expected number of chunks
    num_chunks = max(1, int(round(target_zeroed_s / mean_chunk_s)))

    for _ in range(num_chunks):
        chunk_s = random.uniform(min_duration_ms / 1000.0, max_duration_ms / 1000.0)
        max_start = max(0.0, total_duration_s - chunk_s)
        start_s = random.uniform(0.0, max_start)
        start_sample = int(start_s * sr)
        end_sample   = min(len(audio), int((start_s + chunk_s) * sr))
        result[start_sample:end_sample] = 0.0

    return result.astype(np.float32)


def degrade_once(
    audio: np.ndarray,
    sr: int,
    noise_dir: str,
    apply_prob: float = 0.5,
) -> np.ndarray:
    """Apply full degradation pipeline once.

    Each of the 6 degradations is applied independently with probability
    apply_prob (default 0.5, as per Sidon paper).
    """
    aug = audio.copy()
    fns = [
        lambda a: _apply_reverberation(a, sr),
        lambda a: _apply_noise(a, sr, noise_dir),
        lambda a: _apply_band_limitation(a, sr),
        lambda a: _apply_clipping(a),
        lambda a: _apply_codec(a, sr),
        lambda a: _apply_packet_loss(a, sr),
    ]
    for fn in fns:
        if random.random() < apply_prob:
            aug = fn(aug)
    return aug.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-utterance worker
# ---------------------------------------------------------------------------

def _process_one(
    uttid: str,
    wav_path: str,
    noise_dir: str,
    out_wav_dir: str,
    n_repeat: int,
    sr: int,
    apply_prob: float,
    min_samples: int,
) -> Tuple[List[str], List[str]]:
    """Process one utterance, generating n_repeat noisy versions.

    Returns (noisy_lines, clean_lines) — empty lists if utterance is skipped.
    """
    try:
        info = sf.info(wav_path)
    except Exception as exc:
        logger.warning("SKIP %s — cannot read info: %s", uttid, exc)
        return [], []

    orig_samples = info.frames
    orig_sr      = info.samplerate
    if orig_sr != sr:
        est_samples = int(orig_samples * sr / orig_sr)
    else:
        est_samples = orig_samples

    if est_samples < min_samples:
        logger.debug("SKIP %s — too short (%d < %d samples)", uttid, est_samples, min_samples)
        return [], []

    try:
        clean, read_sr = sf.read(wav_path, dtype="float32")
        if clean.ndim > 1:
            clean = clean.mean(axis=1)
        if read_sr != sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(read_sr, sr)
            clean = resample_poly(clean, sr // g, read_sr // g)
    except Exception as exc:
        logger.warning("SKIP %s — read/resample failed: %s", uttid, exc)
        return [], []

    n = len(clean)
    if n < min_samples:
        logger.debug("SKIP %s — too short after resample (%d samples)", uttid, n)
        return [], []

    if np.abs(clean).max() < 1e-6:
        logger.warning("SKIP %s — silent audio", uttid)
        return [], []

    noisy_lines = []
    clean_lines = []
    clean_dir   = out_wav_dir.replace("/noisy/", "/clean/")
    os.makedirs(clean_dir, exist_ok=True)

    for rep in range(n_repeat):
        noisy      = degrade_once(clean, sr, noise_dir, apply_prob)
        new_id     = f"{uttid}_{rep}"
        noisy_path = os.path.join(out_wav_dir, f"{new_id}.wav")
        clean_path = os.path.join(clean_dir,   f"{new_id}.wav")

        try:
            sf.write(noisy_path, noisy, samplerate=sr)
            check = sf.info(noisy_path)
            if check.frames == 0:
                raise ValueError("Written file has 0 frames")
            sf.write(clean_path, clean, samplerate=sr)
        except Exception as exc:
            logger.warning("SKIP %s rep %d — write/verify failed: %s", uttid, rep, exc)
            for p in [noisy_path, clean_path]:
                try:
                    os.remove(p)
                except OSError:
                    pass
            continue

        noisy_lines.append(f"{new_id} {noisy_path}")
        clean_lines.append(f"{new_id} {clean_path}")

    return noisy_lines, clean_lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_parser():
    p = argparse.ArgumentParser(description="Sidon degradation pipeline")
    p.add_argument("--clean_wav_scp", required=True,
                   help="Kaldi-style wav.scp for clean utterances")
    p.add_argument("--noise_dir",     required=True,
                   help="Directory containing noise WAV/FLAC files (recursive)")
    p.add_argument("--out_dir",       required=True,
                   help="Output directory for paired data")
    p.add_argument("--n_repeat",      type=int,   default=4,
                   help="Number of noisy versions per clean utterance (default 4)")
    p.add_argument("--sr",            type=int,   default=16000,
                   help="Target sample rate: 16000 for FP, 48000 for vocoder")
    p.add_argument("--apply_prob",    type=float, default=0.5,
                   help="Probability of applying each degradation (default 0.5)")
    p.add_argument("--min_duration",  type=float, default=0.5,
                   help="Skip utterances shorter than this (seconds, default 0.5)")
    p.add_argument("--nj",            type=int,   default=8,
                   help="Number of parallel workers (default 8)")
    return p


def main():
    args = get_parser().parse_args()

    min_samples = int(args.min_duration * args.sr)

    out_wav_dir   = os.path.join(args.out_dir, "noisy", "wavs")
    out_noisy_scp = os.path.join(args.out_dir, "noisy", "wav.scp")
    out_clean_scp = os.path.join(args.out_dir, "clean", "wav.scp")
    os.makedirs(out_wav_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "clean"), exist_ok=True)

    entries = []
    with open(args.clean_wav_scp) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))

    logger.info(
        "Processing %d utterances × %d repeats (sr=%d, min=%.1fs, apply_prob=%.1f)",
        len(entries), args.n_repeat, args.sr, args.min_duration, args.apply_prob,
    )

    all_noisy, all_clean = [], []
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.nj) as ex:
        futures = {
            ex.submit(
                _process_one,
                uttid, path, args.noise_dir,
                out_wav_dir, args.n_repeat, args.sr, args.apply_prob,
                min_samples,
            ): uttid
            for uttid, path in entries
        }
        for i, fut in enumerate(as_completed(futures), 1):
            noisy_lines, clean_lines = fut.result()
            if not noisy_lines:
                skipped += 1
            all_noisy.extend(noisy_lines)
            all_clean.extend(clean_lines)
            if i % 500 == 0:
                logger.info("  %d / %d done (skipped so far: %d)", i, len(entries), skipped)

    all_noisy.sort()
    all_clean.sort()

    with open(out_noisy_scp, "w") as f:
        f.write("\n".join(all_noisy) + "\n")
    with open(out_clean_scp, "w") as f:
        f.write("\n".join(all_clean) + "\n")

    logger.info(
        "Done. Generated %d pairs from %d utterances (%d skipped).",
        len(all_noisy), len(entries), skipped,
    )
    logger.info("noisy scp: %s", out_noisy_scp)
    logger.info("clean scp: %s", out_clean_scp)


if __name__ == "__main__":
    main()