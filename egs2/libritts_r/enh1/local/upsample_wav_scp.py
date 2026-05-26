#!/usr/bin/env python3
"""Upsample WAV files from source SR to target SR and write a new wav.scp.

Used in Stage 3 to convert LibriTTS-R (24 kHz) to 48 kHz before the
degradation pipeline, since the Sidon vocoder only trains on 48 kHz data.

Usage
-----
python local/upsample_wav_scp.py \
    --wav_scp   data/train/wav.scp \
    --out_dir   data/train_48k \
    --target_sr 48000 \
    --nj        16
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _resample_one(
    uttid: str,
    src_path: str,
    out_wav_dir: str,
    target_sr: int,
) -> str:
    """Read, resample, write. Returns 'uttid out_path'."""
    wav, sr = sf.read(src_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != target_sr:
        g = gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)

    out_path = os.path.join(out_wav_dir, f"{uttid}.wav")
    sf.write(out_path, wav, samplerate=target_sr)
    return f"{uttid} {out_path}"


def get_parser():
    p = argparse.ArgumentParser(description="Upsample wav.scp to target SR")
    p.add_argument("--wav_scp",   required=True,  help="Input wav.scp")
    p.add_argument("--out_dir",   required=True,  help="Output directory")
    p.add_argument("--target_sr", type=int, default=48000)
    p.add_argument("--nj",        type=int, default=8)
    return p


def main():
    args = get_parser().parse_args()

    out_wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(out_wav_dir, exist_ok=True)

    entries = []
    with open(args.wav_scp) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))

    logger.info("Resampling %d files to %d Hz", len(entries), args.target_sr)

    results = []
    with ProcessPoolExecutor(max_workers=args.nj) as ex:
        futures = {
            ex.submit(_resample_one, uid, path, out_wav_dir, args.target_sr): uid
            for uid, path in entries
        }
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                results.append(fut.result())
            except Exception as e:
                logger.warning("Failed: %s", e)
            if i % 1000 == 0:
                logger.info("  %d / %d", i, len(entries))

    results.sort()
    scp_path = os.path.join(args.out_dir, "wav.scp")
    with open(scp_path, "w") as f:
        f.write("\n".join(results) + "\n")

    logger.info("Done. wav.scp written to %s", scp_path)


if __name__ == "__main__":
    main()
