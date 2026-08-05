#!/usr/bin/env python3
"""Loudness-normalize all wavs listed in a wav.scp to a target dBFS (RMS).

Usage:
    python3 normalize_wavscp.py \
        --wav_scp  exp/restored_xeus_multi_all/test-clean/wav.scp \
        --out_dir  exp/restored_xeus_multi_all/test-clean-normalize \
        --target_dbfs -20.0
CUDA_VISIBLE_DEVICES=3    python3 normalize_wavscp.py \
        --wav_scp exp_ver1/restored_sidon_libritts_test-other-degrad/wav.scp \
        --out_dir exp_ver1/restored_sidon_libritts_test-other-degrad-normalize/wav \
        --target_dbfs -20.0
Writes:
    <out_dir>/<uttid>.wav          (normalized copies)
    <out_dir>/wav.scp              (new scp pointing to normalized files)
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def normalize_dbfs(wav: np.ndarray, target_dbfs: float, max_gain_db: float = 20.0) -> np.ndarray:
    rms = np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-12)
    if rms < 1e-8:
        return wav.astype(np.float32)
    gain_db = np.clip(target_dbfs - 20 * np.log10(rms), -max_gain_db, max_gain_db)
    out = wav * (10 ** (gain_db / 20))
    peak = np.abs(out).max()
    if peak > 0.99:
        out = out / peak * 0.99
    return out.astype(np.float32)


def read_scp(path: str) -> dict:
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_scp", required=True,
                   help="Input wav.scp (uttid <space> path per line)")
    p.add_argument("--out_dir", required=True,
                   help="Output directory for normalized wavs + new wav.scp")
    p.add_argument("--target_dbfs", type=float, default=-20.0,
                   help="Target RMS loudness in dBFS (default: -20.0)")
    p.add_argument("--max_gain_db", type=float, default=20.0,
                   help="Max absolute gain applied, to avoid extreme boosts "
                        "on near-silent utterances (default: 20.0)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_map = read_scp(args.wav_scp)
    print(f"Found {len(wav_map)} utterances in {args.wav_scp}")
    print(f"Target: {args.target_dbfs} dBFS  ->  {out_dir}")

    scp_lines = []
    n_ok, n_fail = 0, 0
    for uid, path in wav_map.items():
        try:
            wav, sr = sf.read(path)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            normed = normalize_dbfs(wav, args.target_dbfs, args.max_gain_db)
            out_path = out_dir / f"{uid}.wav"
            sf.write(str(out_path), normed, sr)
            scp_lines.append(f"{uid} {out_path.resolve()}")
            n_ok += 1
        except Exception as e:
            print(f"  FAILED {uid} ({path}): {e}")
            n_fail += 1

    scp_path = out_dir / "wav.scp"
    with open(scp_path, "w") as f:
        f.write("\n".join(scp_lines) + "\n")

    print(f"Done. {n_ok} ok, {n_fail} failed -> {scp_path}")


if __name__ == "__main__":
    main()