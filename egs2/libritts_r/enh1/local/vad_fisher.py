#!/usr/bin/env python3
"""Generate Kaldi segments file for Fisher (or any long-form) data using Silero-VAD.

Samuele's recommendation: evaluate on oracle-VAD speech segments only,
since UTMOS/DNSMOS assume clean speech input (silence frames distort scores).

For Fisher this is essential — Fisher conversations have long silence gaps
between speaker turns.

For AMI, the ESPnet recipe already provides a segments file (from forced
alignment / RTTM), so this script is only needed for Fisher.

Usage:
    python local/vad_fisher.py \
        --wav_scp    data/fisher_longform/wav.scp \
        --out_dir    data/fisher_longform \
        --min_speech 0.5 \
        --min_silence 0.3 \
        --nj 8

Outputs:
    data/fisher_longform/segments   — Kaldi segments: utt_id rec_id start end
    data/fisher_longform/text       — empty placeholder (no transcripts needed
                                      for non-WER metrics; fill from Fisher
                                      transcripts separately if WER needed)
"""

import argparse
import logging
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_SR = 16000


def read_wav_scp(path):
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def load_wav(path, target_sr=TARGET_SR):
    import librosa
    wav, sr = sf.read(path, always_2d=True)
    wav = wav.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


def vad_one(args_tuple):
    """Run Silero-VAD on a single recording. Returns list of (start, end) tuples."""
    rec_id, wav_path, min_speech_sec, min_silence_sec = args_tuple

    try:
        # Load Silero VAD (each worker loads its own copy)
        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        get_speech_timestamps = utils[0]

        wav_np = load_wav(wav_path)
        wav_t  = torch.from_numpy(wav_np)

        # Silero VAD expects 16kHz mono float32 tensor
        speech_timestamps = get_speech_timestamps(
            wav_t, vad_model,
            threshold=0.5,
            sampling_rate=TARGET_SR,
            min_speech_duration_ms=int(min_speech_sec * 1000),
            min_silence_duration_ms=int(min_silence_sec * 1000),
            return_seconds=True,   # get float seconds directly
        )

        segments = [(ts["start"], ts["end"]) for ts in speech_timestamps]
        return rec_id, segments, None
    except Exception as e:
        return rec_id, [], str(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_scp",      required=True)
    p.add_argument("--out_dir",      required=True)
    p.add_argument("--min_speech",   type=float, default=0.5,
                   help="Min speech segment duration in seconds")
    p.add_argument("--min_silence",  type=float, default=0.3,
                   help="Min silence duration to split segments")
    p.add_argument("--nj",           type=int, default=4,
                   help="Number of parallel workers")
    p.add_argument("--resume",       action="store_true",
                   help="If an existing segments file is present, skip "
                        "recordings that already have segments written "
                        "(only re-run previously-failed recordings).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Pre-download Silero-VAD ONCE in the main process ─────────────────────
    # Avoids a race condition where multiple worker processes simultaneously
    # try to download/unzip the same torch.hub repo (causes "hubconf.py not
    # found" / "Directory not empty: 'examples'" errors when several workers
    # race to extract master.zip into the same cache directory).
    logger.info("Pre-downloading Silero-VAD (single process, avoids worker race)...")
    torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    logger.info("Silero-VAD cached. Spawning %d workers...", args.nj)

    wav_map = read_wav_scp(args.wav_scp)
    logger.info("%d recordings to process with Silero-VAD", len(wav_map))

    # ── Resume: skip recordings that already have segments ───────────────────
    existing_segments = []
    already_done = set()
    seg_path = out_dir / "segments"
    if args.resume and seg_path.exists():
        with open(seg_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    existing_segments.append(tuple(parts[:2] + [float(parts[2]), float(parts[3])]))
                    already_done.add(parts[1])   # rec_id
        logger.info("Resume: %d recordings already have segments, skipping them.",
                    len(already_done))
        wav_map = {k: v for k, v in wav_map.items() if k not in already_done}
        logger.info("%d recordings remaining to process.", len(wav_map))

    if not wav_map:
        logger.info("Nothing to do (all recordings already processed).")
        return

    tasks = [
        (rec_id, wav_path, args.min_speech, args.min_silence)
        for rec_id, wav_path in sorted(wav_map.items())
    ]

    all_segments = []   # (utt_id, rec_id, start, end)
    errors = []

    with ProcessPoolExecutor(max_workers=args.nj) as executor:
        futures = {executor.submit(vad_one, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            rec_id, segments, err = fut.result()
            done += 1
            if err:
                logger.warning("VAD failed for %s: %s", rec_id, err)
                errors.append(rec_id)
                continue
            for i, (start, end) in enumerate(segments):
                # utt_id format: recid_000001-000234 (start/end in centiseconds)
                utt_id = f"{rec_id}_{int(start*100):06d}-{int(end*100):06d}"
                all_segments.append((utt_id, rec_id, start, end))

            if done % 50 == 0 or done == len(tasks):
                logger.info("  VAD progress: %d/%d", done, len(tasks))

    # Write segments file (merge with any resumed existing segments)
    all_segments_combined = existing_segments + all_segments
    seg_path = out_dir / "segments"
    with open(seg_path, "w") as f:
        for utt_id, rec_id, start, end in sorted(all_segments_combined):
            f.write(f"{utt_id} {rec_id} {start:.3f} {end:.3f}\n")
    logger.info("Wrote %d total segments to %s (%d new, %d resumed)",
                len(all_segments_combined), seg_path,
                len(all_segments), len(existing_segments))

    # Write/append placeholder text for newly processed segments
    text_path = out_dir / "text"
    existing_text_uttids = set()
    if text_path.exists():
        with open(text_path) as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if parts:
                    existing_text_uttids.add(parts[0])
    with open(text_path, "a" if text_path.exists() else "w") as f:
        for utt_id, *_ in sorted(all_segments):
            if utt_id not in existing_text_uttids:
                f.write(f"{utt_id} \n")
    logger.info("text file updated at %s (fill from Fisher .tdf for WER)", text_path)

    if errors:
        logger.warning("VAD failed for %d recordings: %s", len(errors), errors[:5])

    logger.info("Done. %d segments from %d recordings.", len(all_segments), len(wav_map))
    logger.info("Next: run inference on data/fisher_longform/wav.scp,")
    logger.info("      then score with --segments data/fisher_longform/segments")


if __name__ == "__main__":
    main()