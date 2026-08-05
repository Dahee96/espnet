#!/usr/bin/env python3
"""Segment-level evaluation for long-form data (AMI, Fisher).

For AMI IHM: use the existing segments file (forced-alignment based).
For Fisher:  use Lhotse-manifest-derived segments (local/fisher_segments_from_lhotse.py).

For WER on AMI: reference text comes from ami_ihm_longform/text (segment-level).
For WER on Fisher: reference text comes from fisher_longform/text (segment-level,
                   built from the Lhotse supervision manifest).

SpkSim: computed between restored segment and NOISY segment (input ↔ restored),
        NOT against a clean reference — Fisher has no clean reference, and AMI
        IHM close-talk is itself the noisy input here (restoration target).
        --spksim_backend selects the embedding model: wavlm | ecapa | rawnet3,
        following Samuele's windowed (3s/0.5s stride) averaging protocol —
        see local/spksim_models.py and local/score.py for details.

Usage:
------
# AMI IHM evaluation (ECAPA, Samuele's protocol)
python local/score_ami_fisher.py \
    --restored_scp  exp/restored_espnet-sidon/ami_ihm_longform/wav.scp \
    --noisy_scp     data/ami_ihm_longform/wav.scp \
    --segments      data/ami_ihm_longform/segments \
    --text          data/ami_ihm_longform/text \
    --out_dir       exp/scores/espnet-sidon/ami_ihm \
    --dataset_name  ami_ihm \
    --metrics wer dnsmos nisqa spksim utmos squim_noref \
    --spksim_backend ecapa \
    --device cuda

# Fisher evaluation (RawNet3, matches VERSA's underlying model)
python local/score_ami_fisher.py \
    --restored_scp  exp/restored_espnet-sidon/fisher_longform/wav.scp \
    --noisy_scp     data/fisher_longform/wav.scp \
    --segments      data/fisher_longform/segments \
    --text          data/fisher_longform/text \
    --out_dir       exp/scores/espnet-sidon/fisher \
    --dataset_name  fisher \
    --metrics wer dnsmos nisqa spksim utmos squim_noref \
    --spksim_backend rawnet3 \
    --device cuda

NOTE: --restored_scp can point to either:
  (a) segment-level wav.scp (if inference was run in segment mode, uttid matches)
  (b) full-recording wav.scp (if inference was run in full-recording mode,
      this script will slice segments from the full restored recording on-the-fly)
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_SR = 16000


# =============================================================================
# I/O
# =============================================================================

def read_scp(path):
    d = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split(None, 1)
            if len(p) == 2:
                d[p[0]] = p[1]
    return d


def read_text(path):
    d = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split(None, 1)
            if len(p) == 2:
                d[p[0]] = p[1]
    return d


def read_segments(path):
    """Returns {utt_id: (rec_id, start_sec, end_sec)}."""
    d = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 4:
                d[p[0]] = (p[1], float(p[2]), float(p[3]))
    return d


def load_wav(path, target_sr=TARGET_SR):
    """Load from file path or sox pipe command ending with '|'.

    wav.scp entries built by sox-based recipes (e.g. AMI) look like:
        sox /DB/AMI/.../ES2004a.Headset-0.wav -r 16000 -c 1 -t wavpcm - |
    soundfile.read() cannot open these directly — they must be run as a
    shell command and the resulting raw PCM bytes parsed.
    """
    path = path.strip()
    if path.endswith("|"):
        import subprocess
        proc = subprocess.run(
            path[:-1].strip(), shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"sox pipe failed: {proc.stderr.decode()}\ncmd: {path}")
        wav = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return wav.astype(np.float32)

    import librosa
    wav, sr = sf.read(path, always_2d=True)
    wav = wav.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


def load_wav_torch(path, target_sr=TARGET_SR):
    """Load as torch.Tensor, supporting sox pipe commands via load_wav()."""
    path_stripped = path.strip()
    if path_stripped.endswith("|"):
        wav_np = load_wav(path_stripped, target_sr=target_sr)
        return torch.from_numpy(wav_np)

    import torchaudio
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def _slice_segment(wav_map, rec_id, start, end, target_sr=TARGET_SR):
    """Load a time segment from a recording, resampling to target_sr."""
    wav = load_wav(wav_map[rec_id], target_sr=target_sr)
    s = int(start * target_sr)
    e = int(end   * target_sr)
    return wav[s:e].astype(np.float32)


# =============================================================================
# Build segment-level scp from full-recording scp + segments
# =============================================================================

def build_segment_scps(
    restored_scp: Dict[str, str],
    noisy_scp: Dict[str, str],
    segments: Dict[str, Tuple[str, float, float]],
    tmp_dir: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Slice restored and noisy recordings into per-segment wav files.

    If restored_scp already has segment-level uttids (i.e., uttid in segments),
    those are used directly without slicing.

    Returns (seg_restored_scp, seg_noisy_scp) — both at 16kHz.
    """
    seg_restored = {}
    seg_noisy    = {}

    os.makedirs(tmp_dir, exist_ok=True)

    _restored_cache = {}
    _noisy_cache    = {}

    for utt_id, (rec_id, start, end) in segments.items():
        if utt_id in restored_scp:
            seg_restored[utt_id] = restored_scp[utt_id]
        elif rec_id in restored_scp:
            if rec_id not in _restored_cache:
                _restored_cache[rec_id] = load_wav(restored_scp[rec_id],
                                                    target_sr=TARGET_SR)
            wav_r = _restored_cache[rec_id]
            s = int(start * TARGET_SR)
            e = int(end   * TARGET_SR)
            seg_wav = wav_r[s:min(e, len(wav_r))]
            if len(seg_wav) < 100:
                continue
            out_path = os.path.join(tmp_dir, f"restored_{utt_id}.wav")
            sf.write(out_path, seg_wav, TARGET_SR)
            seg_restored[utt_id] = out_path
        else:
            continue

        if utt_id in noisy_scp:
            seg_noisy[utt_id] = noisy_scp[utt_id]
        elif rec_id in noisy_scp:
            if rec_id not in _noisy_cache:
                _noisy_cache[rec_id] = load_wav(noisy_scp[rec_id],
                                                 target_sr=TARGET_SR)
            wav_n = _noisy_cache[rec_id]
            s = int(start * TARGET_SR)
            e = int(end   * TARGET_SR)
            seg_wav = wav_n[s:min(e, len(wav_n))]
            if len(seg_wav) < 100:
                continue
            out_path = os.path.join(tmp_dir, f"noisy_{utt_id}.wav")
            sf.write(out_path, seg_wav, TARGET_SR)
            seg_noisy[utt_id] = out_path

    logger.info("Segment scps built: %d restored, %d noisy",
                len(seg_restored), len(seg_noisy))
    return seg_restored, seg_noisy


# =============================================================================
# Import metric functions from score.py
# =============================================================================

def _import_score_module():
    """Import metric compute functions from score.py (or legacy score_v2.py)."""
    import importlib.util, sys
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in [os.path.join(here, "score.py"),
                      os.path.join(here, "score_v2.py"),
                      "local/score.py", "local/score_v2.py"]:
        if os.path.exists(candidate):
            spec = importlib.util.spec_from_file_location("score_module", candidate)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules["score_module"] = mod
            return mod
    raise ImportError("Cannot find local/score.py or local/score_v2.py")


# =============================================================================
# Main
# =============================================================================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--restored_scp",  required=True,
                   help="Restored wav.scp. Can be segment-level or full-recording.")
    p.add_argument("--noisy_scp",     required=True,
                   help="Noisy (input) wav.scp. Can be segment-level or full-recording.")
    p.add_argument("--segments",      required=True,
                   help="Kaldi segments file (from AMI recipe or Lhotse-derived).")
    p.add_argument("--text",          default=None,
                   help="Kaldi text file for WER. Optional.")
    p.add_argument("--out_dir",       required=True)
    p.add_argument("--dataset_name",  default="longform",
                   help="Name tag for logging (ami_ihm, fisher, etc.)")
    p.add_argument("--metrics",       nargs="+",
                   default=["dnsmos", "nisqa", "spksim", "utmos", "squim_noref"],
                   choices=["wer", "dnsmos", "nisqa", "spksim", "utmos",
                            "squim_noref", "squim_ref", "speechbertscore"])
    p.add_argument("--asr_model",     default="owsm-v3.1",
                   choices=["mms", "whisper-large-v3", "whisper-large-v3-turbo",
                            "owsm-v3", "owsm-v3.1"])
    p.add_argument("--spksim_backend", default="rawnet3",
                   choices=["wavlm", "ecapa", "rawnet3"],
                   help="Speaker embedding model for SpkSim (see score.py).")
    p.add_argument("--spksim_win_sec", type=float, default=3.0,
                   help="Sliding window length (s) for SpkSim.")
    p.add_argument("--spksim_hop_sec", type=float, default=1.5,
                   help="Sliding window stride (s) for SpkSim.")
    p.add_argument("--nisqa_model",   default="nisqa_pretrained_model")
    p.add_argument("--dnsmos_cache",  default=".dnsmos_cache")
    p.add_argument("--device",        default="cuda")
    return p.parse_args()


def main():
    args   = get_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info("Dataset: %s | Device: %s | SpkSim backend: %s",
                args.dataset_name, device, args.spksim_backend)

    restored_scp = read_scp(args.restored_scp)
    noisy_scp    = read_scp(args.noisy_scp)
    segments     = read_segments(args.segments)
    ref_text     = read_text(args.text) if args.text else {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = str(out_dir / "_seg_tmp")
    seg_restored, seg_noisy = build_segment_scps(
        restored_scp, noisy_scp, segments, tmp_dir,
    )

    if not seg_restored:
        logger.error("No segments could be built. Check wav.scp and segments file.")
        return

    sv = _import_score_module()
    summary = {}
    metrics = args.metrics

    if "wer" in metrics:
        if not ref_text:
            logger.warning("--text not provided; skipping WER")
        else:
            seg_text = {k: v for k, v in ref_text.items() if k in seg_restored}
            r = sv.compute_wer(seg_restored, seg_text, device, args.asr_model)
            summary["wer"] = r["wer"]
            with open(out_dir / "wer_per_utt.json", "w") as f:
                json.dump(r.get("wer_per_utt", {}), f, indent=2)

    if "dnsmos" in metrics:
        r = sv.compute_dnsmos(seg_restored, cache_dir=args.dnsmos_cache)
        summary["dnsmos"] = r["dnsmos"]
        with open(out_dir / "dnsmos_per_utt.json", "w") as f:
            json.dump(r["dnsmos_per_utt"], f, indent=2)

    if "nisqa" in metrics:
        r = sv.compute_nisqa(seg_restored, device, args.nisqa_model)
        summary["nisqa"] = r["nisqa"]
        with open(out_dir / "nisqa_per_utt.json", "w") as f:
            json.dump(r["nisqa_per_utt"], f, indent=2)

    if "spksim" in metrics:
        if not noisy_scp:
            logger.warning("--noisy_scp not provided; skipping SpkSim")
        else:
            # Pass the ORIGINAL (un-sliced) restored_scp/noisy_scp + segments
            # rather than the pre-sliced seg_restored/seg_noisy used by other
            # metrics: compute_spksim() does its own direct random-access
            # [start,end] read at native sample rate, THEN resamples to 16k,
            # matching Samuele's load_segment_16k() ordering exactly. Reusing
            # the whole-file-resampled-then-sliced seg_restored/seg_noisy
            # here would reintroduce the resample-order mismatch.
            r = sv.compute_spksim(
                restored_scp, noisy_scp, device,
                backend=args.spksim_backend,
                win_sec=args.spksim_win_sec,
                hop_sec=args.spksim_hop_sec,
                segments=segments,
            )
            summary["spksim"] = r["spksim"]
            with open(out_dir / "spksim_per_utt.json", "w") as f:
                json.dump(r["spksim_per_utt"], f, indent=2)

    if "utmos" in metrics:
        r = sv.compute_utmos(seg_restored, device)
        summary["utmos"] = r["utmos"]
        with open(out_dir / "utmos_per_utt.json", "w") as f:
            json.dump(r["utmos_per_utt"], f, indent=2)

    if "squim_noref" in metrics:
        r = sv.compute_squim_noref(seg_restored, device)
        summary.update({k: v for k, v in r.items() if not k.endswith("_per_utt")})
        with open(out_dir / "squim_noref_per_utt.json", "w") as f:
            json.dump(r.get("squim_noref_per_utt", {}), f, indent=2)

    with open(out_dir / "scores.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("SUMMARY [%s] → %s/scores.json", args.dataset_name, out_dir)
    for k, v in summary.items():
        logger.info("  %-20s : %.4f", k.upper(), v)
    logger.info("=" * 60)

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()