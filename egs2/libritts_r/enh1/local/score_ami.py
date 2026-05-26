#!/usr/bin/env python3
"""Score restored AMI long-form speech.

Workflow:
  1. Load restored long-form wav (output of overlap-add inference)
  2. Load IHM mix long-form wav (clean reference, pre-mixed Headset 0~3)
  3. For each segment in --segments:
       - Slice restored wav at segment timing
       - Slice IHM mix wav at segment timing
       - Compute reference-based: PESQ, STOI, SI-SDR, CI-SDR
       - Compute reference-free:  DNSMOS, NISQA, SpkSim
  4. WER computed on full-meeting transcript

All metrics report mean ± 95% CI.

Usage:
CUDA_VISIBLE_DEVICES=1    python3 local/score_ami.py \
        --restored_scp  exp/restored_sidon_ami_sdm_longform/wav.scp \
        --noisy_scp     data/ami_sdm_longform/wav.scp \
        --ihm_mix_scp   data/ami_ihm_mix/wav.scp \
        --segments      data/ami_ihm_mix/segments \
        --text          data/ami_sdm_longform/text \
        --out_dir       exp/scores/sidon_ami \
        --device        cuda

Prerequisites:
    pip install pesq pystoi onnxruntime requests soundfile librosa nisqa scipy
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats
import torch

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

DNSMOS_ONNX_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/"
    "master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)
SR = 16000  # all audio processed at 16kHz


# =============================================================================
# I/O utilities
# =============================================================================

def read_scp(path: str) -> Dict[str, str]:
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def read_text(path: str) -> Dict[str, str]:
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def read_segments(path: str) -> List[Tuple[str, str, float, float]]:
    """Returns list of (uttid, rec_id, start_sec, end_sec)."""
    segs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                segs.append((parts[0], parts[1], float(parts[2]), float(parts[3])))
    return segs


def load_wav_full(wav_path: str, target_sr: int = SR) -> np.ndarray:
    """Load full wav from file path or sox pipe."""
    import soundfile as sf
    wav_path = wav_path.strip()
    if wav_path.endswith("|"):
        cmd  = wav_path[:-1].strip()
        proc = subprocess.run(cmd, shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"sox failed: {proc.stderr.decode()}")
        wav = (np.frombuffer(proc.stdout, dtype=np.int16)
               .astype(np.float32) / 32768.0)
    else:
        import librosa
        wav, sr = sf.read(wav_path, always_2d=True)
        wav = wav.mean(axis=1).astype(np.float32)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


def slice_segment(wav: np.ndarray, start: float, end: float,
                  sr: int = SR) -> np.ndarray:
    """Slice wav array at given time boundaries."""
    s = int(start * sr)
    e = int(end   * sr)
    e = min(e, len(wav))
    return wav[s:e]


# =============================================================================
# Statistics
# =============================================================================

def _ci95(values: List[float]) -> Tuple[float, float]:
    if len(values) < 2:
        return (float(np.mean(values)) if values else float("nan")), float("nan")
    mean = float(np.mean(values))
    ci   = float(scipy.stats.t.ppf(0.975, df=len(values)-1)
                 * scipy.stats.sem(values))
    return mean, ci


def _summarize(scores: Dict[str, float], name: str) -> Dict:
    vals = [v for v in scores.values() if not np.isnan(v)]
    mean, ci = _ci95(vals)
    logger.info("  %-10s: %.4f ± %.4f  (n=%d)", name.upper(), mean, ci, len(vals))
    return {"mean": mean, "ci95": ci, "n": len(vals)}


# =============================================================================
# Reference-based metrics (per segment)
# =============================================================================

def compute_pesq(
    restored_segs: Dict[str, np.ndarray],
    ref_segs: Dict[str, np.ndarray],
) -> Dict:
    logger.info("Computing PESQ ...")
    try:
        from pesq import pesq as pesq_fn
    except ImportError:
        logger.error("pip install pesq")
        return {"pesq": {"mean": float("nan"), "ci95": float("nan"), "n": 0},
                "pesq_per_seg": {}}

    scores = {}
    for uttid in restored_segs:
        if uttid not in ref_segs:
            continue
        try:
            deg = restored_segs[uttid]
            ref = ref_segs[uttid]
            T   = min(len(deg), len(ref))
            if T < SR * 0.1:  # skip < 100ms
                continue
            scores[uttid] = float(pesq_fn(SR, ref[:T], deg[:T], "wb"))
        except Exception as e:
            logger.warning("PESQ failed %s: %s", uttid, e)

    return {"pesq": _summarize(scores, "PESQ"), "pesq_per_seg": scores}


def compute_stoi(
    restored_segs: Dict[str, np.ndarray],
    ref_segs: Dict[str, np.ndarray],
) -> Dict:
    logger.info("Computing STOI (extended) ...")
    try:
        from pystoi import stoi as stoi_fn
    except ImportError:
        logger.error("pip install pystoi")
        return {"stoi": {"mean": float("nan"), "ci95": float("nan"), "n": 0},
                "stoi_per_seg": {}}

    scores = {}
    for uttid in restored_segs:
        if uttid not in ref_segs:
            continue
        try:
            deg = restored_segs[uttid]
            ref = ref_segs[uttid]
            T   = min(len(deg), len(ref))
            if T < SR * 1.0:  # STOI needs >= 1.0s
                continue
            scores[uttid] = float(stoi_fn(ref[:T], deg[:T], SR, extended=True))
        except Exception as e:
            logger.warning("STOI failed %s: %s", uttid, e)

    return {"stoi": _summarize(scores, "STOI"), "stoi_per_seg": scores}


def _si_sdr(ref: np.ndarray, deg: np.ndarray) -> float:
    ref = ref - ref.mean()
    deg = deg - deg.mean()
    alpha = np.dot(deg, ref) / (np.dot(ref, ref) + 1e-8)
    proj  = alpha * ref
    noise = deg - proj
    return float(10 * np.log10(
        (np.dot(proj, proj) + 1e-8) / (np.dot(noise, noise) + 1e-8)))


def compute_si_sdr(
    restored_segs: Dict[str, np.ndarray],
    ref_segs: Dict[str, np.ndarray],
) -> Dict:
    logger.info("Computing SI-SDR ...")
    scores = {}
    for uttid in restored_segs:
        if uttid not in ref_segs:
            continue
        try:
            deg = restored_segs[uttid]
            ref = ref_segs[uttid]
            T   = min(len(deg), len(ref))
            if T < SR * 0.1:
                continue
            scores[uttid] = _si_sdr(ref[:T], deg[:T])
        except Exception as e:
            logger.warning("SI-SDR failed %s: %s", uttid, e)

    summary = _summarize(scores, "SI-SDR")

    # CI-SDR: same values, just reported with CI (already in summary)
    logger.info("  %-10s: %.4f ± %.4f  (n=%d) [= CI-SDR]",
                "CI-SDR", summary["mean"], summary["ci95"], summary["n"])

    return {"si_sdr": summary, "si_sdr_per_seg": scores}


# =============================================================================
# Reference-free metrics (per segment)
# =============================================================================

def compute_dnsmos(restored_segs: Dict[str, np.ndarray],
                   cache_dir: str = ".dnsmos_cache") -> Dict:
    logger.info("Computing DNSMOS ...")
    try:
        import onnxruntime as ort
        import requests
    except ImportError:
        logger.error("pip install onnxruntime requests")
        return {"dnsmos": {"mean": float("nan"), "ci95": float("nan"), "n": 0},
                "dnsmos_per_seg": {}}

    os.makedirs(cache_dir, exist_ok=True)
    onnx_path = os.path.join(cache_dir, "sig_bak_ovr.onnx")
    if not os.path.exists(onnx_path):
        logger.info("Downloading DNSMOS ONNX ...")
        r = requests.get(DNSMOS_ONNX_URL, timeout=60)
        r.raise_for_status()
        with open(onnx_path, "wb") as f:
            f.write(r.content)

    sess      = ort.InferenceSession(onnx_path)
    N_SAMPLES = 144160  # 9.01s at 16kHz
    p_ovr     = np.poly1d([-0.06766283, 1.11546468, 0.04602535])

    scores = {}
    for uttid, wav in restored_segs.items():
        try:
            if len(wav) < N_SAMPLES:
                wav_in = np.pad(wav, (0, N_SAMPLES - len(wav)))
            else:
                wav_in = wav[:N_SAMPLES]
            out = sess.run(None, {"input_1": wav_in[np.newaxis, :].astype(np.float32)})[0][0]
            scores[uttid] = float(np.clip(p_ovr(out[2]), 1.0, 5.0))
        except Exception as e:
            logger.warning("DNSMOS failed %s: %s", uttid, e)

    return {"dnsmos": _summarize(scores, "DNSMOS"), "dnsmos_per_seg": scores}


def compute_nisqa(restored_segs: Dict[str, np.ndarray],
                  device: str,
                  nisqa_model_path: str = "nisqa_pretrained_model") -> Dict:
    logger.info("Computing NISQA ...")
    try:
        from nisqa.NISQA_model import nisqaModel
        import soundfile as sf2
    except ImportError:
        logger.error("pip install nisqa")
        return {"nisqa": {"mean": float("nan"), "ci95": float("nan"), "n": 0},
                "nisqa_per_seg": {}}

    if not os.path.exists(nisqa_model_path):
        logger.error("NISQA model not found: %s", nisqa_model_path)
        return {"nisqa": {"mean": float("nan"), "ci95": float("nan"), "n": 0},
                "nisqa_per_seg": {}}

    tmp_dir = tempfile.mkdtemp()
    try:
        for uttid, wav in restored_segs.items():
            sf2.write(os.path.join(tmp_dir, f"{uttid}.wav"), wav, SR)

        nisqa = nisqaModel({
            "mode": "predict_dir", "pretrained_model": nisqa_model_path,
            "deg": None, "data_dir": tmp_dir, "output_dir": None,
            "ms_channel": None, "device": device,
        })
        df     = nisqa.predict()
        scores = dict(zip(
            df["deg"].apply(lambda p: Path(p).stem),
            df["mos_pred"].astype(float),
        ))
    except Exception as e:
        logger.error("NISQA failed: %s", e)
        scores = {}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"nisqa": _summarize(scores, "NISQA"), "nisqa_per_seg": scores}


def compute_spksim(
    restored_segs: Dict[str, np.ndarray],
    noisy_segs: Dict[str, np.ndarray],
    device: str,
) -> Dict:
    logger.info("Computing SpkSim (noisy ↔ restored, per segment) ...")
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel
    import torch.nn.functional as F

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "microsoft/wavlm-base-plus-sv")
    model = WavLMModel.from_pretrained(
        "microsoft/wavlm-base-plus-sv").to(device).eval()

    def _embed(wav_np):
        inputs = extractor(wav_np, sampling_rate=SR,
                           return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1)
        return F.normalize(emb, dim=-1)

    scores = {}
    uttids = [u for u in restored_segs if u in noisy_segs]
    for i, uttid in enumerate(uttids):
        try:
            sim = (_embed(restored_segs[uttid]) * _embed(noisy_segs[uttid])
                   ).sum().item()
            scores[uttid] = sim
        except Exception as e:
            logger.warning("SpkSim failed %s: %s", uttid, e)
        if (i+1) % 500 == 0:
            logger.info("  SpkSim: %d/%d", i+1, len(uttids))

    return {"spksim": _summarize(scores, "SpkSim"), "spksim_per_seg": scores}


# =============================================================================
# WER (full-meeting level)
# =============================================================================

def _normalize(text: str) -> str:
    text = text.upper()
    text = re.sub(r"['\-,\.\!\?;:\(\)]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _edit_distance(hyp, ref):
    n, m = len(ref), len(hyp)
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev, dp[0] = dp[0], i
        for j in range(1, m+1):
            temp  = dp[j]
            dp[j] = (prev if ref[i-1] == hyp[j-1]
                     else 1 + min(prev, dp[j], dp[j-1]))
            prev  = temp
    return dp[m]


def compute_wer(
    restored_scp: Dict[str, str],
    ref_text: Dict[str, str],
    device: str,
    batch_size: int = 4,
) -> Dict:
    """WER at meeting level (full long-form transcript)."""
    logger.info("Computing WER (full-meeting, facebook/mms-1b-all) ...")
    from transformers import Wav2Vec2ForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    model     = (Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
                 .to(device).eval())
    processor.tokenizer.set_target_lang("eng")
    model.load_adapter("eng")

    uttids = [k for k in restored_scp if k in ref_text]
    total_errors, total_words = 0, 0
    results = {}

    for uid in uttids:
        try:
            wav = load_wav_full(restored_scp[uid])
            # Process in 30s chunks for long-form WER
            chunk = 30 * SR
            hyp_words = []
            for start in range(0, len(wav), chunk):
                seg   = wav[start:start+chunk]
                inp   = processor([seg], sampling_rate=SR,
                                  return_tensors="pt", padding=True).to(device)
                with torch.inference_mode():
                    logits = model(**inp).logits
                hyp_words += processor.batch_decode(
                    torch.argmax(logits, -1))[0].split()

            ref_words    = _normalize(ref_text[uid]).split()
            hyp_words_n  = [_normalize(w) for w in hyp_words]
            errors        = _edit_distance(hyp_words_n, ref_words)
            total_errors += errors
            total_words  += max(len(ref_words), 1)
            results[uid]  = errors / max(len(ref_words), 1)
        except Exception as e:
            logger.error("WER failed %s: %s", uid, e)

    wer = total_errors / max(total_words, 1)
    logger.info("  WER (corpus): %.4f", wer)
    summary = _summarize(results, "WER")
    return {"wer": wer, "wer_ci": summary, "wer_per_meeting": results}


# =============================================================================
# Main
# =============================================================================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--restored_scp",  required=True,
                   help="wav.scp of restored long-form files")
    p.add_argument("--noisy_scp",     required=True,
                   help="wav.scp of noisy SDM long-form files (for SpkSim)")
    p.add_argument("--ihm_mix_scp",   required=True,
                   help="wav.scp of IHM mix long-form files (clean reference)")
    p.add_argument("--segments",      required=True,
                   help="segments file from data/ami_ihm_mix/segments")
    p.add_argument("--text",          required=True,
                   help="text file (full meeting transcripts)")
    p.add_argument("--out_dir",       default="exp/scores/ami")
    p.add_argument("--device",        default="cuda")
    p.add_argument("--nisqa_model",   default="nisqa_pretrained_model")
    p.add_argument("--dnsmos_cache",  default=".dnsmos_cache")
    p.add_argument("--metrics", nargs="+",
                   default=["wer","dnsmos","nisqa","spksim",
                            "pesq","stoi","si_sdr"],
                   choices=["wer","dnsmos","nisqa","spksim",
                            "pesq","stoi","si_sdr"])
    return p.parse_args()


def main():
    args   = get_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load SCPs and metadata
    restored_scp = read_scp(args.restored_scp)
    noisy_scp    = read_scp(args.noisy_scp)
    ihm_mix_scp  = read_scp(args.ihm_mix_scp)
    ref_text     = read_text(args.text)
    segments     = read_segments(args.segments)

    logger.info("Restored meetings : %d", len(restored_scp))
    logger.info("Segments          : %d", len(segments))

    # ── Pre-load long-form wavs ───────────────────────────────────────────
    # Cache all full-length wavs to avoid re-reading for each segment
    logger.info("Loading restored wavs ...")
    restored_full: Dict[str, np.ndarray] = {}
    for rec_id, wav_path in restored_scp.items():
        try:
            restored_full[rec_id] = load_wav_full(wav_path)
        except Exception as e:
            logger.error("Failed to load restored %s: %s", rec_id, e)

    logger.info("Loading IHM mix wavs ...")
    ihm_full: Dict[str, np.ndarray] = {}
    for rec_id, wav_path in ihm_mix_scp.items():
        try:
            ihm_full[rec_id] = load_wav_full(wav_path)
        except Exception as e:
            logger.error("Failed to load IHM %s: %s", rec_id, e)

    logger.info("Loading noisy SDM wavs ...")
    noisy_full: Dict[str, np.ndarray] = {}
    for rec_id, wav_path in noisy_scp.items():
        try:
            noisy_full[rec_id] = load_wav_full(wav_path)
        except Exception as e:
            logger.error("Failed to load noisy %s: %s", rec_id, e)

    # ── Slice segments from long-form wavs ───────────────────────────────
    # segments: (uttid, ihm_rec_id, start, end)
    # restored rec_id: replace _IHM → _SDM to find matching restored file
    logger.info("Slicing segments ...")
    restored_segs: Dict[str, np.ndarray] = {}
    ihm_ref_segs:  Dict[str, np.ndarray] = {}
    noisy_segs:    Dict[str, np.ndarray] = {}

    for uttid, ihm_rec_id, start, end in segments:
        # Map IHM rec_id → SDM rec_id
        # e.g. AMI_ES2004a_IHM → AMI_ES2004a_SDM
        sdm_rec_id = ihm_rec_id.replace("_IHM", "_SDM")

        # Restored segment
        if sdm_rec_id in restored_full:
            seg = slice_segment(restored_full[sdm_rec_id], start, end)
            if len(seg) > 0:
                restored_segs[uttid] = seg

        # IHM reference segment
        if ihm_rec_id in ihm_full:
            seg = slice_segment(ihm_full[ihm_rec_id], start, end)
            if len(seg) > 0:
                ihm_ref_segs[uttid] = seg

        # Noisy segment (for SpkSim)
        if sdm_rec_id in noisy_full:
            seg = slice_segment(noisy_full[sdm_rec_id], start, end)
            if len(seg) > 0:
                noisy_segs[uttid] = seg

    logger.info("Valid restored segments : %d", len(restored_segs))
    logger.info("Valid IHM ref segments  : %d", len(ihm_ref_segs))

    # ── Compute metrics ───────────────────────────────────────────────────
    metrics = args.metrics
    summary = {}

    if "wer" in metrics:
        r = compute_wer(restored_scp, ref_text, device)
        summary["wer"]    = r["wer"]
        summary["wer_ci"] = r["wer_ci"]
        with open(out_dir / "wer_per_meeting.json", "w") as f:
            json.dump(r["wer_per_meeting"], f, indent=2)

    if "dnsmos" in metrics:
        r = compute_dnsmos(restored_segs, cache_dir=args.dnsmos_cache)
        summary["dnsmos"] = r["dnsmos"]
        with open(out_dir / "dnsmos_per_seg.json", "w") as f:
            json.dump(r["dnsmos_per_seg"], f, indent=2)

    if "nisqa" in metrics:
        r = compute_nisqa(restored_segs, device, args.nisqa_model)
        summary["nisqa"] = r["nisqa"]
        with open(out_dir / "nisqa_per_seg.json", "w") as f:
            json.dump(r["nisqa_per_seg"], f, indent=2)

    if "spksim" in metrics:
        r = compute_spksim(restored_segs, noisy_segs, device)
        summary["spksim"] = r["spksim"]
        with open(out_dir / "spksim_per_seg.json", "w") as f:
            json.dump(r["spksim_per_seg"], f, indent=2)

    if "pesq" in metrics:
        r = compute_pesq(restored_segs, ihm_ref_segs)
        summary["pesq"] = r["pesq"]
        with open(out_dir / "pesq_per_seg.json", "w") as f:
            json.dump(r["pesq_per_seg"], f, indent=2)

    if "stoi" in metrics:
        r = compute_stoi(restored_segs, ihm_ref_segs)
        summary["stoi"] = r["stoi"]
        with open(out_dir / "stoi_per_seg.json", "w") as f:
            json.dump(r["stoi_per_seg"], f, indent=2)

    if "si_sdr" in metrics:
        r = compute_si_sdr(restored_segs, ihm_ref_segs)
        summary["si_sdr"] = r["si_sdr"]  # includes CI
        with open(out_dir / "si_sdr_per_seg.json", "w") as f:
            json.dump(r["si_sdr_per_seg"], f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    with open(out_dir / "scores.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("SUMMARY → %s/scores.json", out_dir)
    for k, v in summary.items():
        if isinstance(v, dict) and "mean" in v:
            logger.info("  %-12s: %.4f ± %.4f  (n=%d)",
                        k.upper(), v["mean"], v["ci95"], v["n"])
        else:
            logger.info("  %-12s: %.4f", k.upper(), float(v))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()