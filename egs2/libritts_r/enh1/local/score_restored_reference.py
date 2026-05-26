#!/usr/bin/env python3
"""Evaluate restored speech: WER, NISQA, DNSMOS, SpkSim, PESQ, STOI, SI-SDR.

Metrics
-------
  Reference-free (always computed if data available):
    WER    : facebook/mms-1b-all + text normalization
    NISQA  : nisqa package pretrained model
    DNSMOS : Microsoft DNSMOS P.835 ONNX
    SpkSim : wavlm-base-plus-sv cosine sim (noisy ↔ restored)

  Reference-based (computed only when --ref_scp is provided):
    PESQ   : ITU-T P.862 perceptual quality (pip install pesq)
    STOI   : Short-Time Objective Intelligibility (pip install pystoi)
    SI-SDR : Scale-Invariant SDR (pip install torch-audiomentations or torchmetrics)

  All metrics report mean ± 95% CI.

Usage:
    # Reference-free (LibriTTS style)
    python3 local/score_restored.py \
        --restored_scp  exp/restored/wav.scp \
        --noisy_scp     data/test-clean-degrad/noisy/wav.scp \
        --text          data/test-clean-degrad/text \
        --out_dir       exp/scores/test-clean-degrad

    # Reference-based (AMI style: IHM as clean reference)
    python3 local/score_restored.py \
        --restored_scp  exp/restored_ami/wav.scp \
        --noisy_scp     data/ami_sdm_test/wav.scp \
        --ref_scp       data/ami_ihm_test/wav.scp \
        --text          data/ami_ihm_test/text \
        --out_dir       exp/scores/ami_test

Prerequisites:
    pip install pesq pystoi torchmetrics onnxruntime requests soundfile librosa nisqa
"""

import argparse
import json
import logging
import os
import re
import shutil
import tempfile
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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


def load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    import soundfile as sf
    import librosa
    wav, sr = sf.read(path, always_2d=True)
    wav = wav.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


def _ci95(values: List[float]) -> Tuple[float, float]:
    """Return (mean, 95% CI half-width) using t-distribution."""
    if len(values) < 2:
        return float(np.mean(values)) if values else float("nan"), float("nan")
    mean = float(np.mean(values))
    se   = float(scipy.stats.sem(values))
    ci   = float(scipy.stats.t.ppf(0.975, df=len(values) - 1) * se)
    return mean, ci


def _summarize(scores: Dict[str, float], name: str) -> Dict:
    """Compute mean ± 95% CI and log."""
    vals = [v for v in scores.values() if not np.isnan(v)]
    mean, ci = _ci95(vals)
    logger.info("  %-8s: %.4f ± %.4f  (n=%d)", name.upper(), mean, ci, len(vals))
    return {"mean": mean, "ci95": ci, "n": len(vals)}


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"['\-,\.\!\?;:\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def _edit_distance(hyp: List[str], ref: List[str]) -> int:
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp  = dp[j]
            dp[j] = (prev if ref[i-1] == hyp[j-1]
                     else 1 + min(prev, dp[j], dp[j-1]))
            prev  = temp
    return dp[m]


def compute_wer(
    restored_scp: Dict[str, str],
    ref_text: Dict[str, str],
    device: str,
    batch_size: int = 16,
) -> Dict:
    logger.info("Computing WER ...")
    from transformers import Wav2Vec2ForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    model     = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(device).eval()
    processor.tokenizer.set_target_lang("eng")
    model.load_adapter("eng")

    uttids       = [k for k in restored_scp if k in ref_text]
    total_errors = 0
    total_words  = 0
    results      = {}

    for start in range(0, len(uttids), batch_size):
        batch_ids = uttids[start:start + batch_size]
        wavs      = [load_wav(restored_scp[u], 16000) for u in batch_ids]
        inputs    = processor(
            wavs, sampling_rate=16000, return_tensors="pt", padding=True,
        ).to(device)
        with torch.inference_mode():
            logits = model(
                input_values=inputs["input_values"],
                attention_mask=inputs.get("attention_mask"),
            ).logits
        hypotheses = processor.batch_decode(torch.argmax(logits, dim=-1))
        for uid, hyp in zip(batch_ids, hypotheses):
            hyp_words = _normalize_text(hyp).split()
            ref_words = _normalize_text(ref_text[uid]).split()
            errors    = _edit_distance(hyp_words, ref_words)
            total_errors += errors
            total_words  += max(len(ref_words), 1)
            results[uid]  = {
                "wer":        errors / max(len(ref_words), 1),
                "hypothesis": " ".join(hyp_words),
                "reference":  " ".join(ref_words),
            }
        done = min(start + batch_size, len(uttids))
        if done % 200 == 0 or done == len(uttids):
            logger.info("  WER progress: %d/%d", done, len(uttids))

    wer_scores = {uid: v["wer"] for uid, v in results.items()}
    summary    = _summarize(wer_scores, "WER")
    # WER is reported as total (not per-utt mean) by convention
    total_wer  = total_errors / max(total_words, 1)
    logger.info("  WER (corpus-level): %.4f", total_wer)

    hyp_lines = [f"{uid} {v['hypothesis']}" for uid, v in results.items()]
    return {
        "wer":         total_wer,
        "wer_ci":      summary,
        "wer_per_utt": wer_scores,
        "wer_details": results,
        "hyp_lines":   hyp_lines,
    }


# ---------------------------------------------------------------------------
# NISQA
# ---------------------------------------------------------------------------

def compute_nisqa(
    restored_scp: Dict[str, str],
    device: str,
    nisqa_model_path: str = "nisqa_pretrained_model",
) -> Dict:
    logger.info("Computing NISQA ...")
    try:
        from nisqa.NISQA_model import nisqaModel
    except ImportError:
        logger.error("nisqa not installed: pip install nisqa")
        return {"nisqa": float("nan"), "nisqa_per_utt": {}}

    tmp_wav_dir = tempfile.mkdtemp()
    import soundfile as _sf
    for uid, orig_path in restored_scp.items():
        wav = load_wav(orig_path, target_sr=16000)
        _sf.write(os.path.join(tmp_wav_dir, f"{uid}.wav"), wav, 16000)

    try:
        nisqa = nisqaModel({
            "mode":             "predict_dir",
            "pretrained_model": nisqa_model_path,
            "deg":              None,
            "data_dir":         tmp_wav_dir,
            "output_dir":       None,
            "ms_channel":       None,
            "device":           device,
        })
        df     = nisqa.predict()
        scores = dict(zip(
            df["deg"].apply(lambda p: Path(p).stem),
            df["mos_pred"].astype(float),
        ))
    except Exception as e:
        logger.error("NISQA failed: %s", e)
        import traceback; traceback.print_exc()
        scores = {}
    finally:
        shutil.rmtree(tmp_wav_dir, ignore_errors=True)

    summary = _summarize(scores, "NISQA")
    return {"nisqa": summary, "nisqa_per_utt": scores}


# ---------------------------------------------------------------------------
# DNSMOS
# ---------------------------------------------------------------------------

def _download_dnsmos_onnx(cache_dir: str) -> str:
    import requests
    os.makedirs(cache_dir, exist_ok=True)
    onnx_path = os.path.join(cache_dir, "sig_bak_ovr.onnx")
    if not os.path.exists(onnx_path):
        logger.info("Downloading DNSMOS ONNX model ...")
        r = requests.get(DNSMOS_ONNX_URL, timeout=60)
        r.raise_for_status()
        with open(onnx_path, "wb") as f:
            f.write(r.content)
    return onnx_path


def _dnsmos_polyfit(sig, bak, ovr):
    p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
    p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052344 ])
    p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])
    return p_sig(sig), p_bak(bak), p_ovr(ovr)


def compute_dnsmos(
    restored_scp: Dict[str, str],
    cache_dir: str = ".dnsmos_cache",
) -> Dict:
    logger.info("Computing DNSMOS ...")
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime not installed: pip install onnxruntime")
        return {"dnsmos": float("nan"), "dnsmos_per_utt": {}}

    onnx_path = _download_dnsmos_onnx(cache_dir)
    sess      = ort.InferenceSession(onnx_path)
    N_SAMPLES = 144160  # 9.01s × 16kHz

    scores = {}
    for uttid, path in restored_scp.items():
        try:
            wav = load_wav(path, target_sr=16000)
            if len(wav) < N_SAMPLES:
                wav = np.pad(wav, (0, N_SAMPLES - len(wav)))
            wav = wav[:N_SAMPLES].astype(np.float32)
            out = sess.run(None, {"input_1": wav[np.newaxis, :]})[0][0]
            _, _, ovr_corr = _dnsmos_polyfit(*out)
            scores[uttid] = float(np.clip(ovr_corr, 1.0, 5.0))
        except Exception as e:
            logger.warning("DNSMOS failed for %s: %s", uttid, e)

    summary = _summarize(scores, "DNSMOS")
    return {"dnsmos": summary, "dnsmos_per_utt": scores}


# ---------------------------------------------------------------------------
# SpkSim
# ---------------------------------------------------------------------------

def compute_spksim(
    restored_scp: Dict[str, str],
    noisy_scp: Dict[str, str],
    device: str,
) -> Dict:
    logger.info("Computing SpkSim (noisy ↔ restored) ...")
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel
    import torch.nn.functional as F

    model_id  = "microsoft/wavlm-base-plus-sv"
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model     = WavLMModel.from_pretrained(model_id).to(device).eval()

    def _embed(wav_np):
        inputs = extractor(
            wav_np, sampling_rate=16000, return_tensors="pt", padding=True,
        ).to(device)
        with torch.inference_mode():
            out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1)
        return F.normalize(emb, dim=-1)

    uttids = [k for k in restored_scp if k in noisy_scp]
    scores = {}
    for i, uid in enumerate(uttids):
        try:
            sim = (_embed(load_wav(noisy_scp[uid], 16000))
                   * _embed(load_wav(restored_scp[uid], 16000))).sum().item()
            scores[uid] = sim
        except Exception as e:
            logger.warning("SpkSim failed for %s: %s", uid, e)
        if (i + 1) % 500 == 0:
            logger.info("  SpkSim progress: %d/%d", i + 1, len(uttids))

    summary = _summarize(scores, "SpkSim")
    return {"spksim": summary, "spksim_per_utt": scores}


# ---------------------------------------------------------------------------
# PESQ  (reference-based)
# ---------------------------------------------------------------------------

def compute_pesq(
    restored_scp: Dict[str, str],
    ref_scp: Dict[str, str],
) -> Dict:
    """PESQ: perceptual quality vs clean reference.

    Uses wideband (wb) mode at 16kHz.
    Score range: -0.5 (bad) to 4.5 (excellent).
    """
    logger.info("Computing PESQ (wideband, restored vs reference) ...")
    try:
        from pesq import pesq as pesq_fn
    except ImportError:
        logger.error("pesq not installed: pip install pesq")
        return {"pesq": {"mean": float("nan"), "ci95": float("nan"), "n": 0},
                "pesq_per_utt": {}}

    uttids = [k for k in restored_scp if k in ref_scp]
    scores = {}
    for uid in uttids:
        try:
            ref  = load_wav(ref_scp[uid],      target_sr=16000)
            deg  = load_wav(restored_scp[uid], target_sr=16000)
            # Trim to same length
            T    = min(len(ref), len(deg))
            score = pesq_fn(16000, ref[:T], deg[:T], "wb")
            scores[uid] = float(score)
        except Exception as e:
            logger.warning("PESQ failed for %s: %s", uid, e)

    summary = _summarize(scores, "PESQ")
    return {"pesq": summary, "pesq_per_utt": scores}


# ---------------------------------------------------------------------------
# STOI  (reference-based)
# ---------------------------------------------------------------------------

def compute_stoi(
    restored_scp: Dict[str, str],
    ref_scp: Dict[str, str],
) -> Dict:
    """STOI: intelligibility vs clean reference.

    Score range: 0 (bad) to 1 (excellent).
    Uses extended STOI (ESTOI) which handles noise better.
    """
    logger.info("Computing STOI (extended, restored vs reference) ...")
    try:
        from pystoi import stoi as stoi_fn
    except ImportError:
        logger.error("pystoi not installed: pip install pystoi")
        return {"stoi": {"mean": float("nan"), "ci95": float("nan"), "n": 0},
                "stoi_per_utt": {}}

    uttids = [k for k in restored_scp if k in ref_scp]
    scores = {}
    for uid in uttids:
        try:
            ref  = load_wav(ref_scp[uid],      target_sr=16000)
            deg  = load_wav(restored_scp[uid], target_sr=16000)
            T    = min(len(ref), len(deg))
            # extended=True → ESTOI, handles non-stationary noise better
            score = stoi_fn(ref[:T], deg[:T], 16000, extended=True)
            scores[uid] = float(score)
        except Exception as e:
            logger.warning("STOI failed for %s: %s", uid, e)

    summary = _summarize(scores, "STOI")
    return {"stoi": summary, "stoi_per_utt": scores}


# ---------------------------------------------------------------------------
# SI-SDR  (reference-based)
# ---------------------------------------------------------------------------

def compute_si_sdr(
    restored_scp: Dict[str, str],
    ref_scp: Dict[str, str],
) -> Dict:
    """SI-SDR: scale-invariant signal-to-distortion ratio vs clean reference.

    Higher is better (dB). Computed in pure numpy (no extra deps).
    CI reported alongside mean for long-form datasets like AMI.
    """
    logger.info("Computing SI-SDR (restored vs reference) ...")

    def _si_sdr_np(ref: np.ndarray, deg: np.ndarray) -> float:
        ref = ref - ref.mean()
        deg = deg - deg.mean()
        T   = min(len(ref), len(deg))
        ref, deg = ref[:T], deg[:T]
        alpha   = np.dot(deg, ref) / (np.dot(ref, ref) + 1e-8)
        proj    = alpha * ref
        noise   = deg - proj
        si_sdr  = 10 * np.log10(
            (np.dot(proj, proj) + 1e-8) / (np.dot(noise, noise) + 1e-8)
        )
        return float(si_sdr)

    uttids = [k for k in restored_scp if k in ref_scp]
    scores = {}
    for uid in uttids:
        try:
            ref = load_wav(ref_scp[uid],      target_sr=16000)
            deg = load_wav(restored_scp[uid], target_sr=16000)
            scores[uid] = _si_sdr_np(ref, deg)
        except Exception as e:
            logger.warning("SI-SDR failed for %s: %s", uid, e)

    summary = _summarize(scores, "SI-SDR")
    return {"si_sdr": summary, "si_sdr_per_utt": scores}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="Evaluate restored speech: WER / NISQA / DNSMOS / SpkSim "
                    "[/ PESQ / STOI / SI-SDR if --ref_scp provided]"
    )
    p.add_argument("--restored_scp", default=None)
    p.add_argument("--restored_dir", default=None)
    p.add_argument("--noisy_scp",    default=None,
                   help="Noisy input wav.scp — required for SpkSim")
    p.add_argument("--ref_scp",      default=None,
                   help="Clean reference wav.scp (e.g. IHM for AMI). "
                        "If provided, PESQ / STOI / SI-SDR are also computed.")
    p.add_argument("--text",         default=None,
                   help="Kaldi text file — required for WER")
    p.add_argument("--out_dir",      default="exp/scores")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--metrics",      nargs="+",
                   default=["wer", "nisqa", "dnsmos", "spksim"],
                   choices=["wer", "nisqa", "dnsmos", "spksim",
                            "pesq", "stoi", "si_sdr"])
    p.add_argument("--nisqa_model",  default="nisqa_pretrained_model")
    p.add_argument("--dnsmos_cache", default=".dnsmos_cache")
    return p.parse_args()


def main():
    args   = get_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    if args.restored_scp:
        restored_scp = read_scp(args.restored_scp)
    elif args.restored_dir:
        from score_restored import dir_to_scp
        restored_scp = dir_to_scp(args.restored_dir)
    else:
        raise ValueError("Provide --restored_scp or --restored_dir")
    logger.info("Restored: %d utterances", len(restored_scp))

    noisy_scp = read_scp(args.noisy_scp) if args.noisy_scp else {}
    ref_scp   = read_scp(args.ref_scp)   if args.ref_scp   else {}
    ref_text  = read_text(args.text)     if args.text       else {}

    # Auto-add reference-based metrics if ref_scp provided
    metrics = list(args.metrics)
    if ref_scp:
        for m in ["pesq", "stoi", "si_sdr"]:
            if m not in metrics:
                metrics.append(m)
        logger.info("--ref_scp provided → adding PESQ, STOI, SI-SDR")
    else:
        for m in ["pesq", "stoi", "si_sdr"]:
            if m in metrics:
                logger.warning(
                    "%s requires --ref_scp; skipping", m.upper())
                metrics.remove(m)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    # ── Reference-free ────────────────────────────────────────────────────
    if "wer" in metrics:
        if not ref_text:
            logger.warning("--text not provided; skipping WER")
        else:
            r = compute_wer(restored_scp, ref_text, device)
            summary["wer"] = r["wer"]
            summary["wer_ci"] = r["wer_ci"]
            with open(out_dir / "wer_per_utt.json",  "w") as f:
                json.dump(r["wer_per_utt"],  f, indent=2)
            with open(out_dir / "wer_details.json",  "w") as f:
                json.dump(r["wer_details"],  f, indent=2, ensure_ascii=False)
            with open(out_dir / "hypothesis",         "w") as f:
                f.write("\n".join(r["hyp_lines"]) + "\n")

    if "nisqa" in metrics:
        r = compute_nisqa(restored_scp, device,
                          nisqa_model_path=args.nisqa_model)
        summary["nisqa"] = r["nisqa"]
        with open(out_dir / "nisqa_per_utt.json", "w") as f:
            json.dump(r["nisqa_per_utt"], f, indent=2)

    if "dnsmos" in metrics:
        r = compute_dnsmos(restored_scp, cache_dir=args.dnsmos_cache)
        summary["dnsmos"] = r["dnsmos"]
        with open(out_dir / "dnsmos_per_utt.json", "w") as f:
            json.dump(r["dnsmos_per_utt"], f, indent=2)

    if "spksim" in metrics:
        if not noisy_scp:
            logger.warning("--noisy_scp not provided; skipping SpkSim")
        else:
            r = compute_spksim(restored_scp, noisy_scp, device)
            summary["spksim"] = r["spksim"]
            with open(out_dir / "spksim_per_utt.json", "w") as f:
                json.dump(r["spksim_per_utt"], f, indent=2)

    # ── Reference-based ───────────────────────────────────────────────────
    if "pesq" in metrics:
        r = compute_pesq(restored_scp, ref_scp)
        summary["pesq"] = r["pesq"]
        with open(out_dir / "pesq_per_utt.json", "w") as f:
            json.dump(r["pesq_per_utt"], f, indent=2)

    if "stoi" in metrics:
        r = compute_stoi(restored_scp, ref_scp)
        summary["stoi"] = r["stoi"]
        with open(out_dir / "stoi_per_utt.json", "w") as f:
            json.dump(r["stoi_per_utt"], f, indent=2)

    if "si_sdr" in metrics:
        r = compute_si_sdr(restored_scp, ref_scp)
        summary["si_sdr"] = r["si_sdr"]
        with open(out_dir / "si_sdr_per_utt.json", "w") as f:
            json.dump(r["si_sdr_per_utt"], f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    with open(out_dir / "scores.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("SUMMARY → %s/scores.json", out_dir)
    for k, v in summary.items():
        if isinstance(v, dict) and "mean" in v:
            logger.info("  %-10s : %.4f ± %.4f  (n=%d)",
                        k.upper(), v["mean"], v["ci95"], v["n"])
        else:
            logger.info("  %-10s : %.4f", k.upper(), v)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()