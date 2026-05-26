#!/usr/bin/env python3
"""Evaluate restored speech: WER, NISQA, DNSMOS, SpkSim.

Follows Sidon paper metrics:
  - WER    : facebook/mms-1b-all + text normalization (punctuation removal)
  - NISQA  : nisqa package pretrained model (pip install nisqa)
  - DNSMOS : Microsoft DNSMOS P.835 ONNX — input: [N, 144160] raw 16kHz samples
  - SpkSim : microsoft/wavlm-base-plus-sv cosine sim (noisy ↔ restored)

Prerequisites:
    pip install onnxruntime requests soundfile librosa nisqa

Usage:
    python3 local/score_restored.py \
        --restored_scp  exp/restored_sidon_orig_libritts_test-clean/wav.scp \
        --noisy_scp     data/libritts_test-clean/wav.scp \
        --text          data/libritts_test-clean/text \
        --out_dir       exp/scores/sidon_orig_libritts_test-clean \
        --nisqa_model   nisqa_pretrained_model \
        --device        cuda
"""
"""
# GPU 0: test-clean noisy + libritts-r
CUDA_VISIBLE_DEVICES=0 python3 local/score_restored.py \
    --restored_scp data/libritts_test-other/wav.scp \
    --noisy_scp    data/libritts_test-other/wav.scp \
    --text         data/libritts_test-other/text \
    --out_dir      exp/scores/libritts_test-other \
    --nisqa_model  nisqa_pretrained_model --device cuda &

# GPU 1: libritts-r test-clean
CUDA_VISIBLE_DEVICES=1 python3 local/score_restored.py \
    --restored_scp data/libritts_r_test-other/wav.scp \
    --noisy_scp    data/libritts_test-other/wav.scp \
    --text         data/libritts_test-other/text \
    --out_dir      exp/scores/libritts_r_test-other \
    --nisqa_model  nisqa_pretrained_model --device cuda &

# GPU 2: sidon orig test-clean
CUDA_VISIBLE_DEVICES=2 python3 local/score_restored.py \
    --restored_scp exp/restored_sidon_orig_libritts_test-other/wav.scp \
    --noisy_scp    data/libritts_test-other/wav.scp \
    --text         data/libritts_test-other/text \
    --out_dir      exp/scores/sidon_orig_libritts_test-other \
    --nisqa_model  nisqa_pretrained_model --device cuda &

# GPU 3: sidon espnet test-clean
CUDA_VISIBLE_DEVICES=3 python3 local/score_restored.py \
    --restored_scp exp/restored_sidon_libritts_test-clean/wav.scp \
    --noisy_scp    data/libritts_test-clean/wav.scp \
    --text         data/libritts_test-clean/text \
    --out_dir      exp/scores/sidon_espnet_libritts_test-clean \
    --nisqa_model  nisqa_pretrained_model --device cuda &

wait
echo "test-other done"

"""
"""
CUDA_VISIBLE_DEVICES=0 python3 local/score_restored.py \
    --restored_scp exp/restored_sidon_orig_libritts_test-other-degrad/wav.scp \
    --noisy_scp    data/libritts_test-other-degrad/noisy/wav.scp \
    --text         data/libritts_test-other-degrad/text \
    --out_dir      exp/scores/sidon_orig_libritts_test-other-degrad \
    --nisqa_model  nisqa_pretrained_model --device cuda &

# GPU 3: sidon espnet test-clean
CUDA_VISIBLE_DEVICES=1 python3 local/score_restored.py \
    --restored_scp exp/restored_sidon_libritts_test-other-degrad/wav.scp \
    --noisy_scp    data/libritts_test-other-degrad/noisy/wav.scp \
    --text         data/libritts_test-other-degrad/text \
    --out_dir      exp/scores/sidon_espnet_libritts_test-other-degrad \
    --nisqa_model  nisqa_pretrained_model --device cuda &

CUDA_VISIBLE_DEVICES=2 python3 local/score_restored.py \
    --restored_scp exp/restored_sidon_orig_libritts_test-clean-degrad/wav.scp \
    --noisy_scp    data/libritts_test-clean-degrad/noisy/wav.scp \
    --text         data/libritts_test-clean-degrad/text \
    --out_dir      exp/scores/sidon_orig_libritts_test-clean-degrad \
    --nisqa_model  nisqa_pretrained_model --device cuda &

# GPU 3: sidon espnet test-clean
CUDA_VISIBLE_DEVICES=0 python3 local/score_restored.py \
    --restored_scp exp/restored_sidon_libritts_test-clean-degrad/wav.scp \
    --noisy_scp    data/libritts_test-clean-degrad/noisy/wav.scp \
    --text         data/libritts_test-clean-degrad/text \
    --out_dir      exp/scores/sidon_espnet_libritts_test-clean-degrad \
    --nisqa_model  nisqa_pretrained_model --device cuda &

wait
echo "test degrad done"
"""
import argparse
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

DNSMOS_ONNX_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/"
    "master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)


# ---------------------------------------------------------------------------
# I/O utilities
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


def dir_to_scp(wav_dir: str) -> Dict[str, str]:
    d = {}
    for p in sorted(Path(wav_dir).glob("**/*.wav")):
        d[p.stem] = str(p)
    return d


def load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load wav → mono float32 numpy at target_sr."""
    import soundfile as sf
    import librosa
    wav, sr = sf.read(path, always_2d=True)
    wav = wav.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Text normalization before WER computation.

    Removes punctuation that ASR models may handle differently
    (hyphens, apostrophes, commas, periods, etc.).
    Matches common evaluation practice in speech restoration papers.
    """
    text = text.upper()
    text = re.sub(r"['\-,\.\!\?;:\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# WER  (facebook/mms-1b-all, batch inference + text normalization)
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
    logger.info("Computing WER with facebook/mms-1b-all (+ text normalization) ...")
    from transformers import Wav2Vec2ForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    model     = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(device).eval()
    processor.tokenizer.set_target_lang("eng")
    model.load_adapter("eng")

    uttids       = [k for k in restored_scp if k in ref_text]
    total_errors = 0
    total_words  = 0
    results      = {}
    logger.info("  %d utterances with reference text", len(uttids))

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
            running_wer = total_errors / max(total_words, 1)
            logger.info("  WER progress: %d/%d  running=%.4f",
                        done, len(uttids), running_wer)

    wer = total_errors / max(total_words, 1)
    logger.info("  WER: %.4f", wer)

    hyp_lines  = [f"{uid} {v['hypothesis']}" for uid, v in results.items()]
    wer_scores = {uid: v["wer"] for uid, v in results.items()}
    return {
        "wer":         wer,
        "wer_per_utt": wer_scores,
        "wer_details": results,
        "hyp_lines":   hyp_lines,
    }


# ---------------------------------------------------------------------------
# NISQA  (pip install nisqa)
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

    if not os.path.exists(nisqa_model_path):
        logger.error(
            "NISQA model not found: %s\n"
            "Download: wget https://github.com/gabrielmittag/NISQA/raw/master"
            "/weights/nisqa.tar -O %s",
            nisqa_model_path, nisqa_model_path,
        )
        return {"nisqa": float("nan"), "nisqa_per_utt": {}}

    # Pre-convert to 16kHz wav (NISQA's librosa backend may fail on 48kHz)
    tmp_wav_dir = tempfile.mkdtemp()
    resampled_paths = {}
    import soundfile as _sf
    for uid, orig_path in restored_scp.items():
        wav = load_wav(orig_path, target_sr=16000)
        tmp_path = os.path.join(tmp_wav_dir, f"{uid}.wav")
        _sf.write(tmp_path, wav, 16000)
        resampled_paths[uid] = tmp_path

    # Use predict_dir mode (avoids CSV parsing bug in predict_file mode)
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
        mean_score = float(np.mean(list(scores.values())))
    except Exception as e:
        logger.error("NISQA failed: %s", e)
        import traceback; traceback.print_exc()
        scores, mean_score = {}, float("nan")
    finally:
        shutil.rmtree(tmp_wav_dir, ignore_errors=True)

    logger.info("  NISQA: %.4f", mean_score)
    return {"nisqa": mean_score, "nisqa_per_utt": scores}


# ---------------------------------------------------------------------------
# DNSMOS  (Microsoft ONNX, input: [N, 144160] raw 16kHz samples)
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


def _dnsmos_polyfit(sig: float, bak: float, ovr: float):
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

    onnx_path  = _download_dnsmos_onnx(cache_dir)
    sess       = ort.InferenceSession(onnx_path)
    N_SAMPLES  = 144160  # 9.01s × 16kHz

    scores = {}
    for uttid, path in restored_scp.items():
        try:
            wav = load_wav(path, target_sr=16000)
            if len(wav) < N_SAMPLES:
                wav = np.pad(wav, (0, N_SAMPLES - len(wav)))
            wav = wav[:N_SAMPLES].astype(np.float32)
            inp = wav[np.newaxis, :]
            out = sess.run(None, {"input_1": inp})[0][0]
            sig, bak, ovr = out
            _, _, ovr_corr = _dnsmos_polyfit(sig, bak, ovr)
            scores[uttid] = float(np.clip(ovr_corr, 1.0, 5.0))
        except Exception as e:
            logger.warning("DNSMOS failed for %s: %s", uttid, e)

    mean_score = float(np.mean(list(scores.values()))) if scores else float("nan")
    logger.info("  DNSMOS: %.4f", mean_score)
    return {"dnsmos": mean_score, "dnsmos_per_utt": scores}


# ---------------------------------------------------------------------------
# SpkSim  (microsoft/wavlm-base-plus-sv, noisy ↔ restored)
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

    def _embed(wav_np: np.ndarray) -> torch.Tensor:
        inputs = extractor(
            wav_np, sampling_rate=16000, return_tensors="pt", padding=True,
        ).to(device)
        with torch.inference_mode():
            out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1)
        return F.normalize(emb, dim=-1)

    uttids = [k for k in restored_scp if k in noisy_scp]
    logger.info("  %d utterance pairs", len(uttids))

    scores = {}
    for i, uid in enumerate(uttids):
        try:
            wav_noisy    = load_wav(noisy_scp[uid],    16000)
            wav_restored = load_wav(restored_scp[uid], 16000)
            sim = (_embed(wav_noisy) * _embed(wav_restored)).sum().item()
            scores[uid] = sim
        except Exception as e:
            logger.warning("SpkSim failed for %s: %s", uid, e)
        if (i + 1) % 500 == 0:
            logger.info("  SpkSim progress: %d/%d", i + 1, len(uttids))

    mean_sim = float(np.mean(list(scores.values()))) if scores else float("nan")
    logger.info("  SpkSim: %.4f", mean_sim)
    return {"spksim": mean_sim, "spksim_per_utt": scores}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="Evaluate restored speech: WER / NISQA / DNSMOS / SpkSim"
    )
    p.add_argument("--restored_scp", default=None)
    p.add_argument("--restored_dir", default=None)
    p.add_argument("--noisy_scp",    default=None,
                   help="Noisy (input) wav.scp — required for SpkSim")
    p.add_argument("--text",         default=None,
                   help="Kaldi text file — required for WER")
    p.add_argument("--out_dir",      default="exp/scores")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--metrics",      nargs="+",
                   default=["wer", "nisqa", "dnsmos", "spksim"],
                   choices=["wer", "nisqa", "dnsmos", "spksim"])
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
        restored_scp = dir_to_scp(args.restored_dir)
    else:
        raise ValueError("Provide --restored_scp or --restored_dir")
    logger.info("Restored: %d utterances", len(restored_scp))

    noisy_scp = read_scp(args.noisy_scp) if args.noisy_scp else {}
    ref_text  = read_text(args.text)     if args.text     else {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    metrics = args.metrics

    if "wer" in metrics:
        if not ref_text:
            logger.warning("--text not provided; skipping WER")
        else:
            r = compute_wer(restored_scp, ref_text, device)
            summary["wer"] = r["wer"]
            with open(out_dir / "wer_per_utt.json", "w") as f:
                json.dump(r["wer_per_utt"], f, indent=2)
            with open(out_dir / "wer_details.json", "w") as f:
                json.dump(r["wer_details"], f, indent=2, ensure_ascii=False)
            with open(out_dir / "hypothesis", "w") as f:
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

    with open(out_dir / "scores.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 50)
    logger.info("SUMMARY → %s/scores.json", out_dir)
    for k, v in summary.items():
        logger.info("  %-10s : %.4f", k.upper(), v)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()