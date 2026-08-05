#!/usr/bin/env python3
# versa owsm 포함 + multi-backend spksim (ECAPA / RawNet3 / WavLM, windowed)
"""Unified speech restoration evaluation.

Metrics
-------
Non-reference (always available):
  wer          : ASR-based WER.  --asr_model: mms | whisper-large-v3 | whisper-large-v3-turbo | owsm-v3 | owsm-v3.1
  dnsmos       : DNSMOS P.835 OVRL (ONNX)
  nisqa        : NISQA MOS prediction
  spksim       : Speaker cosine similarity (noisy <-> restored).
                 --spksim_backend: wavlm | ecapa | rawnet3
                 Follows Samuele's protocol: embeddings extracted on
                 3s windows with 0.5s stride (configurable via
                 --spksim_win_sec / --spksim_hop_sec), then averaged
                 per utterance, for BOTH the noisy and restored signal,
                 each resampled to 16kHz before embedding extraction.
                 For utterances shorter than the window, this naturally
                 reduces to a single whole-utterance window — so the same
                 logic applies uniformly across LibriTTS / AMI / Fisher.
  utmos        : UTokyo-SaruLab MOS predictor
  squim_noref  : TorchAudio-Squim reference-free (predicts STOI, PESQ, SI-SDR)

Reference-required:
  speechbertscore : SpeechBERTScore (restored ↔ clean reference)
  squim_ref       : TorchAudio-Squim reference-based MOS

Long-form / conversational:
  der          : Diarization Error Rate via pyannote (--mode longform only)

Modes
-----
  standard  : utterance-level (LibriTTS test-clean/other, degrad sets)
  longform  : full-meeting inference + segment-level WER + DER (AMI)

Install notes for spksim backends
----------------------------------
  wavlm   : pip install transformers   (already required elsewhere here)
  ecapa   : pip install speechbrain
  rawnet3 : pip install espnet espnet_model_zoo
            (already present if espnet2 is installed for this repo;
             this is the same model VERSA's speaker.py uses internally)

Usage examples
--------------
# Standard LibriTTS eval (Whisper, default WavLM spksim)
python local/score.py \
    --restored_scp exp/restored_xeus_multi_all/test-clean/wav.scp \
    --noisy_scp    data/libritts_test-clean/wav.scp \
    --text         data/libritts_test-clean/text \
    --out_dir      exp/scores/xeus_multi_all_test-clean \
    --metrics wer dnsmos nisqa spksim utmos squim_noref \
    --asr_model whisper-large-v3-turbo \
    --device cuda

# Same, but spksim with ECAPA-TDNN (Samuele's protocol)
python local/score.py \
    --restored_scp exp/restored_xeus_multi_all/test-clean/wav.scp \
    --noisy_scp    data/libritts_test-clean/wav.scp \
    --text         data/libritts_test-clean/text \
    --out_dir      exp/scores/xeus_multi_all_test-clean_ecapa \
    --metrics spksim \
    --spksim_backend ecapa --spksim_win_sec 3.0 --spksim_hop_sec 1.5 \
    --device cuda

# Same with RawNet3 (matches VERSA's underlying model, but windowed)
python local/score.py \
    --restored_scp exp/restored_xeus_multi_all/test-clean/wav.scp \
    --noisy_scp    data/libritts_test-clean/wav.scp \
    --out_dir      exp/scores/xeus_multi_all_test-clean_rawnet3 \
    --metrics spksim \
    --spksim_backend rawnet3 \
    --device cuda

# Standard LibriTTS eval (OWSM v3.1 — same ESPnet ecosystem as training)
python local/score.py \
    --restored_scp exp/restored_xeus_multi_all/test-clean/wav.scp \
    --noisy_scp    data/libritts_test-clean/wav.scp \
    --text         data/libritts_test-clean/text \
    --out_dir      exp/scores/xeus_multi_all_test-clean_owsm \
    --metrics wer dnsmos nisqa spksim \
    --asr_model owsm-v3.1 \
    --device cuda

# Degraded set (with reference)
python local/score.py \
    --restored_scp exp/restored_xeus/test-other-degrad/wav.scp \
    --noisy_scp    data/libritts_test-other-degrad/noisy/wav.scp \
    --ref_scp      data/libritts_test-other/wav.scp \
    --text         data/libritts_test-other-degrad/text \
    --out_dir      exp/scores/xeus_test-other-degrad \
    --metrics wer dnsmos nisqa spksim utmos squim_noref speechbertscore squim_ref \
    --device cuda

# AMI long-form
python local/score.py \
    --restored_scp exp/restored_xeus/ami_sdm_longform/wav.scp \
    --noisy_scp    data/ami_sdm_longform/wav.scp \
    --ref_scp      data/ami_ihm_mix/wav.scp \
    --text         data/ami_ihm_mix/text \
    --segments     data/ami_ihm_mix/segments \
    --out_dir      exp/scores/xeus_ami_sdm_longform \
    --metrics wer dnsmos nisqa spksim utmos squim_noref der \
    --spksim_backend ecapa \
    --mode longform \
    --device cuda
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Make sure spksim_models.py (placed alongside this script in local/) is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spksim_models import get_spk_embedder  # noqa: E402

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

DNSMOS_ONNX_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/"
    "master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)


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


def read_segments(path: str) -> Dict[str, tuple]:
    """Returns {uttid: (rec_id, start_sec, end_sec)}."""
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                d[parts[0]] = (parts[1], float(parts[2]), float(parts[3]))
    return d


def load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load from file path or sox pipe command ending with '|'."""
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

    import soundfile as sf
    import librosa
    wav, sr = sf.read(path, always_2d=True)
    wav = wav.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


def load_wav_torch(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load as torch.Tensor, supporting sox pipe commands via load_wav()."""
    path_stripped = path.strip()
    if path_stripped.endswith("|"):
        return torch.from_numpy(load_wav(path_stripped, target_sr=target_sr))

    import torchaudio
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


# =============================================================================
# Text normalization
# =============================================================================

def normalize_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"['\-,\.!\?;:\(\)\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def edit_distance(hyp: List[str], ref: List[str]) -> int:
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


# =============================================================================
# WER
# =============================================================================

def compute_wer(
    restored_scp: Dict[str, str],
    ref_text: Dict[str, str],
    device: str,
    asr_model: str = "owsm-v3.1",
    batch_size: int = 16,
    segments: Optional[Dict[str, tuple]] = None,
) -> Dict:
    """Compute WER using the specified ASR model.

    asr_model options:
      mms                    : facebook/mms-1b-all (multilingual, 1B)
      whisper-large-v3       : openai/whisper-large-v3 (best accuracy)
      whisper-large-v3-turbo : openai/whisper-large-v3-turbo (fast, good accuracy)
      owsm-v3                : espnet/owsm_v3 (889M, 180k hrs, ESPnet)
      owsm-v3.1              : espnet/owsm_v3.1_ebf (better/faster, E-Branchformer)

    All models expect 16kHz mono input — load_wav() already handles resampling.
    """
    logger.info("Computing WER with %s ...", asr_model)

    if asr_model == "mms":
        return _wer_mms(restored_scp, ref_text, device, batch_size)
    elif asr_model in ("whisper-large-v3", "whisper-large-v3-turbo"):
        return _wer_whisper(restored_scp, ref_text, device, asr_model,
                            batch_size, segments)
    elif asr_model in ("owsm-v3", "owsm-v3.1"):
        return _wer_owsm(restored_scp, ref_text, device, asr_model)
    else:
        raise ValueError(f"Unknown asr_model: {asr_model}. "
                         f"Choose: mms, whisper-large-v3, whisper-large-v3-turbo, "
                         f"owsm-v3, owsm-v3.1")


def _wer_mms(restored_scp, ref_text, device, batch_size=16):
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    model     = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(device).eval()
    processor.tokenizer.set_target_lang("eng")
    model.load_adapter("eng")

    uttids = [k for k in restored_scp if k in ref_text]
    total_errors = total_words = 0
    results = {}

    for start in range(0, len(uttids), batch_size):
        batch_ids = uttids[start:start + batch_size]
        wavs      = [load_wav(restored_scp[u], 16000) for u in batch_ids]
        inputs    = processor(wavs, sampling_rate=16000,
                              return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        hypotheses = processor.batch_decode(torch.argmax(logits, dim=-1))
        for uid, hyp in zip(batch_ids, hypotheses):
            hyp_w = normalize_text(hyp).split()
            ref_w = normalize_text(ref_text[uid]).split()
            err   = edit_distance(hyp_w, ref_w)
            total_errors += err
            total_words  += max(len(ref_w), 1)
            results[uid]  = {"wer": err / max(len(ref_w), 1),
                              "hypothesis": " ".join(hyp_w),
                              "reference":  " ".join(ref_w)}

    wer = total_errors / max(total_words, 1)
    logger.info("  WER (MMS): %.4f", wer)
    return {"wer": wer, "wer_per_utt": {u: v["wer"] for u, v in results.items()},
            "wer_details": results}


def _wer_whisper(restored_scp, ref_text, device, model_name, batch_size=8,
                 segments=None):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    model_id_map = {
        "whisper-large-v3":       "openai/whisper-large-v3",
        "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
    }
    model_id = model_id_map[model_name]
    logger.info("  Loading %s ...", model_id)

    dtype    = torch.float16 if "cuda" in device else torch.float32
    model    = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype, device=device,
        chunk_length_s=30,
    )

    uttids = [k for k in restored_scp if k in ref_text]
    total_errors = total_words = 0
    results = {}

    for uid in uttids:
        try:
            out = pipe(restored_scp[uid])
            hyp = out["text"]
        except Exception as e:
            logger.warning("Whisper failed for %s: %s", uid, e)
            hyp = ""
        hyp_w = normalize_text(hyp).split()
        ref_w = normalize_text(ref_text[uid]).split()
        err   = edit_distance(hyp_w, ref_w)
        total_errors += err
        total_words  += max(len(ref_w), 1)
        results[uid]  = {"wer": err / max(len(ref_w), 1),
                         "hypothesis": " ".join(hyp_w),
                         "reference":  " ".join(ref_w)}

    wer = total_errors / max(total_words, 1)
    logger.info("  WER (%s): %.4f", model_name, wer)
    return {"wer": wer, "wer_per_utt": {u: v["wer"] for u, v in results.items()},
            "wer_details": results}


def _wer_owsm(restored_scp, ref_text, device, model_name="owsm-v3.1"):
    """WER using OWSM, following versa/corpus_metrics/owsm_wer.py exactly."""
    model_id_map = {
        "owsm-v3":   "espnet/owsm_v3",
        "owsm-v3.1": "espnet/owsm_v3.1_ebf",
    }
    hf_model_id = model_id_map[model_name]
    logger.info("  Loading OWSM: %s ...", hf_model_id)

    try:
        from espnet2.bin.s2t_inference import Speech2Text
        from espnet2.text.cleaner import TextCleaner
    except ImportError:
        logger.error(
            "ESPnet2 not installed. Run in the espnet conda env or: pip install espnet"
        )
        return {"wer": float("nan"), "wer_per_utt": {}, "wer_details": {}}

    import librosa

    s2t = Speech2Text.from_pretrained(
        model_tag=hf_model_id,
        device=device,
        task_sym="<asr>",
        beam_size=5,
        predict_time=False,
    )
    cleaner = TextCleaner("whisper_basic")

    TARGET_FS  = 16000
    CHUNK_SIZE = 30  # seconds

    def _owsm_predict(wav):
        assert len(wav.shape) == 1
        s2t.beam_search.beam_size = 5
        lang_sym = "<eng>"
        task_sym = "<asr>"
        if len(wav) > CHUNK_SIZE * TARGET_FS:
            try:
                s2t.maxlenratio = -300
                utts = s2t.decode_long(
                    wav,
                    condition_on_prev_text=False,
                    init_text="",
                    end_time_threshold="<29.00>",
                    lang_sym=lang_sym,
                    task_sym=task_sym,
                )
                return " ".join(res for _, _, res in utts)
            except Exception as e:
                logger.warning("    decode_long failed (%s), falling back to first 30s", e)
        s2t.maxlenratio = -min(300, int((len(wav) / TARGET_FS) * 10))
        wav_30 = librosa.util.fix_length(wav, size=TARGET_FS * CHUNK_SIZE)
        return s2t(wav_30, "", lang_sym=lang_sym, task_sym=task_sym)[0][-2]

    uttids = [k for k in restored_scp if k in ref_text]
    logger.info("  %d utterances to decode", len(uttids))

    total_errors = total_words = 0
    results = {}

    for i, uid in enumerate(uttids):
        try:
            wav = load_wav(restored_scp[uid], target_sr=TARGET_FS)
            with torch.no_grad():
                hyp_raw = _owsm_predict(wav)
        except Exception as e:
            logger.warning("  OWSM failed for %s: %s", uid, e)
            hyp_raw = ""

        hyp_clean = cleaner(hyp_raw).strip()
        ref_clean  = cleaner(ref_text[uid]).strip()

        hyp_w = hyp_clean.split()
        ref_w  = ref_clean.split()
        err    = edit_distance(hyp_w, ref_w)
        total_errors += err
        total_words  += max(len(ref_w), 1)
        results[uid]  = {
            "wer":        err / max(len(ref_w), 1),
            "hypothesis": hyp_clean,
            "reference":  ref_clean,
        }

        if (i + 1) % 200 == 0 or (i + 1) == len(uttids):
            logger.info("  OWSM WER progress: %d/%d  running=%.4f",
                        i + 1, len(uttids), total_errors / max(total_words, 1))

    wer = total_errors / max(total_words, 1)
    logger.info("  WER (%s): %.4f", model_name, wer)
    return {
        "wer":         wer,
        "wer_per_utt": {u: v["wer"] for u, v in results.items()},
        "wer_details": results,
    }


# =============================================================================
# DNSMOS
# =============================================================================

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


def compute_dnsmos(restored_scp, cache_dir=".dnsmos_cache"):
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
            wav = load_wav(path, 16000)
            if len(wav) < N_SAMPLES:
                wav = np.pad(wav, (0, N_SAMPLES - len(wav)))
            wav = wav[:N_SAMPLES].astype(np.float32)
            out = sess.run(None, {"input_1": wav[np.newaxis, :]})[0][0]
            _, _, ovr_corr = _dnsmos_polyfit(*out)
            scores[uttid] = float(np.clip(ovr_corr, 1.0, 5.0))
        except Exception as e:
            logger.warning("DNSMOS failed for %s: %s", uttid, e)

    mean = float(np.mean(list(scores.values()))) if scores else float("nan")
    logger.info("  DNSMOS: %.4f", mean)
    return {"dnsmos": mean, "dnsmos_per_utt": scores}


# =============================================================================
# NISQA
# =============================================================================

def compute_nisqa(restored_scp, device, nisqa_model_path="nisqa_pretrained_model"):
    logger.info("Computing NISQA ...")
    try:
        from nisqa.NISQA_model import nisqaModel
    except ImportError:
        logger.error("nisqa not installed: pip install nisqa")
        return {"nisqa": float("nan"), "nisqa_per_utt": {}}

    if not os.path.exists(nisqa_model_path):
        logger.error("NISQA model not found at %s", nisqa_model_path)
        return {"nisqa": float("nan"), "nisqa_per_utt": {}}

    import soundfile as sf
    tmp_dir = tempfile.mkdtemp()
    try:
        for uid, path in restored_scp.items():
            wav = load_wav(path, 16000)
            sf.write(os.path.join(tmp_dir, f"{uid}.wav"), wav, 16000)
        nisqa = nisqaModel({
            "mode": "predict_dir", "pretrained_model": nisqa_model_path,
            "deg": None, "data_dir": tmp_dir, "output_dir": None,
            "ms_channel": None, "device": device,
        })
        df     = nisqa.predict()
        scores = dict(zip(df["deg"].apply(lambda p: Path(p).stem),
                          df["mos_pred"].astype(float)))
        mean   = float(np.mean(list(scores.values()))) if scores else float("nan")
    except Exception as e:
        logger.error("NISQA failed: %s", e)
        scores, mean = {}, float("nan")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("  NISQA: %.4f", mean)
    return {"nisqa": mean, "nisqa_per_utt": scores}


# =============================================================================
# SpkSim — now multi-backend (wavlm / ecapa / rawnet3), windowed averaging
# =============================================================================

def compute_spksim(
    restored_scp: Dict[str, str],
    noisy_scp: Dict[str, str],
    device: str,
    backend: str = "wavlm",
    win_sec: float = 3.0,
    hop_sec: float = 1.5,
    segments: Optional[Dict[str, tuple]] = None,
) -> Dict:
    """Speaker cosine similarity between noisy (input) and restored signals.

    Follows Samuele's actual protocol (confirmed from his reference code):
      - 3s windows, 1.5s hop (not 0.5s)
      - segments shorter than one window are zero-padded to exactly one
        window, not used at their original length
      - the final window is shifted to end exactly at the segment boundary
      - per-window embeddings are mean-pooled THEN L2-normalized once
      - everything resampled to 16kHz before embedding extraction

    If `segments` is given ({uttid: (rec_id, start_sec, end_sec)}), audio is
    loaded via direct random-access read of [start, end] at the recording's
    native sample rate, resampled to 16kHz AFTER slicing — matching
    Samuele's load_segment_16k() ordering exactly (read-then-resample,
    rather than resample-the-whole-file-then-slice). This matters for
    AMI/Fisher long-form evaluation, where uttids in restored_scp/noisy_scp
    may be full-recording paths rather than per-segment files; in that case
    rec_id is looked up in restored_scp/noisy_scp and the [start, end]
    window is read directly from that recording.

    If `segments` is None (e.g. LibriTTS-style utterance-level data where
    restored_scp/noisy_scp keys are already per-utterance files), falls back
    to the previous whole-file load_wav() path — equivalent to passing
    start=0, end=<full file> through the same windowing logic.

    backend: "wavlm" (microsoft/wavlm-base-plus-sv, windowed averaging),
             "ecapa" (speechbrain/spkrec-ecapa-voxceleb, windowed averaging),
             "rawnet3" (espnet/voxcelebs12_rawnet3, DEFAULT — VERSA protocol:
             whole-segment embedding via RawNet3's own attentive pooling,
             NO windowing; win_sec/hop_sec are accepted but ignored for
             this backend, kept only so the same call signature works for
             all three backends).
    """
    if backend == "rawnet3":
        logger.info("Computing SpkSim (noisy <-> restored) backend=rawnet3 "
                    "(VERSA protocol: whole-segment, no windowing; "
                    "win_sec/hop_sec ignored) mode=%s ...",
                    "segment-direct" if segments else "whole-file")
    else:
        logger.info("Computing SpkSim (noisy <-> restored) backend=%s win=%.1fs hop=%.1fs "
                    "mode=%s ...", backend, win_sec, hop_sec,
                    "segment-direct" if segments else "whole-file")
    embedder = get_spk_embedder(backend, device)

    from spksim_models import load_segment_16k

    if segments:
        uttids = [u for u in segments
                  if (u in restored_scp or segments[u][0] in restored_scp)
                  and (u in noisy_scp or segments[u][0] in noisy_scp)]
        logger.info("  %d segments (direct random-access load)", len(uttids))

        scores = {}
        for i, uid in enumerate(uttids):
            try:
                rec_id, start, end = segments[uid]

                # Restored: segment-level file if inference was run per-segment,
                # otherwise direct random-access read from the full recording.
                if uid in restored_scp:
                    wav_r = load_wav(restored_scp[uid], 16000)
                else:
                    wav_r = load_segment_16k(restored_scp[rec_id], start, end)

                # Noisy/input: always read directly from the full recording at
                # [start, end] (noisy_scp is recording-level by construction
                # for AMI/Fisher).
                if uid in noisy_scp:
                    wav_n = load_wav(noisy_scp[uid], 16000)
                else:
                    wav_n = load_segment_16k(noisy_scp[rec_id], start, end)

                if wav_n is None or wav_r is None:
                    continue
                e_n = embedder.embed(wav_n, win_sec=win_sec, hop_sec=hop_sec)
                e_r = embedder.embed(wav_r, win_sec=win_sec, hop_sec=hop_sec)
                sim = (e_n * e_r).sum().item()
                scores[uid] = sim
            except Exception as e:
                logger.warning("SpkSim failed for %s: %s", uid, e)
            if (i + 1) % 500 == 0:
                logger.info("  SpkSim: %d/%d", i + 1, len(uttids))
    else:
        uttids = [k for k in restored_scp if k in noisy_scp]
        logger.info("  %d utterance pairs (whole-file load)", len(uttids))

        scores = {}
        for i, uid in enumerate(uttids):
            try:
                wav_n = load_wav(noisy_scp[uid],    16000)
                wav_r = load_wav(restored_scp[uid], 16000)
                e_n = embedder.embed(wav_n, win_sec=win_sec, hop_sec=hop_sec)
                e_r = embedder.embed(wav_r, win_sec=win_sec, hop_sec=hop_sec)
                sim = (e_n * e_r).sum().item()
                scores[uid] = sim
            except Exception as e:
                logger.warning("SpkSim failed for %s: %s", uid, e)
            if (i + 1) % 500 == 0:
                logger.info("  SpkSim: %d/%d", i + 1, len(uttids))

    mean = float(np.mean(list(scores.values()))) if scores else float("nan")
    logger.info("  SpkSim (%s): %.4f", backend, mean)
    return {"spksim": mean, "spksim_per_utt": scores}


# =============================================================================
# UTMOS
# =============================================================================

def compute_utmos(restored_scp, device):
    """UTokyo-SaruLab MOS predictor (reference-free)."""
    logger.info("Computing UTMOS ...")
    try:
        import utmos
        predictor = utmos.Score(device=device)
    except ImportError:
        try:
            predictor = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                trust_repo=True,
            ).to(device).eval()
        except Exception as e:
            logger.error("UTMOS not available (%s). pip install utmos", e)
            return {"utmos": float("nan"), "utmos_per_utt": {}}

    scores = {}
    for uid, path in restored_scp.items():
        try:
            wav = load_wav_torch(path, 16000).unsqueeze(0).to(device)
            with torch.inference_mode():
                score = predictor(wav, 16000)
                scores[uid] = float(score.item() if hasattr(score, "item") else score)
        except Exception as e:
            logger.warning("UTMOS failed for %s: %s", uid, e)

    mean = float(np.mean(list(scores.values()))) if scores else float("nan")
    logger.info("  UTMOS: %.4f", mean)
    return {"utmos": mean, "utmos_per_utt": scores}


# =============================================================================
# TorchAudio-Squim
# =============================================================================

def compute_squim_noref(restored_scp, device):
    """TorchAudio-Squim reference-free: predicts STOI, PESQ, SI-SDR."""
    logger.info("Computing TorchAudio-Squim (reference-free) ...")
    try:
        from torchaudio.pipelines import SQUIM_OBJECTIVE
        model = SQUIM_OBJECTIVE.get_model().to(device).eval()
    except Exception as e:
        logger.error("TorchAudio-Squim not available: %s", e)
        return {"squim_stoi": float("nan"), "squim_pesq": float("nan"),
                "squim_sisdr": float("nan"), "squim_per_utt": {}}

    stoi_all, pesq_all, sisdr_all = [], [], []
    per_utt = {}

    for uid, path in restored_scp.items():
        try:
            wav = load_wav_torch(path, 16000).unsqueeze(0).to(device)
            with torch.inference_mode():
                stoi, pesq, sisdr = model(wav)
            per_utt[uid] = {
                "stoi":  float(stoi.item()),
                "pesq":  float(pesq.item()),
                "sisdr": float(sisdr.item()),
            }
            stoi_all.append(per_utt[uid]["stoi"])
            pesq_all.append(per_utt[uid]["pesq"])
            sisdr_all.append(per_utt[uid]["sisdr"])
        except Exception as e:
            logger.warning("Squim(noref) failed for %s: %s", uid, e)

    mean_stoi  = float(np.mean(stoi_all))  if stoi_all  else float("nan")
    mean_pesq  = float(np.mean(pesq_all))  if pesq_all  else float("nan")
    mean_sisdr = float(np.mean(sisdr_all)) if sisdr_all else float("nan")
    logger.info("  Squim(noref) STOI=%.4f PESQ=%.4f SI-SDR=%.2f dB",
                mean_stoi, mean_pesq, mean_sisdr)
    return {
        "squim_stoi":  mean_stoi,
        "squim_pesq":  mean_pesq,
        "squim_sisdr": mean_sisdr,
        "squim_noref_per_utt": per_utt,
    }


def compute_squim_ref(restored_scp, ref_scp, device):
    """TorchAudio-Squim reference-based MOS."""
    logger.info("Computing TorchAudio-Squim (reference-based) ...")
    try:
        from torchaudio.pipelines import SQUIM_SUBJECTIVE
        model = SQUIM_SUBJECTIVE.get_model().to(device).eval()
    except Exception as e:
        logger.error("TorchAudio-Squim (subjective) not available: %s", e)
        return {"squim_mos": float("nan"), "squim_ref_per_utt": {}}

    scores = {}
    uttids = [k for k in restored_scp if k in ref_scp]

    for uid in uttids:
        try:
            wav_r = load_wav_torch(restored_scp[uid], 16000).unsqueeze(0).to(device)
            wav_c = load_wav_torch(ref_scp[uid],      16000).unsqueeze(0).to(device)
            T = min(wav_r.shape[-1], wav_c.shape[-1])
            wav_r, wav_c = wav_r[..., :T], wav_c[..., :T]
            with torch.inference_mode():
                mos = model(wav_r, wav_c)
            scores[uid] = float(mos.item())
        except Exception as e:
            logger.warning("Squim(ref) failed for %s: %s", uid, e)

    mean = float(np.mean(list(scores.values()))) if scores else float("nan")
    logger.info("  Squim(ref) MOS: %.4f", mean)
    return {"squim_mos": mean, "squim_ref_per_utt": scores}


# =============================================================================
# SpeechBERTScore
# =============================================================================

def compute_speechbertscore(restored_scp, ref_scp, device, layer=9):
    """SpeechBERTScore: WavLM-Large frame-level cosine similarity."""
    logger.info("Computing SpeechBERTScore (WavLM-Large layer %d) ...", layer)
    try:
        from transformers import AutoFeatureExtractor, WavLMModel
        import torch.nn.functional as F

        model_id  = "microsoft/wavlm-large"
        extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model     = WavLMModel.from_pretrained(model_id).to(device).eval()
    except Exception as e:
        logger.error("SpeechBERTScore: failed to load WavLM-Large: %s", e)
        return {"speechbertscore": float("nan"), "speechbertscore_per_utt": {}}

    import torch.nn.functional as F
    uttids = [k for k in restored_scp if k in ref_scp]
    scores = {}

    for uid in uttids:
        try:
            wav_r = load_wav(restored_scp[uid], 16000)
            wav_c = load_wav(ref_scp[uid],      16000)
            inp_r = extractor(wav_r, sampling_rate=16000,
                              return_tensors="pt").to(device)
            inp_c = extractor(wav_c, sampling_rate=16000,
                              return_tensors="pt").to(device)
            with torch.inference_mode():
                out_r = model(**inp_r, output_hidden_states=True)
                out_c = model(**inp_c, output_hidden_states=True)
            h_r = F.normalize(out_r.hidden_states[layer], dim=-1)
            h_c = F.normalize(out_c.hidden_states[layer], dim=-1)
            T = min(h_r.shape[1], h_c.shape[1])
            sim = (h_r[0, :T] * h_c[0, :T]).sum(-1).mean()
            scores[uid] = float(sim.item())
        except Exception as e:
            logger.warning("SpeechBERTScore failed for %s: %s", uid, e)

    mean = float(np.mean(list(scores.values()))) if scores else float("nan")
    logger.info("  SpeechBERTScore: %.4f", mean)
    return {"speechbertscore": mean, "speechbertscore_per_utt": scores}


# =============================================================================
# DER (Diarization Error Rate) — long-form only
# =============================================================================

def compute_der(restored_scp, ref_scp, device):
    """Diarization Error Rate using pyannote.audio."""
    logger.info("Computing DER (pyannote) ...")
    try:
        from pyannote.audio import Pipeline
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        logger.error("pyannote.audio not installed: pip install pyannote.audio")
        return {"der": float("nan"), "der_per_utt": {}}

    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        pipeline = pipeline.to(torch.device(device))
    except Exception as e:
        logger.error("pyannote pipeline failed to load: %s", e)
        return {"der": float("nan"), "der_per_utt": {}}

    metric  = DiarizationErrorRate()
    scores  = {}
    uttids  = [k for k in restored_scp if k in ref_scp]

    for uid in uttids:
        try:
            import torchaudio
            wav, sr = torchaudio.load(restored_scp[uid])
            hypothesis = pipeline({"waveform": wav.to(device), "sample_rate": sr})
            wav_ref, sr_ref = torchaudio.load(ref_scp[uid])
            reference = pipeline({"waveform": wav_ref.to(device),
                                   "sample_rate": sr_ref})
            der_val = float(metric(reference, hypothesis))
            scores[uid] = der_val
        except Exception as e:
            logger.warning("DER failed for %s: %s", uid, e)

    mean = float(np.mean(list(scores.values()))) if scores else float("nan")
    logger.info("  DER: %.4f", mean)
    return {"der": mean, "der_per_utt": scores}


# =============================================================================
# Long-form segment-level WER helper
# =============================================================================

def compute_wer_longform(
    restored_scp: Dict[str, str],
    ref_text: Dict[str, str],
    segments: Dict[str, tuple],
    device: str,
    asr_model: str = "owsm-v3.1",
) -> Dict:
    """WER on long-form recordings by slicing restored audio at segment boundaries."""
    import soundfile as sf
    from collections import defaultdict
    logger.info("Computing long-form WER (segments) with %s ...", asr_model)

    rec_to_segs = defaultdict(list)
    for uttid, (rec_id, start, end) in segments.items():
        if uttid in ref_text:
            rec_to_segs[rec_id].append((uttid, start, end))

    seg_scp = {}
    tmp_dir = tempfile.mkdtemp()
    try:
        for rec_id, segs in rec_to_segs.items():
            if rec_id not in restored_scp:
                continue
            try:
                wav = load_wav(restored_scp[rec_id], 16000)
            except Exception as e:
                logger.warning("Cannot load %s: %s", rec_id, e)
                continue
            for uttid, start, end in segs:
                s = int(start * 16000)
                e = int(end   * 16000)
                seg_wav = wav[s:e]
                seg_path = os.path.join(tmp_dir, f"{uttid}.wav")
                sf.write(seg_path, seg_wav, 16000)
                seg_scp[uttid] = seg_path

        if not seg_scp:
            return {"wer": float("nan"), "wer_per_utt": {}}

        result = compute_wer(seg_scp, ref_text, device, asr_model)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


# =============================================================================
# Main
# =============================================================================

def get_args():
    p = argparse.ArgumentParser(
        description="Unified speech restoration evaluation"
    )
    p.add_argument("--restored_scp", default=None)
    p.add_argument("--restored_dir", default=None)
    p.add_argument("--noisy_scp",    default=None,
                   help="Required for SpkSim")
    p.add_argument("--ref_scp",      default=None,
                   help="Clean reference. Required for SpeechBERTScore, Squim-ref, DER")
    p.add_argument("--text",         default=None,
                   help="Kaldi text file. Required for WER")
    p.add_argument("--segments",     default=None,
                   help="Kaldi segments file (longform mode)")
    p.add_argument("--out_dir",      default="exp/scores")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--mode",         default="standard",
                   choices=["standard", "longform"],
                   help="standard: utterance-level. longform: full-meeting + DER")
    p.add_argument("--metrics", nargs="+",
                   default=["wer", "dnsmos", "nisqa", "spksim", "utmos", "squim_noref"],
                   choices=["wer", "dnsmos", "nisqa", "spksim", "utmos",
                            "squim_noref", "squim_ref", "speechbertscore", "der"])
    p.add_argument("--asr_model",    default="owsm-v3.1",
                   choices=["mms", "whisper-large-v3", "whisper-large-v3-turbo",
                            "owsm-v3", "owsm-v3.1"],
                   help="ASR model for WER. All expect 16kHz mono input.")
    p.add_argument("--spksim_backend", default="rawnet3",
                   choices=["wavlm", "ecapa", "rawnet3"],
                   help="Speaker embedding model for SpkSim. "
                        "wavlm=microsoft/wavlm-base-plus-sv (prior default), "
                        "ecapa=speechbrain/spkrec-ecapa-voxceleb, "
                        "rawnet3=espnet/voxcelebs12_rawnet3 (matches VERSA's "
                        "internal model).")
    p.add_argument("--spksim_win_sec", type=float, default=3.0,
                   help="Sliding window length (s) for SpkSim embedding "
                        "averaging (Samuele's protocol: 3.0s).")
    p.add_argument("--spksim_hop_sec", type=float, default=1.5,
                   help="Sliding window stride (s) for SpkSim embedding "
                        "averaging (Samuele's protocol: 0.5s).")
    p.add_argument("--nisqa_model",  default="nisqa_pretrained_model")
    p.add_argument("--dnsmos_cache", default=".dnsmos_cache")
    return p.parse_args()


def main():
    args   = get_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s | Mode: %s | ASR: %s | SpkSim backend: %s",
                device, args.mode, args.asr_model, args.spksim_backend)

    if args.restored_scp:
        restored_scp = read_scp(args.restored_scp)
    elif args.restored_dir:
        restored_scp = {
            Path(p).stem: str(p)
            for p in sorted(Path(args.restored_dir).glob("**/*.wav"))
        }
    else:
        raise ValueError("Provide --restored_scp or --restored_dir")
    logger.info("Restored: %d utterances", len(restored_scp))

    noisy_scp = read_scp(args.noisy_scp) if args.noisy_scp else {}
    ref_scp   = read_scp(args.ref_scp)   if args.ref_scp   else {}
    ref_text  = read_text(args.text)     if args.text      else {}
    segments  = read_segments(args.segments) if args.segments else {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    metrics = args.metrics

    # ── WER ──────────────────────────────────────────────────────────────────
    if "wer" in metrics:
        if not ref_text:
            logger.warning("--text not provided; skipping WER")
        else:
            if args.mode == "longform" and segments:
                r = compute_wer_longform(
                    restored_scp, ref_text, segments, device, args.asr_model
                )
            else:
                r = compute_wer(restored_scp, ref_text, device, args.asr_model)
            summary["wer"] = r["wer"]
            with open(out_dir / "wer_per_utt.json", "w") as f:
                json.dump(r.get("wer_per_utt", {}), f, indent=2)
            if "wer_details" in r:
                with open(out_dir / "wer_details.json", "w") as f:
                    json.dump(r["wer_details"], f, indent=2, ensure_ascii=False)

    # ── DNSMOS ───────────────────────────────────────────────────────────────
    if "dnsmos" in metrics:
        r = compute_dnsmos(restored_scp, cache_dir=args.dnsmos_cache)
        summary["dnsmos"] = r["dnsmos"]
        with open(out_dir / "dnsmos_per_utt.json", "w") as f:
            json.dump(r["dnsmos_per_utt"], f, indent=2)

    # ── NISQA ────────────────────────────────────────────────────────────────
    if "nisqa" in metrics:
        r = compute_nisqa(restored_scp, device, args.nisqa_model)
        summary["nisqa"] = r["nisqa"]
        with open(out_dir / "nisqa_per_utt.json", "w") as f:
            json.dump(r["nisqa_per_utt"], f, indent=2)

    # ── SpkSim ───────────────────────────────────────────────────────────────
    if "spksim" in metrics:
        if not noisy_scp:
            logger.warning("--noisy_scp not provided; skipping SpkSim")
        else:
            r = compute_spksim(
                restored_scp, noisy_scp, device,
                backend=args.spksim_backend,
                win_sec=args.spksim_win_sec,
                hop_sec=args.spksim_hop_sec,
            )
            summary["spksim"] = r["spksim"]
            with open(out_dir / "spksim_per_utt.json", "w") as f:
                json.dump(r["spksim_per_utt"], f, indent=2)

    # ── UTMOS ────────────────────────────────────────────────────────────────
    if "utmos" in metrics:
        r = compute_utmos(restored_scp, device)
        summary["utmos"] = r["utmos"]
        with open(out_dir / "utmos_per_utt.json", "w") as f:
            json.dump(r["utmos_per_utt"], f, indent=2)

    # ── Squim (reference-free) ───────────────────────────────────────────────
    if "squim_noref" in metrics:
        r = compute_squim_noref(restored_scp, device)
        summary.update({k: v for k, v in r.items()
                        if not k.endswith("_per_utt")})
        with open(out_dir / "squim_noref_per_utt.json", "w") as f:
            json.dump(r.get("squim_noref_per_utt", {}), f, indent=2)

    # ── Squim (reference-based) ──────────────────────────────────────────────
    if "squim_ref" in metrics:
        if not ref_scp:
            logger.warning("--ref_scp not provided; skipping squim_ref")
        else:
            r = compute_squim_ref(restored_scp, ref_scp, device)
            summary["squim_mos"] = r["squim_mos"]
            with open(out_dir / "squim_ref_per_utt.json", "w") as f:
                json.dump(r.get("squim_ref_per_utt", {}), f, indent=2)

    # ── SpeechBERTScore ──────────────────────────────────────────────────────
    if "speechbertscore" in metrics:
        if not ref_scp:
            logger.warning("--ref_scp not provided; skipping SpeechBERTScore")
        else:
            r = compute_speechbertscore(restored_scp, ref_scp, device)
            summary["speechbertscore"] = r["speechbertscore"]
            with open(out_dir / "speechbertscore_per_utt.json", "w") as f:
                json.dump(r.get("speechbertscore_per_utt", {}), f, indent=2)

    # ── DER (long-form) ──────────────────────────────────────────────────────
    if "der" in metrics:
        if args.mode != "longform":
            logger.warning("DER is only computed in --mode longform; skipping")
        elif not ref_scp:
            logger.warning("--ref_scp not provided; skipping DER")
        else:
            r = compute_der(restored_scp, ref_scp, device)
            summary["der"] = r["der"]
            with open(out_dir / "der_per_utt.json", "w") as f:
                json.dump(r.get("der_per_utt", {}), f, indent=2)

    # ── Save summary ─────────────────────────────────────────────────────────
    with open(out_dir / "scores.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("SUMMARY  →  %s/scores.json", out_dir)
    for k, v in summary.items():
        unit = " dB" if "sisdr" in k or k == "squim_sisdr" else ""
        logger.info("  %-22s : %.4f%s", k.upper(), v, unit)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()