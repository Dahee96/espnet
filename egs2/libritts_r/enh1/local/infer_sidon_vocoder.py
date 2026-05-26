#!/usr/bin/env python3
"""Inference: ESPnet Feature Predictor + Sidon public vocoder.

Supports:
  - File paths and sox pipe commands in wav.scp
  - Kaldi segments file for segment-level inference
  - Chunk-based inference for long-form audio with overlap-add

Usage:
    # Segment-level (ami_sdm_test)
    python3 local/infer_sidon_vocoder.py \
        --fp_checkpoint exp/speech_cleaner_fp_w2v_bert2/valid.loss.best.pth \
        --fp_config     exp/speech_cleaner_fp_w2v_bert2/config.yaml \
        --sidon_dir /home/samsung/.cache/huggingface/hub/models--sarulab-speech--sidon-v0.1/snapshots/b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52 \
        --wav_scp       data/ami_sdm_test/wav.scp \
        --segments      data/ami_sdm_test/segments \
        --out_dir       exp/restored_sidon_ami_sdm_test \
        --device        cuda

    # Long-form (ami_sdm_longform)
CUDA_VISIBLE_DEVICES=2    python3 local/infer_sidon_vocoder.py \
        --fp_checkpoint exp/speech_cleaner_fp_w2v_bert2/valid.loss.best.pth \
        --fp_config     exp/speech_cleaner_fp_w2v_bert2/config.yaml \
        --sidon_dir /home/samsung/.cache/huggingface/hub/models--sarulab-speech--sidon-v0.1/snapshots/b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52 \
        --wav_scp       data/ami_sdm_longform/wav.scp \
        --out_dir       exp/restored_sidon_ami_sdm_longform \
        --chunk_sec     20.0 \
        --device        cuda

CUDA_VISIBLE_DEVICES=3    python3 local/infer_sidon_vocoder.py \
        --fp_checkpoint exp/speech_cleaner_fp_w2v_bert2/valid.loss.best.pth \
        --fp_config     exp/speech_cleaner_fp_w2v_bert2/config.yaml \
        --sidon_dir /home/samsung/.cache/huggingface/hub/models--sarulab-speech--sidon-v0.1/snapshots/b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52 \
        --wav_scp       data/libritts_test-clean/wav.scp \
        --out_dir       exp/restored_sidon_libritts_test-clean_8s \
        --device        cuda
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fp_checkpoint", required=True)
    p.add_argument("--fp_config",     required=True)
    p.add_argument("--sidon_dir",     required=True)
    p.add_argument("--wav_scp",       required=True)
    p.add_argument("--segments",      default=None,
                   help="Kaldi segments file for segment-level inference.")
    p.add_argument("--out_dir",       default="exp/restored_sidon_test")
    p.add_argument("--max_samples",   type=int,   default=0)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--input_sr",      type=int,   default=16000)
    p.add_argument("--chunk_sec",     type=float, default=20.0)
    p.add_argument("--overlap_sec",   type=float, default=0.5)
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
    segs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                segs.append((parts[0], parts[1], float(parts[2]), float(parts[3])))
    return segs


def load_wav_from_pipe_or_file(wav_cmd: str, target_sr: int = 16000) -> np.ndarray:
    wav_cmd = wav_cmd.strip()
    if wav_cmd.endswith("|"):
        cmd  = wav_cmd[:-1].strip()
        proc = subprocess.run(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"sox pipe failed: {proc.stderr.decode()}\ncmd: {cmd}")
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
    wav_cmd: str, start_sec: float, end_sec: float, target_sr: int = 16000,
) -> np.ndarray:
    wav_cmd = wav_cmd.strip()
    if wav_cmd.endswith("|"):
        base = wav_cmd[:-1].strip()
        cmd  = f"{base} trim {start_sec:.3f} ={end_sec:.3f} |"
    else:
        cmd = (f"sox {wav_cmd} -r {target_sr} -c 1 -t wavpcm - "
               f"trim {start_sec:.3f} ={end_sec:.3f} |")
    return load_wav_from_pipe_or_file(cmd, target_sr)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_processor(model_tag: str = "facebook/w2v-bert-2.0"):
    from transformers import SeamlessM4TFeatureExtractor
    return SeamlessM4TFeatureExtractor.from_pretrained(model_tag)


def load_espnet_fp(config_path: str, ckpt_path: str, device: str):
    import yaml
    from argparse import Namespace

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    args = Namespace(**cfg)
    for k, v in [
        ("ssl_encoder",         "w2v_bert2"),
        ("ssl_encoder_conf",    None),
        ("target_layer",        8),
        ("lora_rank",           64),
        ("lora_alpha",          16),
        ("lora_dropout",        0.1),
        ("input_sr",            16000),
        ("use_flash_attention", False),
        ("use_multilayer_loss", False),
        ("multilayer_mode",     "low"),
    ]:
        if not hasattr(args, k):
            setattr(args, k, v)

    sys.path.insert(0, str(Path(config_path).parent.parent.parent.parent))
    from espnet2.tasks.speech_cleaner import SpeechCleanerFPTask

    model = SpeechCleanerFPTask.build_model(args)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd    = state.get("model", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing keys: %s", missing[:5])
    model = model.eval().to(device)
    logger.info("ESPnet FP loaded: %s", ckpt_path)
    return model


def load_sidon_vocoder(sidon_dir: str, device: str):
    suffix = "cuda" if device.startswith("cuda") else "cpu"
    pt_path = os.path.join(sidon_dir, f"decoder_{suffix}.pt")
    if not os.path.exists(pt_path):
        alt = "cpu" if suffix == "cuda" else "cuda"
        pt_path = os.path.join(sidon_dir, f"decoder_{alt}.pt")
        logger.warning("decoder_%s.pt not found, using %s", suffix, pt_path)
    vocoder = torch.jit.load(pt_path, map_location=device).eval().to(device)
    logger.info("Sidon vocoder loaded: %s", pt_path)
    return vocoder


# ---------------------------------------------------------------------------
# Core inference (single chunk)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _infer_chunk(
    wav_np: np.ndarray, fp_model, vocoder, processor, device: str,
    input_sr: int = 16000,
) -> np.ndarray:
    wav_padded = np.pad(wav_np, (40, 40), mode="constant")
    ssl_inputs = processor(
        [wav_padded], sampling_rate=input_sr,
        return_tensors="pt", padding=True,
    )
    ssl_inputs = {k: v.to(device) for k, v in ssl_inputs.items()}

    pred_feat, _ = fp_model.ssl_encoder(ssl_inputs)   # [1, T_ssl, 1024]
    restored     = vocoder(pred_feat.transpose(1, 2))  # [1, 1, T_wav] or [1, T_wav]
    if restored.dim() == 3:
        restored = restored.squeeze(1)
    return restored.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Long-form overlap-add
# ---------------------------------------------------------------------------

def restore_longform(
    wav_np: np.ndarray, fp_model, vocoder, processor, device: str,
    input_sr: int = 16000, output_sr: int = 48000,
    chunk_sec: float = 20.0, overlap_sec: float = 0.5,
) -> np.ndarray:
    chunk_samples   = int(chunk_sec   * input_sr)
    overlap_samples = int(overlap_sec * input_sr)
    hop_samples     = chunk_samples - overlap_samples
    total_samples   = len(wav_np)
    up_ratio        = output_sr / input_sr

    chunks = []
    pos = 0
    while pos < total_samples:
        end   = min(pos + chunk_samples, total_samples)
        chunk = wav_np[pos:end]
        if len(chunk) >= int(0.1 * input_sr):
            chunks.append((pos, chunk))
        pos += hop_samples

    if len(chunks) == 0:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return _infer_chunk(chunks[0][1], fp_model, vocoder, processor,
                            device, input_sr)

    restored_chunks = []
    for i, (pos, chunk) in enumerate(chunks):
        try:
            out = _infer_chunk(chunk, fp_model, vocoder, processor, device, input_sr)
            restored_chunks.append((pos, out))
        except Exception as e:
            logger.warning("  chunk %d failed: %s", i, e)
            restored_chunks.append((pos, np.zeros(int(len(chunk) * up_ratio),
                                                   dtype=np.float32)))

    output_len  = int(total_samples * up_ratio) + int(overlap_sec * output_sr)
    output      = np.zeros(output_len, dtype=np.float32)
    weight      = np.zeros(output_len, dtype=np.float32)
    overlap_out = int(overlap_sec * output_sr)

    for i, (pos, out) in enumerate(restored_chunks):
        out_start = int(pos * up_ratio)
        out_end   = min(out_start + len(out), output_len)
        out       = out[:out_end - out_start]

        env = np.ones(len(out), dtype=np.float32)
        if i > 0 and overlap_out > 0:
            fade_len = min(overlap_out, len(out))
            env[:fade_len] = np.linspace(0.0, 1.0, fade_len)
        if i < len(restored_chunks) - 1 and overlap_out > 0:
            fade_len = min(overlap_out, len(out))
            env[-fade_len:] = np.minimum(
                env[-fade_len:], np.linspace(1.0, 0.0, fade_len))

        output[out_start:out_end] += out * env
        weight[out_start:out_end] += env

    valid = weight > 1e-6
    output[valid] /= weight[valid]
    return output[:int(total_samples * up_ratio)].astype(np.float32)


def restore_one(
    wav_np: np.ndarray, fp_model, vocoder, processor, device: str,
    input_sr: int = 16000, output_sr: int = 48000,
    chunk_sec: float = 20.0, overlap_sec: float = 0.5,
) -> np.ndarray:
    dur = len(wav_np) / input_sr
    if dur <= chunk_sec:
        return _infer_chunk(wav_np, fp_model, vocoder, processor, device, input_sr)
    else:
        logger.debug("  long-form (%.1fs): chunked inference", dur)
        return restore_longform(
            wav_np, fp_model, vocoder, processor, device,
            input_sr=input_sr, output_sr=output_sr,
            chunk_sec=chunk_sec, overlap_sec=overlap_sec,
        )


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def main():
    args   = get_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    processor = load_processor()
    fp_model  = load_espnet_fp(args.fp_config, args.fp_checkpoint, device)
    vocoder   = load_sidon_vocoder(args.sidon_dir, device)

    wav_map = read_wav_scp(args.wav_scp)

    if args.segments and os.path.exists(args.segments):
        logger.info("Segment mode: %s", args.segments)
        segs = read_segments(args.segments)
        if args.max_samples > 0:
            segs = segs[:args.max_samples]
        entries = segs   # (uttid, recid, start, end)
        mode = "segment"
    else:
        entries = [(recid, recid, None, None) for recid in sorted(wav_map)]
        if args.max_samples > 0:
            entries = entries[:args.max_samples]
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
                wav_np = load_wav_segment(wav_cmd, start, end, args.input_sr)
            else:
                wav_np = load_wav_from_pipe_or_file(wav_cmd, args.input_sr)

            restored = restore_one(
                wav_np, fp_model, vocoder, processor, device,
                input_sr=args.input_sr, output_sr=48000,
                chunk_sec=args.chunk_sec, overlap_sec=args.overlap_sec,
            )

            out_path = wav_dir / f"{uttid}.wav"
            sf.write(str(out_path), restored, 48000)
            scp_lines.append(f"{uttid} {out_path}")

            if (i + 1) % 200 == 0 or i == 0:
                logger.info("  [%d/%d] %s (%.1fs → %.1fs)",
                            i+1, len(entries), uttid,
                            len(wav_np) / args.input_sr,
                            len(restored) / 48000)
        except Exception as e:
            logger.error("FAILED %s: %s", uttid, e)

    scp_path = out_dir / "wav.scp"
    with open(scp_path, "w") as f:
        f.write("\n".join(scp_lines) + "\n")
    logger.info("Done. %d/%d files written.", len(scp_lines), len(entries))


if __name__ == "__main__":
    main()