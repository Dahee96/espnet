#!/usr/bin/env python3
"""Speech Cleaner inference: Stage 1 FP + Stage 2/3 Vocoder.

Supports:
  - XEUS and w2v-BERT2/WavLM ssl encoders
  - Long-form chunk-based inference with overlap-add (for AMI/Fisher)
  - Kaldi segments file for segment-level inference (AMI IHM)
  - Full-recording long-form inference (Fisher, AMI SDM longform)

Usage
-----
# LibriTTS utterance-level (XEUS single)
python local/enh_inference_speech_cleaner_longform.py \
    --fp_train_config  exp/speech_cleaner_fp_xeus_single/config.yaml \
    --fp_model_file    exp/speech_cleaner_fp_xeus_single/valid.loss.best.pth \
    --voc_train_config exp/speech_cleaner_voc_finetune_xeus_single/config.yaml \
    --voc_model_file   exp/speech_cleaner_voc_finetune_xeus_single/valid.loss_G.best.pth \
    --wav_scp          data/libritts_test-clean_16k/wav.scp \
    --output_dir       exp/restored_xeus_single/test-clean \
    --batch_size 4 --device cuda --dtype bfloat16

# AMI IHM longform — segments mode (each segment restored individually)
python local/enh_inference_speech_cleaner_longform.py \
    --fp_train_config  exp/speech_cleaner_fp_xeus_multi_all/config.yaml \
    --fp_model_file    exp/speech_cleaner_fp_xeus_multi_all/valid.loss.best.pth \
    --voc_train_config exp/speech_cleaner_voc_finetune_xeus_multi_all/config.yaml \
    --voc_model_file   exp/speech_cleaner_voc_finetune_xeus_multi_all/valid.loss_G.best.pth \
    --wav_scp          data/ami_ihm_longform/wav.scp \
    --segments         data/ami_ihm_longform/segments \
    --output_dir       exp/restored_xeus_multi/ami_ihm_longform \
    --batch_size 1 --device cuda --dtype bfloat16

# Fisher long-form — chunk inference (no segments file)
python local/enh_inference_speech_cleaner_longform.py \
    --fp_train_config  exp/speech_cleaner_fp_xeus_multi_all/config.yaml \
    --fp_model_file    exp/speech_cleaner_fp_xeus_multi_all/valid.loss.best.pth \
    --voc_train_config exp/speech_cleaner_voc_finetune_xeus_multi_all/config.yaml \
    --voc_model_file   exp/speech_cleaner_voc_finetune_xeus_multi_all/valid.loss_G.best.pth \
    --wav_scp          data/fisher_longform/wav.scp \
    --output_dir       exp/restored_xeus_multi/fisher_longform \
    --chunk_sec 20.0 --overlap_sec 0.5 \
    --batch_size 1 --device cuda --dtype bfloat16

"""

import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import yaml

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_SR  = 16000
OUTPUT_SR = 48000

_SEAMLESS_ENCODERS = {"w2v_bert2", "wavlm"}
_WAVEFORM_ENCODERS = {"xeus"}


# ---------------------------------------------------------------------------
# I/O helpers
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
    """Returns list of (utt_id, rec_id, start_sec, end_sec)."""
    segs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                segs.append((parts[0], parts[1], float(parts[2]), float(parts[3])))
    return segs


def load_wav(wav_cmd: str, target_sr: int = INPUT_SR) -> np.ndarray:
    wav_cmd = wav_cmd.strip()
    if wav_cmd.endswith("|"):
        proc = subprocess.run(wav_cmd[:-1].strip(), shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"sox pipe failed: {proc.stderr.decode()}")
        return (np.frombuffer(proc.stdout, dtype=np.int16)
                .astype(np.float32) / 32768.0)
    import librosa
    wav, sr = sf.read(wav_cmd, always_2d=True)
    wav = wav.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


def load_wav_segment(wav_cmd: str, start: float, end: float,
                     target_sr: int = INPUT_SR) -> np.ndarray:
    """Load a time-bounded segment using sox trim."""
    wav_cmd = wav_cmd.strip()
    if wav_cmd.endswith("|"):
        base = wav_cmd[:-1].strip()
        cmd  = f"{base} trim {start:.3f} ={end:.3f} |"
    else:
        cmd = (f"sox {wav_cmd} -r {target_sr} -c 1 -t wavpcm - "
               f"trim {start:.3f} ={end:.3f} |")
    return load_wav(cmd + " " if not cmd.endswith("|") else cmd, target_sr)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _yaml_to_namespace(config_path, extra_defaults=None):
    from argparse import Namespace
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    args = Namespace(**cfg)
    # Defaults biased toward XEUS multi-all (current main experiment)
    defaults = dict(
        ssl_encoder="xeus", ssl_encoder_conf=None,
        target_layer=10, lora_rank=64, lora_alpha=16, lora_dropout=0.1,
        input_sr=INPUT_SR, use_flash_attention=False,
        use_multilayer_loss=False, multilayer_mode="low",
        noise_dir="data/noise_pool", rir_dir="data/rir_pool",
        degrade_prob=0.5, online_degradation=True,
    )
    if extra_defaults:
        defaults.update(extra_defaults)
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


def load_models(fp_config, fp_model_file, voc_config, voc_model_file, device):
    from espnet2.tasks.speech_cleaner import SpeechCleanerFPTask, SpeechCleanerGANTask

    # FP model
    fp_args = _yaml_to_namespace(fp_config)
    fp_model = SpeechCleanerFPTask.build_model(fp_args)
    fp_state = torch.load(fp_model_file, map_location="cpu", weights_only=False)
    fp_model.load_state_dict(fp_state.get("model", fp_state), strict=False)
    fp_model.eval().to(device)
    ssl_type = getattr(fp_args, "ssl_encoder", "xeus")
    logger.info("FP loaded  [ssl=%s]: %s", ssl_type, fp_model_file)

    # Vocoder
    voc_args = _yaml_to_namespace(voc_config, extra_defaults=dict(
        ssl_dim=1024, use_predicted_feat=False, use_multilayer_feat=False,
        fp_model_path=None, mel_loss_weight=15.0,
        adv_loss_weight=2.0, fm_loss_weight=1.0, vocoder_type="hifigan",
    ))
    voc_model = SpeechCleanerGANTask.build_model(voc_args)
    voc_state = torch.load(voc_model_file, map_location="cpu", weights_only=False)
    voc_model.load_state_dict(voc_state.get("model", voc_state), strict=False)
    try:
        voc_model.vocoder.remove_weight_norm()
    except Exception:
        pass
    voc_model.vocoder.eval().to(device)
    logger.info("Vocoder loaded: %s", voc_model_file)

    # Processor (only for seamless encoders)
    processor = None
    if ssl_type in _SEAMLESS_ENCODERS:
        from transformers import SeamlessM4TFeatureExtractor
        processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    return fp_model, voc_model.vocoder, ssl_type, processor


# ---------------------------------------------------------------------------
# Single-chunk inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _infer_chunk(
    wav_np: np.ndarray,
    fp_model, vocoder, ssl_type: str, processor,
    device: str, torch_dtype: torch.dtype,
) -> np.ndarray:
    """Restore a single waveform chunk. Returns 48kHz float32 numpy."""
    use_amp = torch_dtype != torch.float32

    if ssl_type in _WAVEFORM_ENCODERS:
        wav_t   = torch.from_numpy(wav_np).unsqueeze(0).to(device)
        ilens_t = torch.tensor([len(wav_np)], device=device)
        ssl_inputs = {"waveform": wav_t, "ilens": ilens_t}
    else:
        wav_padded = np.pad(wav_np, (40, 40), mode="constant")
        ssl_inputs = dict(processor(
            [wav_padded], sampling_rate=INPUT_SR,
            return_tensors="pt", padding=True,
        ))
        ssl_inputs = {k: v.to(device) for k, v in ssl_inputs.items()}

    with torch.autocast(device_type=device.split(":")[0],
                        dtype=torch_dtype, enabled=use_amp):
        pred_feat, _ = fp_model.ssl_encoder(ssl_inputs)   # [1, T_ssl, D]
        wav_out      = vocoder.generate(pred_feat)         # [1, T_48k]

    return wav_out.squeeze(0).float().cpu().numpy()


# ---------------------------------------------------------------------------
# Long-form overlap-add
# ---------------------------------------------------------------------------

def _restore_longform(
    wav_np: np.ndarray,
    fp_model, vocoder, ssl_type, processor,
    device: str, torch_dtype: torch.dtype,
    chunk_sec: float = 20.0, overlap_sec: float = 0.5,
) -> np.ndarray:
    """Chunk-based inference with linear crossfade overlap-add."""
    chunk_s   = int(chunk_sec   * INPUT_SR)
    overlap_s = int(overlap_sec * INPUT_SR)
    hop_s     = chunk_s - overlap_s
    total_s   = len(wav_np)
    up_ratio  = OUTPUT_SR / INPUT_SR

    # Build chunk list
    chunks = []
    pos = 0
    while pos < total_s:
        end   = min(pos + chunk_s, total_s)
        chunk = wav_np[pos:end]
        if len(chunk) >= int(0.1 * INPUT_SR):
            chunks.append((pos, chunk))
        pos += hop_s

    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return _infer_chunk(chunks[0][1], fp_model, vocoder,
                            ssl_type, processor, device, torch_dtype)

    # Infer each chunk
    restored_chunks = []
    for i, (pos, chunk) in enumerate(chunks):
        try:
            out = _infer_chunk(chunk, fp_model, vocoder, ssl_type,
                               processor, device, torch_dtype)
            restored_chunks.append((pos, out))
        except Exception as e:
            logger.warning("  chunk %d failed: %s", i, e)
            restored_chunks.append(
                (pos, np.zeros(int(len(chunk) * up_ratio), dtype=np.float32)))

    # Overlap-add with linear crossfade
    out_len    = int(total_s * up_ratio) + int(overlap_sec * OUTPUT_SR)
    output     = np.zeros(out_len, dtype=np.float32)
    weight     = np.zeros(out_len, dtype=np.float32)
    overlap_out = int(overlap_sec * OUTPUT_SR)

    for i, (pos, out) in enumerate(restored_chunks):
        out_start = int(pos * up_ratio)
        out_end   = min(out_start + len(out), out_len)
        out       = out[:out_end - out_start]

        env = np.ones(len(out), dtype=np.float32)
        if i > 0 and overlap_out > 0:
            fl = min(overlap_out, len(out))
            env[:fl] = np.linspace(0.0, 1.0, fl)
        if i < len(restored_chunks) - 1 and overlap_out > 0:
            fl = min(overlap_out, len(out))
            env[-fl:] = np.minimum(env[-fl:], np.linspace(1.0, 0.0, fl))

        output[out_start:out_end] += out * env
        weight[out_start:out_end] += env

    valid = weight > 1e-6
    output[valid] /= weight[valid]
    return output[:int(total_s * up_ratio)].astype(np.float32)


def restore_one(
    wav_np: np.ndarray,
    fp_model, vocoder, ssl_type, processor,
    device: str, torch_dtype: torch.dtype,
    chunk_sec: float = 20.0, overlap_sec: float = 0.5,
) -> np.ndarray:
    """Route to single-chunk or long-form inference."""
    dur = len(wav_np) / INPUT_SR
    if dur <= chunk_sec:
        return _infer_chunk(wav_np, fp_model, vocoder, ssl_type,
                            processor, device, torch_dtype)
    return _restore_longform(wav_np, fp_model, vocoder, ssl_type, processor,
                              device, torch_dtype, chunk_sec, overlap_sec)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--fp_train_config",  required=True)
    p.add_argument("--fp_model_file",    required=True)
    p.add_argument("--voc_train_config", required=True)
    p.add_argument("--voc_model_file",   required=True)
    p.add_argument("--wav_scp",          required=True)
    p.add_argument("--output_dir",       required=True)
    p.add_argument("--segments",         default=None,
                   help="Kaldi segments file. If provided, each segment is "
                        "extracted from the recording and restored individually "
                        "(AMI IHM mode). Output uttids match segment uttids.")
    p.add_argument("--chunk_sec",   type=float, default=20.0,
                   help="Chunk length for long-form inference (seconds).")
    p.add_argument("--overlap_sec", type=float, default=0.5,
                   help="Overlap between chunks for crossfade (seconds).")
    p.add_argument("--batch_size",  type=int, default=1,
                   help="Batch size. >1 supported for XEUS utterance-level only.")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--dtype",       default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--max_entries", type=int, default=0,
                   help="Debug: limit number of entries processed.")
    return p


def main(cmd=None):
    args        = get_parser().parse_args(cmd)
    device      = args.device if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(torch, args.dtype)

    fp_model, vocoder, ssl_type, processor = load_models(
        args.fp_train_config, args.fp_model_file,
        args.voc_train_config, args.voc_model_file,
        device,
    )
    logger.info("ssl_encoder=%s  device=%s  dtype=%s", ssl_type, device, args.dtype)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_map   = read_wav_scp(args.wav_scp)
    scp_lines = []

    if args.segments:
        # ── Segment mode (AMI IHM or VAD-segmented Fisher) ───────────────
        segs = read_segments(args.segments)
        if args.max_entries > 0:
            segs = segs[:args.max_entries]
        logger.info("Segment mode: %d segments from %s", len(segs), args.segments)

        for i, (utt_id, rec_id, start, end) in enumerate(segs):
            if rec_id not in wav_map:
                logger.warning("Recording %s not in wav.scp, skipping", rec_id)
                continue
            try:
                wav_np  = load_wav_segment(wav_map[rec_id], start, end)
                restored = restore_one(
                    wav_np, fp_model, vocoder, ssl_type, processor,
                    device, torch_dtype, args.chunk_sec, args.overlap_sec,
                )
                out_path = out_dir / f"{utt_id}.wav"
                sf.write(str(out_path), restored, OUTPUT_SR)
                scp_lines.append(f"{utt_id} {out_path.resolve()}")
            except Exception as e:
                logger.error("FAILED segment %s: %s", utt_id, e)

            if (i + 1) % 200 == 0 or i == 0:
                logger.info("  [%d/%d] %s", i+1, len(segs), utt_id)
    else:
        # ── Full-recording mode (Fisher, AMI SDM longform) ────────────────
        entries = sorted(wav_map.items())
        if args.max_entries > 0:
            entries = entries[:args.max_entries]
        logger.info("Full-recording mode: %d recordings", len(entries))

        for i, (rec_id, wav_cmd) in enumerate(entries):
            try:
                wav_np  = load_wav(wav_cmd)
                dur     = len(wav_np) / INPUT_SR
                restored = restore_one(
                    wav_np, fp_model, vocoder, ssl_type, processor,
                    device, torch_dtype, args.chunk_sec, args.overlap_sec,
                )
                out_path = out_dir / f"{rec_id}.wav"
                sf.write(str(out_path), restored, OUTPUT_SR)
                scp_lines.append(f"{rec_id} {out_path.resolve()}")
            except Exception as e:
                logger.error("FAILED %s: %s", rec_id, e)

            if (i + 1) % 50 == 0 or i == 0:
                logger.info("  [%d/%d] %s (%.1fs)", i+1, len(entries), rec_id, dur)

    scp_path = out_dir / "wav.scp"
    with open(scp_path, "w") as f:
        f.write("\n".join(scp_lines) + "\n")
    logger.info("Done. %d entries → %s", len(scp_lines), out_dir)


if __name__ == "__main__":
    main()