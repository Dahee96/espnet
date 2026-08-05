#!/usr/bin/env python3
"""Speech Cleaner inference: Stage 1 FP + Stage 2/3 Vocoder.


Supports both SSL encoder backends:
  w2v_bert2 / wavlm  (_SEAMLESS_ENCODERS): uses SeamlessM4TFeatureExtractor,
                                            40-sample front/back padding,
                                            batch_size=1 recommended (variable length)
  xeus               (_WAVEFORM_ENCODERS): raw waveform input, pad/stack in batch,
                                           batch_size > 1 fully supported

Usage
-----
# XEUS multi-layer model (learnable aggregation, batch_size > 1 OK)
python -m espnet2.bin.enh_inference_speech_cleaner \
    --fp_train_config  exp/speech_cleaner_fp_xeus_multi_all_again/config.yaml \
    --fp_model_file    exp/speech_cleaner_fp_xeus_multi_all_again/valid.loss.best.pth \
    --voc_train_config exp/speech_cleaner_voc_finetune_xeus_multi_all_again/config.yaml \
    --voc_model_file   exp/speech_cleaner_voc_finetune_xeus_multi_all_again/59epoch.pth \
    --wav_scp          data/libritts_test-clean_16k/wav.scp \
    --output_dir       exp/restored_xeus_multi/test-clean \
    --batch_size 4 --device cuda --dtype bfloat16

# XEUS single-layer model (unchanged behavior)
python -m espnet2.bin.enh_inference_speech_cleaner \
    --fp_train_config  exp/speech_cleaner_fp_xeus_single/config.yaml \
    --fp_model_file    exp/speech_cleaner_fp_xeus_single/valid.loss.best.pth \
    --voc_train_config exp/speech_cleaner_voc_pretrain_xeus_single/config.yaml \
    --voc_model_file   exp/speech_cleaner_voc_pretrain_xeus_single/valid.loss_G.best.pth \
    --wav_scp          data/libritts_test-clean_16k/wav.scp \
    --output_dir       exp/restored_xeus_single/test-clean \
    --batch_size 4 --device cuda --dtype bfloat16

# w2v-BERT2 model
python -m espnet2.bin.enh_inference_speech_cleaner \
    --fp_train_config  exp/speech_cleaner_fp_w2v_bert2/config.yaml \
    --fp_model_file    exp/speech_cleaner_fp_w2v_bert2/valid.loss.best.pth \
    --voc_train_config exp/speech_cleaner_voc_pretrain_w2v_bert2/config.yaml \
    --voc_model_file   exp/speech_cleaner_voc_pretrain_w2v_bert2/valid.loss_G.best.pth \
    --wav_scp          data/libritts_test-clean_16k/wav.scp \
    --output_dir       exp/restored_w2v_bert2/test-clean \
    --batch_size 1 --device cuda --dtype bfloat16

Writes 48 kHz restored WAVs to output_dir/<uttid>.wav
and a Kaldi-style wav.scp to output_dir/wav.scp.
"""

import argparse
import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

INPUT_SR  = 16000
OUTPUT_SR = 48000

# Must match espnet2/tasks/speech_cleaner.py
_SEAMLESS_ENCODERS = {"w2v_bert2", "wavlm"}
_WAVEFORM_ENCODERS = {"xeus"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_wav_scp(path: str):
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def load_wav(wav_cmd: str, target_sr: int = INPUT_SR) -> np.ndarray:
    """Load from file path or sox pipe command ending with '|'."""
    wav_cmd = wav_cmd.strip()
    if wav_cmd.endswith("|"):
        proc = subprocess.run(
            wav_cmd[:-1].strip(), shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"sox pipe failed: {proc.stderr.decode()}")
        wav = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        import librosa
        wav, sr = sf.read(wav_cmd, always_2d=True)
        wav = wav.mean(axis=1).astype(np.float32)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_fp_model(config_path: str, model_file: str, device: str):
    from argparse import Namespace
    from espnet2.tasks.speech_cleaner import SpeechCleanerFPTask

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    args = Namespace(**cfg)

    defaults = dict(
        ssl_encoder="w2v_bert2", ssl_encoder_conf=None,
        target_layer=8, lora_rank=64, lora_alpha=16, lora_dropout=0.1,
        input_sr=INPUT_SR, use_flash_attention=False,
        use_multilayer_loss=False, multilayer_mode="low",
        noise_dir="data/noise_pool", rir_dir="data/rir_pool",
        degrade_prob=0.5, online_degradation=True,
    )
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    model = SpeechCleanerFPTask.build_model(args)
    state = torch.load(model_file, map_location="cpu", weights_only=False)
    sd    = state.get("model", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("FP missing keys (%d): %s ...", len(missing), missing[:3])
    model.eval().to(device)
    logger.info("FP loaded  [ssl_encoder=%s  target_layer=%s]: %s",
                getattr(args, "ssl_encoder", "?"),
                getattr(args, "target_layer", "?"),
                model_file)
    return model, getattr(args, "ssl_encoder", "w2v_bert2")


def load_espnet_vocoder(config_path: str, model_file: str, device: str):
    from argparse import Namespace
    from espnet2.tasks.speech_cleaner import SpeechCleanerGANTask

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    args = Namespace(**cfg)

    defaults = dict(
        ssl_encoder="w2v_bert2", ssl_encoder_conf=None,
        target_layer=8, lora_rank=64, lora_alpha=16, lora_dropout=0.1,
        input_sr=INPUT_SR, use_flash_attention=False,
        use_multilayer_loss=False, multilayer_mode="low",
        noise_dir="data/noise_pool", rir_dir="data/rir_pool",
        degrade_prob=0.5, online_degradation=True,
        ssl_dim=1024, use_predicted_feat=False, use_multilayer_feat=False,
        layer_weighting=None,   # NEW
        fp_model_path=None, mel_loss_weight=15.0,
        adv_loss_weight=2.0, fm_loss_weight=1.0, vocoder_type="hifigan",
    )
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    model = SpeechCleanerGANTask.build_model(args)
    state = torch.load(model_file, map_location="cpu", weights_only=False)
    sd    = state.get("model", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Vocoder missing keys (%d): %s ...", len(missing), missing[:5])
    if unexpected:
        logger.warning("Vocoder unexpected keys (%d): %s ...", len(unexpected), unexpected[:5])

    use_multilayer_feat = getattr(args, "use_multilayer_feat", False)
    layer_weighting     = getattr(args, "layer_weighting", None)

    # 체크포인트 키로 역추론하는 대신, config에 저장된 값을 그대로 신뢰.
    if use_multilayer_feat:
        has_layer_weights = any(k.endswith("layer_weights") for k in sd.keys())
        has_layer_router  = any("layer_router" in k for k in sd.keys())
        logger.info(
            "Multilayer sanity check: layer_weighting=%s, "
            "layer_weights_in_ckpt=%s, layer_router_in_ckpt=%s",
            layer_weighting, has_layer_weights, has_layer_router,
        )
        
        if layer_weighting == "global_learnable" and not has_layer_weights:
            logger.error("layer_weighting=global_learnable but no layer_weights in checkpoint!")
        if layer_weighting in ("utterance_dynamic", "frame_dynamic") and not has_layer_router:
            logger.error("layer_weighting=%s but no layer_router in checkpoint!", layer_weighting)

    try:
        model.vocoder.remove_weight_norm()
    except Exception:
        pass
    model.eval().to(device)

    if use_multilayer_feat:
        if getattr(model, "layer_weights", None) is not None:
            w = torch.softmax(model.layer_weights.detach(), dim=0)
            logger.info(
                "Vocoder loaded [MULTI-LAYER, global_learnable]: %s\n"
                "  layer_weights (softmax) = %s",
                model_file, [f"{x:.3f}" for x in w.tolist()],
            )
        elif getattr(model, "layer_router", None) is not None:
            logger.info(
                "Vocoder loaded [MULTI-LAYER, %s router]: %s",
                layer_weighting, model_file,
            )
        else:
            logger.info("Vocoder loaded [MULTI-LAYER, uniform]: %s", model_file)
    else:
        logger.info("Vocoder loaded [SINGLE-LAYER]: %s", model_file)

    return model, use_multilayer_feat


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def infer_batch(
    wav_list: list,
    fp_model,
    voc_model,                 # FULL SpeechCleanerVocoderModel (not .vocoder)
    use_multilayer_feat: bool, # whether to apply learned layer aggregation
    ssl_encoder_type: str,
    processor,
    device: str,
    torch_dtype: torch.dtype,
) -> list:
    """Run FP + vocoder on a batch of waveforms.

    XEUS (_WAVEFORM_ENCODERS):
      - pad to max_len, build {"waveform": [B, T], "ilens": [B]}
      - batch_size > 1 supported

    w2v-BERT2 / WavLM (_SEAMLESS_ENCODERS):
      - 40-sample front/back padding, SeamlessM4TFeatureExtractor handles
        variable-length padding internally
      - batch_size=1 recommended

    If use_multilayer_feat: extracts ALL SSL layers and applies the
    vocoder model's learned softmax-weighted aggregation (matches training
    exactly). Otherwise extracts only the single target_layer (unchanged
    from original behavior).
    """
    B       = len(wav_list)
    use_amp = torch_dtype != torch.float32

    if ssl_encoder_type in _WAVEFORM_ENCODERS:
        max_len = max(len(w) for w in wav_list)
        padded  = torch.zeros(B, max_len, dtype=torch.float32)
        ilens   = torch.zeros(B, dtype=torch.long)
        for i, w in enumerate(wav_list):
            padded[i, :len(w)] = torch.from_numpy(w)
            ilens[i] = len(w)
        ssl_inputs = {
            "waveform": padded.to(device),
            "ilens":    ilens.to(device),
        }
    else:
        padded_wavs = [np.pad(w, (40, 40), mode="constant") for w in wav_list]
        ssl_inputs  = dict(processor(
            padded_wavs, sampling_rate=INPUT_SR,
            return_tensors="pt", padding=True,
        ))
        ssl_inputs = {k: v.to(device) for k, v in ssl_inputs.items()}

    # IMPORTANT: only the SSL encoder ran under bfloat16 autocast during
    # training (_get_ssl_feat). The vocoder ALWAYS ran in plain float32
    # (autocast block closes via feat.float() before vocoder.generate() is
    # called in _forward_G/_forward_D). Replicate that exactly here --
    # vocoder.generate() must be OUTSIDE the autocast context, otherwise
    # PyTorch's autocast will silently downcast its conv ops to bfloat16,
    # a precision the vocoder was never trained/validated under.
    with torch.autocast(device_type=device.split(":")[0],
                        dtype=torch_dtype, enabled=use_amp):
        if use_multilayer_feat:
            # Extract ALL layers, then apply the model's LEARNED weighted sum
            # -- matches exactly what _get_ssl_feat() does during training.
            _, others = fp_model.ssl_encoder(ssl_inputs, use_multilayer=True)
            feat_list = others["all_feats"]
            pred_feat = voc_model._weighted_sum(feat_list)
        else:
            pred_feat, _ = fp_model.ssl_encoder(ssl_inputs, use_multilayer=False)
        pred_feat = pred_feat.float()

    # vocoder.generate() runs in full float32, exactly as during training
    wav_out = voc_model.vocoder.generate(pred_feat)

    results = []
    for i, w in enumerate(wav_list):
        expected = len(w) * 3
        out_np   = wav_out[i].float().cpu().numpy()
        results.append(out_np[:expected].astype(np.float32))
    return results


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
    p.add_argument("--batch_size",       type=int,  default=4,
                   help="Utterances per batch. >1 fully supported for XEUS; "
                        "use 1 for w2v-BERT2/WavLM.")
    p.add_argument("--device",           default="cuda")
    p.add_argument("--dtype",            default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    return p


def main(cmd=None):
    args        = get_parser().parse_args(cmd)
    device      = args.device if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(torch, args.dtype)

    fp_model, ssl_encoder_type = load_fp_model(
        args.fp_train_config, args.fp_model_file, device)
    voc_model, use_multilayer_feat = load_espnet_vocoder(
        args.voc_train_config, args.voc_model_file, device)

    processor = None
    if ssl_encoder_type in _SEAMLESS_ENCODERS:
        from transformers import SeamlessM4TFeatureExtractor
        processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        if args.batch_size > 1:
            logger.warning(
                "ssl_encoder=%s: batch_size=%d may cause large padding overhead; "
                "batch_size=1 is recommended.",
                ssl_encoder_type, args.batch_size,
            )

    logger.info("ssl_encoder=%s  use_multilayer_feat=%s  device=%s  dtype=%s  batch_size=%d",
                ssl_encoder_type, use_multilayer_feat, device, args.dtype, args.batch_size)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_map    = read_wav_scp(args.wav_scp)
    uttids     = list(wav_map.keys())
    scp_lines  = []
    n_total    = len(uttids)

    for start in range(0, n_total, args.batch_size):
        batch_ids  = uttids[start:start + args.batch_size]
        batch_wavs = []
        valid_ids  = []

        for uid in batch_ids:
            try:
                wav = load_wav(wav_map[uid], INPUT_SR)
                batch_wavs.append(wav)
                valid_ids.append(uid)
            except Exception as e:
                logger.error("Load failed %s: %s", uid, e)

        if not batch_wavs:
            continue

        try:
            restored_list = infer_batch(
                batch_wavs, fp_model, voc_model, use_multilayer_feat,
                ssl_encoder_type, processor, device, torch_dtype,
            )
        except Exception as e:
            logger.error("Batch inference failed [%s]: %s", valid_ids, e)
            continue

        for uid, restored in zip(valid_ids, restored_list):
            out_path = out_dir / f"{uid}.wav"
            sf.write(str(out_path), restored, OUTPUT_SR)
            scp_lines.append(f"{uid} {out_path.resolve()}")

        done = min(start + args.batch_size, n_total)
        if done % 500 == 0 or done == n_total or start == 0:
            logger.info("  [%d/%d]", done, n_total)

    scp_path = out_dir / "wav.scp"
    with open(scp_path, "w") as f:
        f.write("\n".join(scp_lines) + "\n")
    logger.info("Done. %d/%d files -> %s", len(scp_lines), n_total, out_dir)


if __name__ == "__main__":
    main()