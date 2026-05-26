#!/usr/bin/env python3
"""Speech Cleaner inference: restore noisy speech using Stage 1 + Stage 3.

Usage
-----
python -m espnet2.bin.enh_inference_speech_cleaner \
    --fp_train_config  exp/speech_cleaner_fp/config.yaml \
    --fp_model_file    exp/speech_cleaner_fp/valid.loss.best.pth \
    --voc_train_config exp/speech_cleaner_voc_finetune/config.yaml \
    --voc_model_file   exp/speech_cleaner_voc_finetune/valid.loss.G.best.pth \
    --wav_scp          data/test-other/wav.scp \
    --output_dir       exp/restored/test-other \
    --batch_size       8 \
    --dtype            bfloat16

Writes 48 kHz restored WAVs to output_dir/<uttid>.wav.
"""

import argparse
import logging
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import yaml

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class SpeechCleanerInference:
    """Two-stage inference pipeline.

    Args:
        fp_train_config  : path to Stage-1 config.yaml
        fp_model_file    : path to Stage-1 *.pth
        voc_train_config : path to Stage-3 config.yaml
        voc_model_file   : path to Stage-3 *.pth
        device           : 'cuda' or 'cpu'
        dtype            : 'float32', 'float16', or 'bfloat16'
        batch_size       : utterances per batch
    """

    def __init__(
        self,
        fp_train_config: str,
        fp_model_file: str,
        voc_train_config: str,
        voc_model_file: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 8,
    ):
        from espnet2.tasks.speech_cleaner import (
            SpeechCleanerFPTask,
            SpeechCleanerGANTask,
        )

        self.device     = torch.device(device)
        self.torch_dtype = getattr(torch, dtype)
        self.batch_size  = batch_size

        # Feature predictor
        with open(fp_train_config) as f:
            fp_args = argparse.Namespace(**yaml.safe_load(f))
        fp_model = SpeechCleanerFPTask.build_model(fp_args)
        state    = torch.load(fp_model_file, map_location="cpu")
        fp_model.load_state_dict(state.get("model", state), strict=True)
        fp_model.eval().to(self.device)
        self.fp = fp_model.feature_predictor

        # Vocoder (load full GAN model, extract vocoder sub-module)
        with open(voc_train_config) as f:
            voc_args = argparse.Namespace(**yaml.safe_load(f))
        voc_model = SpeechCleanerGANTask.build_model(voc_args)
        state     = torch.load(voc_model_file, map_location="cpu")
        voc_model.load_state_dict(state.get("model", state), strict=False)
        voc_model.vocoder.remove_weight_norm()
        voc_model.vocoder.eval().to(self.device)
        self.vocoder = voc_model.vocoder

        logger.info("Ready  device=%s  dtype=%s  batch=%d",
                    device, dtype, batch_size)

    @torch.inference_mode()
    def restore_batch(self, noisy_wavs: list) -> list:
        """
        Args:
            noisy_wavs : list of float32 numpy arrays at 16 kHz
        Returns:
            list of float32 numpy arrays at 48 kHz
        """
        B       = len(noisy_wavs)
        max_len = max(len(w) for w in noisy_wavs)
        padded  = torch.zeros(B, max_len)
        ilens   = torch.zeros(B, dtype=torch.long)
        for i, w in enumerate(noisy_wavs):
            padded[i, :len(w)] = torch.from_numpy(w.astype(np.float32))
            ilens[i] = len(w)
        padded = padded.to(self.device)
        ilens  = ilens.to(self.device)

        use_amp = self.torch_dtype != torch.float32
        with torch.autocast(self.device.type, dtype=self.torch_dtype,
                            enabled=use_amp):
            _, others   = self.fp(padded, ilens)
            pred_feat   = others["pred_ssl_feat"]
            restored    = self.vocoder.generate(pred_feat)   # [B, T_48k]

        out = []
        for i in range(B):
            T = int(ilens[i].item() * 3)  # 16k → 48k = ×3
            out.append(restored[i, :T].float().cpu().numpy())
        return out

    def restore_wav_scp(self, wav_scp: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        utt_ids, wav_paths = [], []
        with open(wav_scp) as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    utt_ids.append(parts[0])
                    wav_paths.append(parts[1])

        logger.info("%d utterances to restore", len(utt_ids))
        for start in range(0, len(utt_ids), self.batch_size):
            end    = min(start + self.batch_size, len(utt_ids))
            batch_wavs = []
            for p in wav_paths[start:end]:
                wav, sr = sf.read(p, dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(1)
                if sr != 16000:
                    wav = AF.resample(torch.from_numpy(wav), sr, 16000).numpy()
                batch_wavs.append(wav)

            restored = self.restore_batch(batch_wavs)
            for uttid, wav in zip(utt_ids[start:end], restored):
                sf.write(os.path.join(output_dir, f"{uttid}.wav"), wav, 48000)
            logger.info("  [%d/%d]", end, len(utt_ids))

        logger.info("Saved to %s", output_dir)


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--fp_train_config",  required=True)
    p.add_argument("--fp_model_file",    required=True)
    p.add_argument("--voc_train_config", required=True)
    p.add_argument("--voc_model_file",   required=True)
    p.add_argument("--wav_scp",          required=True)
    p.add_argument("--output_dir",       required=True)
    p.add_argument("--batch_size",       type=int, default=8)
    p.add_argument("--device",           default="cuda")
    p.add_argument("--dtype",            default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    return p


def main(cmd=None):
    args = get_parser().parse_args(cmd)
    inf  = SpeechCleanerInference(
        fp_train_config  = args.fp_train_config,
        fp_model_file    = args.fp_model_file,
        voc_train_config = args.voc_train_config,
        voc_model_file   = args.voc_model_file,
        device           = args.device,
        dtype            = args.dtype,
        batch_size       = args.batch_size,
    )
    inf.restore_wav_scp(args.wav_scp, args.output_dir)


if __name__ == "__main__":
    main()
