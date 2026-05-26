"""ESPnet2 task definitions for Speech Cleaner (speech restoration).

Key design:
  - _FPCollateFn / _GANCollateFn are ssl_encoder-aware:
      w2v_bert2, wavlm → SeamlessM4TFeatureExtractor → {"input_features", "attention_mask"}
      xeus             → raw waveform → {"waveform", "ilens"}
  - Crop BEFORE padding (max_duration).
  - 40-sample waveform padding before processor (Sidon-faithful, w2v_bert2/wavlm only).
  - Warmup 2000 steps (Sidon-faithful).
  - plot_attention overridden (no-op) to avoid ESPnet internal crash with dict batch values.
"""

import argparse
import random
from typing import Callable, Tuple

import numpy as np
import torch

from espnet2.tasks.abs_task import AbsTask
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.trainer import Trainer
from espnet2.train.gan_trainer import GANTrainer
from espnet2.utils.types import str2bool, str_or_none

from espnet2.enh.speech_cleaner_model import (
    W2vBert2Encoder,
    XeusEncoder,
    WavLMEncoder,
    SpeechCleanerFPModel,
    SpeechCleanerVocoderModel,
)
from espnet2.enh.decoder.speech_cleaner_vocoder import (
    DACVocoder,
    SpeechCleanerVocoder,
)

# ---------------------------------------------------------------------------
# SSL encoder registry
# ---------------------------------------------------------------------------
SSL_ENCODER_CLASSES = dict(
    w2v_bert2=W2vBert2Encoder,
    xeus=XeusEncoder,
    wavlm=WavLMEncoder,
)

# Encoders that need SeamlessM4TFeatureExtractor preprocessing
_SEAMLESS_ENCODERS = {"w2v_bert2", "wavlm"}
# Encoders that take raw waveform directly
_WAVEFORM_ENCODERS = {"xeus"}


def _build_ssl_encoder(args: argparse.Namespace):
    name = getattr(args, "ssl_encoder", "w2v_bert2")
    if name not in SSL_ENCODER_CLASSES:
        raise ValueError(f"Unknown ssl_encoder '{name}'. Valid: {list(SSL_ENCODER_CLASSES)}")
    conf = dict(getattr(args, "ssl_encoder_conf", None) or {})
    conf.setdefault("target_layer", args.target_layer)
    conf.setdefault("lora_rank",    args.lora_rank)
    conf.setdefault("lora_alpha",   args.lora_alpha)
    conf.setdefault("lora_dropout", args.lora_dropout)
    conf.setdefault("input_sr",     args.input_sr)
    conf.setdefault("use_flash_attention", getattr(args, "use_flash_attention", False))
    return SSL_ENCODER_CLASSES[name](**conf)


def _add_ssl_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("Speech Cleaner — SSL encoder")
    g.add_argument("--ssl_encoder", type=str, default="w2v_bert2",
                   choices=list(SSL_ENCODER_CLASSES))
    g.add_argument("--ssl_encoder_conf", action=_NestedDictAction, default=None)
    g.add_argument("--target_layer", type=int, default=8)
    g.add_argument("--lora_rank",    type=int,   default=64)
    g.add_argument("--lora_alpha",   type=int,   default=16)
    g.add_argument("--lora_dropout", type=float, default=0.1)
    g.add_argument("--max_duration", type=float, default=20.0,
                   help="Max utterance duration (s). Cropped BEFORE padding.")
    g.add_argument("--input_sr", type=int, default=16000)
    g.add_argument("--warmup_steps", type=int, default=2000)


class _NestedDictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, dict):
            setattr(namespace, self.dest, values)
            return
        d = {}
        for item in (values if isinstance(values, list) else [values]):
            if "=" in item:
                k, v = item.split("=", 1)
                if v.lower() == "true":    v = True
                elif v.lower() == "false": v = False
                else:
                    try:    v = int(v)
                    except:
                        try: v = float(v)
                        except: pass
                d[k] = v
        setattr(namespace, self.dest, d)


# =============================================================================
# Picklable collate functors
# =============================================================================

class _FPCollateFn:
    """Stage 1 collate: crop → pad → build ssl_inputs.

    w2v_bert2 / wavlm:
        ssl_inputs = SeamlessM4TFeatureExtractor output
                     {"input_features": [B,T,160], "attention_mask": [B,T]}
        40-sample padding added before processor (Sidon-faithful).

    xeus:
        ssl_inputs = {"waveform": [B,T], "ilens": [B]}
        Raw padded waveform passed directly.
    """
    def __init__(self, max_samples: int, ssl_encoder: str = "w2v_bert2",
                 input_sr: int = 16000,
                 processor_tag: str = "facebook/w2v-bert-2.0"):
        self.max_samples = max_samples
        self.ssl_encoder = ssl_encoder
        self.input_sr    = input_sr
        self._base = CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

        if ssl_encoder in _SEAMLESS_ENCODERS:
            from transformers import SeamlessM4TFeatureExtractor
            self.processor = SeamlessM4TFeatureExtractor.from_pretrained(processor_tag)
        else:
            self.processor = None

    def __call__(self, data):
        # Step 1: crop each utterance independently BEFORE padding
        clipped = []
        for key, d in data:
            new_d = dict(d)
            noisy = d.get("noisy_speech")
            if noisy is not None and noisy.shape[0] > self.max_samples:
                T     = noisy.shape[0]
                start = random.randint(0, T - self.max_samples)
                new_d["noisy_speech"] = noisy[start:start + self.max_samples]
                if "speech_ref1" in d:
                    new_d["speech_ref1"] = d["speech_ref1"][start:start + self.max_samples]
            clipped.append((key, new_d))

        # Step 2: standard padding
        keys, batch = self._base(clipped)

        # Step 3: build ssl_inputs based on encoder type
        if self.ssl_encoder in _SEAMLESS_ENCODERS:
            # SeamlessM4TFeatureExtractor with 40-sample padding (Sidon-faithful)
            noisy_np = self._wav_to_numpy_list(
                batch["noisy_speech"], batch["noisy_speech_lengths"], pad40=True
            )
            clean_np = self._wav_to_numpy_list(
                batch["speech_ref1"], batch["speech_ref1_lengths"], pad40=True
            )
            batch["noisy_speech_ssl"] = dict(self.processor(
                noisy_np, sampling_rate=self.input_sr,
                return_tensors="pt", padding=True,
            ))
            batch["speech_ref1_ssl"] = dict(self.processor(
                clean_np, sampling_rate=self.input_sr,
                return_tensors="pt", padding=True,
            ))
        else:
            # xeus: pass raw padded waveform + lengths
            batch["noisy_speech_ssl"] = {
                "waveform": batch["noisy_speech"],
                "ilens":    batch["noisy_speech_lengths"],
            }
            batch["speech_ref1_ssl"] = {
                "waveform": batch["speech_ref1"],
                "ilens":    batch["speech_ref1_lengths"],
            }

        return keys, batch

    def _wav_to_numpy_list(self, wav_tensor: torch.Tensor,
                           lengths: torch.Tensor, pad40: bool = False):
        B = wav_tensor.shape[0]
        result = []
        for i in range(B):
            w = wav_tensor[i, :lengths[i]].float().numpy()
            if pad40:
                w = np.pad(w, (40, 40), mode="constant")
            result.append(w)
        return result


class _GANCollateFn:
    """Stage 2/3 collate: crop (48k ref + 16k noisy) → pad → build ssl_inputs."""
    def __init__(self, max_16k: int, max_48k: int,
                 ssl_encoder: str = "w2v_bert2",
                 input_sr: int = 16000,
                 processor_tag: str = "facebook/w2v-bert-2.0"):
        self.max_16k     = max_16k
        self.max_48k     = max_48k
        self.ssl_encoder = ssl_encoder
        self.input_sr    = input_sr
        self._base = CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

        if ssl_encoder in _SEAMLESS_ENCODERS:
            from transformers import SeamlessM4TFeatureExtractor
            self.processor = SeamlessM4TFeatureExtractor.from_pretrained(processor_tag)
        else:
            self.processor = None

    def __call__(self, data):
        clipped = []
        for key, d in data:
            new_d = dict(d)
            ref = d.get("speech_ref1")
            if ref is not None and ref.shape[0] > self.max_48k:
                T48       = ref.shape[0]
                start_48k = random.randint(0, T48 - self.max_48k)
                new_d["speech_ref1"] = ref[start_48k:start_48k + self.max_48k]
                if "noisy_speech" in d:
                    noisy     = d["noisy_speech"]
                    start_16k = min(start_48k // 3, max(0, noisy.shape[0] - self.max_16k))
                    new_d["noisy_speech"] = noisy[start_16k:start_16k + self.max_16k]
            clipped.append((key, new_d))

        keys, batch = self._base(clipped)

        if self.ssl_encoder in _SEAMLESS_ENCODERS:
            noisy_np    = self._wav_to_numpy_list(
                batch["noisy_speech"], batch["noisy_speech_lengths"], pad40=True
            )
            clean_16k_np = self._ref48_to_16k_numpy_list(
                batch["speech_ref1"], batch["speech_ref1_lengths"]
            )
            batch["noisy_speech_ssl"] = dict(self.processor(
                noisy_np, sampling_rate=self.input_sr,
                return_tensors="pt", padding=True,
            ))
            batch["speech_ref1_ssl"] = dict(self.processor(
                clean_16k_np, sampling_rate=self.input_sr,
                return_tensors="pt", padding=True,
            ))
        else:
            # xeus: raw waveform
            # noisy is 16k, ref is 48k → downsample ref for ssl encoder
            import torchaudio.functional as AF_
            ref_16k = AF_.resample(batch["speech_ref1"], 48000, self.input_sr)
            ref_lens_16k = (batch["speech_ref1_lengths"].float() / 3.0).long()
            batch["noisy_speech_ssl"] = {
                "waveform": batch["noisy_speech"],
                "ilens":    batch["noisy_speech_lengths"],
            }
            batch["speech_ref1_ssl"] = {
                "waveform": ref_16k,
                "ilens":    ref_lens_16k,
            }

        return keys, batch

    def _wav_to_numpy_list(self, wav_tensor, lengths, pad40=False):
        B = wav_tensor.shape[0]
        result = []
        for i in range(B):
            w = wav_tensor[i, :lengths[i]].float().numpy()
            if pad40:
                w = np.pad(w, (40, 40), mode="constant")
            result.append(w)
        return result

    def _ref48_to_16k_numpy_list(self, wav_tensor, lengths):
        import torchaudio.functional as AF_
        B = wav_tensor.shape[0]
        result = []
        for i in range(B):
            w = wav_tensor[i, :lengths[i]].float()
            w = AF_.resample(w.unsqueeze(0), 48000, self.input_sr).squeeze(0)
            w_np = np.pad(w.numpy(), (40, 40), mode="constant")
            result.append(w_np)
        return result


# =============================================================================
# Stage 1 Task — Feature Predictor
# =============================================================================

class SpeechCleanerFPTask(AbsTask):
    num_optimizers: int = 1

    @classmethod
    def plot_attention(cls, *args, **kwargs) -> None:
        """No-op: speech restoration has no attention to plot.
        Overridden to avoid ESPnet internal crash when batch contains dict values.
        Use num_att_plot: 0 in yaml as well.
        """
        pass

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        _add_ssl_args(parser)

    @classmethod
    def build_collate_fn(cls, args, train: bool) -> Callable:
        max_16k     = int(getattr(args, "max_duration", 20.0) * 16000)
        input_sr    = getattr(args, "input_sr", 16000)
        ssl_encoder = getattr(args, "ssl_encoder", "w2v_bert2")
        return _FPCollateFn(
            max_samples=max_16k,
            ssl_encoder=ssl_encoder,
            input_sr=input_sr,
            processor_tag="facebook/w2v-bert-2.0",
        )

    @classmethod
    def build_preprocess_fn(cls, args, train: bool):
        return None

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        return ("noisy_speech",) if inference else ("noisy_speech", "speech_ref1")

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        return ()

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> SpeechCleanerFPModel:
        ssl_encoder = _build_ssl_encoder(args)
        return SpeechCleanerFPModel(ssl_encoder=ssl_encoder)

    @classmethod
    def build_optimizers(cls, args, model):
        conf  = dict(getattr(args, "optim_conf", None) or {})
        betas = tuple(conf.get("betas", [0.8, 0.98]))
        trainable = [p for p in model.parameters() if p.requires_grad]
        return [torch.optim.AdamW(
            trainable,
            lr=conf.get("lr", 1e-4),
            betas=betas,
            weight_decay=conf.get("weight_decay", 0.01),
        )]

    @classmethod
    def build_scheduler(cls, args, optimizers):
        warmup = getattr(args, "warmup_steps", 2000)
        try:
            from transformers import get_constant_schedule_with_warmup
            return [get_constant_schedule_with_warmup(
                optimizers[0], num_warmup_steps=warmup
            )]
        except ImportError:
            def warmup_fn(step):
                return min(1.0, step / max(warmup, 1))
            return [torch.optim.lr_scheduler.LambdaLR(optimizers[0], warmup_fn)]

    @classmethod
    def get_trainer(cls) -> type:
        return Trainer


# =============================================================================
# Stage 2/3 Task — Vocoder GAN
# =============================================================================

class SpeechCleanerGANTask(AbsTask):
    num_optimizers: int = 2
    trainer = GANTrainer

    @classmethod
    def plot_attention(cls, *args, **kwargs) -> None:
        pass

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        _add_ssl_args(parser)
        g = parser.add_argument_group("Speech Cleaner — Vocoder GAN")
        g.add_argument("--ssl_dim",            type=int,       default=1024)
        g.add_argument("--use_predicted_feat", type=str2bool,  default=False)
        g.add_argument("--fp_model_path",      type=str_or_none, default=None)
        g.add_argument("--mel_loss_weight",    type=float,     default=15.0)
        g.add_argument("--adv_loss_weight",    type=float,     default=2.0)
        g.add_argument("--fm_loss_weight",     type=float,     default=1.0)
        g.add_argument("--vocoder_type",       type=str,       default="hifigan",
                       choices=["hifigan", "dac"])

    @classmethod
    def build_collate_fn(cls, args, train: bool) -> Callable:
        max_dur     = getattr(args, "max_duration", 20.0)
        input_sr    = getattr(args, "input_sr", 16000)
        ssl_encoder = getattr(args, "ssl_encoder", "w2v_bert2")
        return _GANCollateFn(
            max_16k=int(max_dur * 16000),
            max_48k=int(max_dur * 48000),
            ssl_encoder=ssl_encoder,
            input_sr=input_sr,
            processor_tag="facebook/w2v-bert-2.0",
        )

    @classmethod
    def build_preprocess_fn(cls, args, train: bool):
        return None

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        return ("noisy_speech",) if inference else ("noisy_speech", "speech_ref1")

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        return ()

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> SpeechCleanerVocoderModel:
        ssl_encoder = _build_ssl_encoder(args)

        fp_path = getattr(args, "fp_model_path", None)
        if fp_path is not None:
            state  = torch.load(fp_path, map_location="cpu", weights_only=False)
            sd     = state.get("model", state)
            enc_sd = {
                k.replace("ssl_encoder.", "", 1): v
                for k, v in sd.items() if k.startswith("ssl_encoder.")
            }
            missing, _ = ssl_encoder.load_state_dict(enc_sd, strict=False)
            if missing:
                import logging
                logging.getLogger(__name__).warning("ssl_encoder missing: %s", missing[:5])
            for p in ssl_encoder.parameters():
                p.requires_grad = False

        ssl_dim      = args.ssl_dim if args.ssl_dim > 0 else ssl_encoder.ssl_dim
        vocoder_type = getattr(args, "vocoder_type", "hifigan")
        if vocoder_type == "dac":
            vocoder = DACVocoder(input_dim=ssl_dim)
        else:
            vocoder = SpeechCleanerVocoder(input_dim=ssl_dim)

        return SpeechCleanerVocoderModel(
            ssl_encoder=ssl_encoder,
            vocoder=vocoder,
            use_predicted_feat=args.use_predicted_feat,
            mel_loss_weight=args.mel_loss_weight,
            adv_loss_weight=args.adv_loss_weight,
            fm_loss_weight=args.fm_loss_weight,
        )

    @classmethod
    def build_optimizers(cls, args, model):
        conf = dict(getattr(args, "optim_conf", None) or {})
        lr   = conf.get("lr", 1e-4)
        wd   = conf.get("weight_decay", 0.01)
        b    = tuple(conf.get("betas", [0.8, 0.98]))
        return [
            torch.optim.AdamW(list(model.vocoder.parameters()),
                              lr=lr, betas=b, weight_decay=wd),
            torch.optim.AdamW(
                list(model.mpd.parameters()) + list(model.msd.parameters()),
                lr=lr, betas=b, weight_decay=wd),
        ]

    @classmethod
    def build_scheduler(cls, args, optimizers):
        warmup = getattr(args, "warmup_steps", 2000)
        gamma  = (dict(getattr(args, "scheduler_conf", None) or {})).get(
            "gamma", 0.999996
        )
        schedulers = []
        for opt in optimizers:
            try:
                from transformers import get_constant_schedule_with_warmup
                schedulers.append(
                    get_constant_schedule_with_warmup(opt, num_warmup_steps=warmup)
                )
            except ImportError:
                schedulers.append(
                    torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
                )
        return schedulers

    @classmethod
    def get_trainer(cls) -> type:
        return GANTrainer