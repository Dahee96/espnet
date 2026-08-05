"""Speech Cleaner: unified speech restoration model for ESPnet2.

Attention optimizations
-----------------------
W2vBert2Encoder (SDPA monkey-patch):
  w2v-BERT 2.0 uses Conformer with relative positional encoding (Transformer-XL style).
  Flash Attention cannot be applied directly because positional biases must be
  injected into the attention score matrix before softmax.
  Solution: compute positional bias (scores_ac + scores_bd) as usual, then pass
  as attn_mask to F.scaled_dot_product_attention (additive bias convention).
  ~10-20% speedup vs eager. Requires PyTorch >= 2.0, no extra packages.

XeusEncoder (flash_attn):
  ESPnet E-Branchformer supports use_flash_attn=True in encoder_conf.
  Enabled by patching config.yaml before SSLTask.build_model_from_file.
  Requires flash_attn >= 2.0 (available on Delta PC).

Memory optimization via use_multilayer_loss
--------------------------------------------
W2vBert2Encoder / WavLMEncoder:
  use_multilayer_loss=False or multilayer_mode="low" -> load only target_layer layers.
  use_multilayer_loss=True and mode in ("up","all")  -> load full model.
  Equivalent to Sidon: Wav2Vec2BertModel(num_hidden_layers=8) for target_layer=8.

XeusEncoder:
  Always loads all layers (inference_encode runs full forward pass).
  use_multilayer_loss accepted for API consistency, no effect on loading.

References: Nakata et al., arXiv:2509.17052, 2026.
"""

from __future__ import annotations

import abc
import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF

import os
import importlib.util as _ilu
import numpy as np   

try:
    import audiotools
    from audiotools import AudioSignal
    from dac.nn.loss import MelSpectrogramLoss, MultiScaleSTFTLoss
    from dac.nn.loss import L1Loss as DACWavLoss
    _DAC_AVAILABLE = True
except ImportError:
    _DAC_AVAILABLE = False

from torch.nn.utils import remove_weight_norm, weight_norm
from typeguard import typechecked
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel

logger = logging.getLogger(__name__)


# =============================================================================
# SDPA monkey-patch for Wav2Vec2BertSelfAttention
# =============================================================================

def _patch_wav2vec2bert_sdpa(model: nn.Module) -> int:
    """Replace eager attention with SDPA in all Wav2Vec2BertSelfAttention layers.

    For relative positional encoding (Transformer-XL, used by w2v-BERT 2.0):
      Positional bias (scores_ac + scores_bd) is computed as usual, then passed
      as attn_mask to F.scaled_dot_product_attention (additive bias convention).
      Mathematically equivalent to eager. Returns number of patched layers.
    """
    try:
        from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
            Wav2Vec2BertSelfAttention,
        )
    except ImportError:
        logger.warning("Could not import Wav2Vec2BertSelfAttention; SDPA patch skipped")
        return 0

    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        logger.warning(
            "F.scaled_dot_product_attention not available (PyTorch >= 2.0 required); "
            "SDPA patch skipped"
        )
        return 0

    patched = 0

    def _sdpa_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        if output_attentions:
            return _original_forward(
                self, hidden_states, attention_mask,
                relative_position_embeddings, output_attentions,
            )

        batch_size, seq_len, _ = hidden_states.size()
        query_key_states = hidden_states
        value_states = hidden_states

        if self.position_embeddings_type == "rotary":
            if relative_position_embeddings is None:
                raise ValueError("relative_position_embeddings required for rotary")
            query_key_states = self._apply_rotary_embedding(
                query_key_states, relative_position_embeddings
            )

        query = self.linear_q(query_key_states).view(
            batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        key   = self.linear_k(query_key_states).view(
            batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        value = self.linear_v(value_states).view(
            batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        if self.position_embeddings_type == "relative":
            if relative_position_embeddings is None:
                raise ValueError("relative_position_embeddings required for relative")
            q_t = query.transpose(1, 2)
            q_with_bias_u = (q_t + self.pos_bias_u).transpose(1, 2)
            q_with_bias_v = (q_t + self.pos_bias_v).transpose(1, 2)
            scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))

            proj_pos = self.linear_pos(relative_position_embeddings)
            proj_pos = proj_pos.view(
                relative_position_embeddings.size(0), -1,
                self.num_heads, self.head_size
            ).transpose(1, 2).transpose(2, 3)
            scores_bd = torch.matmul(q_with_bias_v, proj_pos)
            scores_bd = self._relative_position_bucket(scores_bd, seq_len)

            pos_bias = (scores_ac + scores_bd) / math.sqrt(self.head_size)
            sdpa_attn_mask = pos_bias + attention_mask if attention_mask is not None \
                             else pos_bias
            out = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=sdpa_attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=1.0,
            )

        elif self.position_embeddings_type == "relative_key":
            query_length = query.shape[2]
            key_length   = key.shape[2]
            pos_ids_l = torch.arange(query_length, device=hidden_states.device).view(-1, 1)
            pos_ids_r = torch.arange(key_length,   device=hidden_states.device).view(1, -1)
            distance  = torch.clamp(
                pos_ids_r - pos_ids_l,
                -self.left_max_position_embeddings,
                self.right_max_position_embeddings,
            )
            pos_emb  = self.distance_embedding(
                distance + self.left_max_position_embeddings
            ).to(query.dtype)
            rel_bias = torch.einsum("bhld,lrd->bhlr", query, pos_emb) / math.sqrt(self.head_size)
            combined = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size) \
                       + rel_bias
            if attention_mask is not None:
                combined = combined + attention_mask
            out = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=combined,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=1.0,
            )

        else:
            out = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_size),
            )

        out = out.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        out = self.linear_out(out)
        return out, None

    def _relative_position_bucket(self_obj, scores_bd, seq_len):
        """Transformer-XL shift: [B, heads, T, 2T-1] -> [B, heads, T, T]."""
        batch, heads, qlen, pos_len = scores_bd.shape
        zeros = scores_bd.new_zeros(batch, heads, qlen, 1)
        scores_bd = torch.cat([zeros, scores_bd], dim=-1)
        scores_bd = scores_bd.view(batch, heads, pos_len + 1, qlen)
        scores_bd = scores_bd[:, :, 1:, :].view(batch, heads, qlen, pos_len)
        return scores_bd[:, :, :, :seq_len]

    for module in model.modules():
        if isinstance(module, Wav2Vec2BertSelfAttention):
            if not hasattr(module, "_original_forward_saved"):
                _original_forward = Wav2Vec2BertSelfAttention.forward
                module._original_forward_saved = _original_forward
                import types
                module._relative_position_bucket = types.MethodType(
                    _relative_position_bucket, module
                )
                module.forward = types.MethodType(_sdpa_forward, module)
                patched += 1

    return patched


# =============================================================================
# 1. SSL Encoder abstraction
# =============================================================================

class AbsSSLEncoder(nn.Module, abc.ABC):
    @property
    @abc.abstractmethod
    def ssl_dim(self) -> int: ...

    @property
    @abc.abstractmethod
    def num_layers(self) -> int: ...

    @property
    @abc.abstractmethod
    def total_layers(self) -> int: ...

    @abc.abstractmethod
    def forward(
        self,
        ssl_inputs: Dict[str, torch.Tensor],
        use_multilayer: bool = False,
    ) -> Tuple[torch.Tensor, OrderedDict]: ...

    @torch.no_grad()
    @abc.abstractmethod
    def extract_clean_features(
        self,
        ssl_inputs: Dict[str, torch.Tensor],
        use_multilayer: bool = False,
    ): ...


def _layer_range(total: int, target: int, mode: str) -> Tuple[int, int]:
    """Return (lo, hi) half-open layer index range.

    "low" : [0, target)        - acoustic layers 1..target
    "up"  : [target-1, total)  - semantic layers target..end
    "all" : [0, total)         - all layers
    """
    if mode == "low":
        return 0, target
    elif mode == "up":
        return max(0, target - 1), total
    elif mode == "all":
        return 0, total
    else:
        raise ValueError(f"multilayer_mode must be low/up/all, got {mode!r}")


def _decide_n_load(
    use_multilayer_loss: bool,
    multilayer_mode: str,
    target_layer: int,
) -> Optional[int]:
    """Decide how many HF transformer layers to load.

    Single mode (use_multilayer_loss=False) or multilayer "low":
      Load only target_layer layers. 
      (Wav2Vec2BertModel(num_hidden_layers=8) for target_layer=8).
    Multilayer "up" or "all":
      Need layers beyond target_layer -> load full model (return None).
    """
    if not use_multilayer_loss or multilayer_mode == "low":
        return target_layer
    return None


# ---------------------------------------------------------------------------
# w2v-BERT 2.0
# ---------------------------------------------------------------------------

class W2vBert2Encoder(AbsSSLEncoder):
    """w2v-BERT 2.0 teacher/student with LoRA and SDPA patch.

    use_multilayer_loss controls how many layers are loaded:
      False or mode="low" -> num_hidden_layers=target_layer (memory efficient)
      True  and mode in ("up","all") -> full 24-layer model
    """

    def __init__(
        self,
        model_tag: str = "facebook/w2v-bert-2.0",
        target_layer: int = 8,
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        input_sr: int = 16000,
        freeze_base: bool = True,
        use_flash_attention: bool = True,
        use_bf16: bool = False,
        multilayer_mode: str = "low",
        use_multilayer_loss: bool = False,
    ):
        super().__init__()
        from transformers import Wav2Vec2BertModel
        from peft import LoraConfig, inject_adapter_in_model

        self._target_layer    = target_layer
        self._multilayer_mode = multilayer_mode
        self.input_sr         = input_sr

        n_load = _decide_n_load(use_multilayer_loss, multilayer_mode, target_layer)

        hf_kwargs = dict(layerdrop=0.0, attn_implementation="eager")
        if n_load is not None:
            hf_kwargs["num_hidden_layers"] = n_load
        if use_bf16:
            hf_kwargs["torch_dtype"] = torch.bfloat16

        self.teacher = Wav2Vec2BertModel.from_pretrained(model_tag, **hf_kwargs)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.student = Wav2Vec2BertModel.from_pretrained(model_tag, **hf_kwargs)
        if freeze_base:
            for p in self.student.parameters():
                p.requires_grad = False

        total = len(self.student.encoder.layers)
        self._total_layers = total
        lo, hi = _layer_range(total, target_layer, multilayer_mode)
        self._lo, self._hi = lo, hi

        self.student = inject_adapter_in_model(
            LoraConfig(lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                       r=lora_rank, bias="lora_only",
                       target_modules=["output_dense"]),
            self.student,
        )
        for i, layer in enumerate(self.student.encoder.layers):
            if not (lo <= i < hi):
                for p in layer.parameters():
                    p.requires_grad = False

        if use_flash_attention:
            n_t = _patch_wav2vec2bert_sdpa(self.teacher)
            n_s = _patch_wav2vec2bert_sdpa(self.student)
            if n_t + n_s > 0:
                logger.info(
                    "W2vBert2Encoder: SDPA patch applied (%d teacher + %d student layers).",
                    n_t, n_s,
                )
            else:
                logger.warning("W2vBert2Encoder: SDPA patch returned 0 layers.")
        else:
            logger.info("W2vBert2Encoder: using eager attention (use_flash_attention=False)")

        trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        logger.info(
            "W2vBert2Encoder: loaded=%d layers (n_load=%s), use_multilayer_loss=%s, "
            "mode=%s, LoRA on [%d:%d], trainable=%.2fM",
            total, n_load, use_multilayer_loss, multilayer_mode, lo, hi, trainable / 1e6,
        )
        self._ssl_dim    = self.student.config.hidden_size
        self._num_layers = hi - lo

    @property
    def ssl_dim(self) -> int: return self._ssl_dim
    @property
    def num_layers(self) -> int: return self._num_layers
    @property
    def total_layers(self) -> int: return self._total_layers

    def _run(self, model, ssl_inputs, use_multilayer=False):
        out = model(**ssl_inputs, output_hidden_states=True)
        if use_multilayer:
            return list(out.hidden_states[1:])[self._lo:self._hi]
        return out.hidden_states[self._target_layer]

    def forward(self, ssl_inputs, use_multilayer=False):
        result = self._run(self.student, ssl_inputs, use_multilayer)
        if use_multilayer:
            return result[-1], OrderedDict(pred_ssl_feat=result[-1], all_feats=result)
        return result, OrderedDict(pred_ssl_feat=result)

    @torch.no_grad()
    def extract_clean_features(self, ssl_inputs, use_multilayer=False):
        return self._run(self.teacher, ssl_inputs, use_multilayer)


# ---------------------------------------------------------------------------
# XEUS
# ---------------------------------------------------------------------------

class XeusEncoder(AbsSSLEncoder):
    """XEUS (ESPnet E-Branchformer) teacher/student with LoRA and optional flash attention.

    NOTE: XEUS always loads all encoder layers because inference_encode() runs a full
    forward pass internally. use_multilayer_loss is accepted for API consistency with
    W2vBert2Encoder but does NOT reduce the number of layers loaded.
    """

    def __init__(
        self,
        model_tag: str = "xeus_checkpoint",
        target_layer: int = 10,
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        input_sr: int = 16000,
        freeze_base: bool = True,
        use_flash_attention: bool = False,
        multilayer_mode: str = "all",
        use_multilayer_loss: bool = False,   # accepted for API consistency; no effect on loading
    ):
        super().__init__()
        self._target_layer    = target_layer
        self._multilayer_mode = multilayer_mode
        self.input_sr         = input_sr

        from espnet2.tasks.ssl import SSLTask
        import os, yaml, tempfile

        ckpt_path   = os.path.join(model_tag, "model", "xeus_checkpoint_new.pth")
        config_path = os.path.join(model_tag, "model", "config.yaml")
        assert os.path.exists(ckpt_path),   f"XEUS checkpoint not found: {ckpt_path}"
        assert os.path.exists(config_path), f"XEUS config not found: {config_path}"

        actual_config_path = config_path
        _tmp_config = None
        if use_flash_attention:
            try:
                import flash_attn
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                enc_conf = cfg.get("encoder_conf", {})
                enc_conf["use_flash_attn"] = True
                cfg["encoder_conf"] = enc_conf
                tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
                yaml.dump(cfg, tmp)
                tmp.close()
                actual_config_path = _tmp_config = tmp.name
                logger.info(
                    "XeusEncoder: flash_attn %s enabled (encoder_conf.use_flash_attn=True)",
                    flash_attn.__version__,
                )
            except ImportError:
                logger.warning(
                    "XeusEncoder: use_flash_attention=True but flash_attn not installed; "
                    "falling back to standard attention."
                )

        xeus_t, _ = SSLTask.build_model_from_file(actual_config_path, ckpt_path, "cpu")
        xeus_s, _ = SSLTask.build_model_from_file(actual_config_path, ckpt_path, "cpu")

        if _tmp_config:
            try: os.unlink(_tmp_config)
            except OSError: pass

        self.teacher = xeus_t
        self.student = xeus_s

        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        if freeze_base:
            for p in self.student.parameters():
                p.requires_grad = False

        total = len(self.student.encoder.encoders)
        self._total_layers = total
        if not use_multilayer_loss:
            lo, hi = 0, target_layer
        else:
            lo, hi = _layer_range(total, target_layer, multilayer_mode)
        self._lo, self._hi = lo, hi

        try:
            from peft import LoraConfig, inject_adapter_in_model
            adapter_config = LoraConfig(
                lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                r=lora_rank, bias="lora_only", target_modules=["w_2"],
            )
            for i, layer in enumerate(self.student.encoder.encoders):
                if lo <= i < hi:
                    inject_adapter_in_model(adapter_config, layer)
                else:
                    for p in layer.parameters():
                        p.requires_grad = False
            trainable = sum(p.numel() for p in self.student.parameters()
                            if p.requires_grad)
            logger.info(
                "XeusEncoder: total=%d layers (always fully loaded), "
                "use_multilayer_loss=%s, mode=%s, LoRA on [%d:%d], trainable=%.2fM",
                total, use_multilayer_loss, multilayer_mode, lo, hi, trainable / 1e6,
            )
        except ImportError:
            logger.warning("peft not available; XeusEncoder has no LoRA")

        self._ssl_dim    = self.student.encoder.output_size()  # 1024 
        self._num_layers = hi - lo

    @property
    def ssl_dim(self) -> int: return self._ssl_dim
    @property
    def num_layers(self) -> int: return self._num_layers
    @property
    def total_layers(self) -> int: return self._total_layers

    # def _run(self, model, ssl_inputs, use_multilayer=False):
    #     wav   = ssl_inputs["waveform"].to(torch.bfloat16).contiguous()
    #     ilens = ssl_inputs["ilens"]
    #     model = model.to(wav.device).to(torch.bfloat16)  # cast model weights too
    #     _, layer_list, _ = model.inference_encode(
    #         wav, ilens, use_mask=False, use_final_output=True,
    #     )
    #     if use_multilayer:
    #         return layer_list[self._lo:self._hi]
    #     return layer_list[self._target_layer - 1]
    
    def _run(self, model, ssl_inputs, use_multilayer=False):
        wav   = ssl_inputs["waveform"].float().contiguous()
        ilens = ssl_inputs["ilens"]
        model = model.to(wav.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, layer_list, _ = model.inference_encode(
                wav, ilens, use_mask=False, use_final_output=True,
            )
        if use_multilayer:
            return layer_list[self._lo:self._hi]
        print(type(layer_list), len(layer_list), layer_list[0].shape) 
        return layer_list[self._target_layer - 1]


    def forward(self, ssl_inputs, use_multilayer=False):
        result = self._run(self.student, ssl_inputs, use_multilayer)
        if use_multilayer:
            return result[-1], OrderedDict(pred_ssl_feat=result[-1], all_feats=result)
        return result, OrderedDict(pred_ssl_feat=result)

    @torch.no_grad()
    def extract_clean_features(self, ssl_inputs, use_multilayer=False):
        return self._run(self.teacher, ssl_inputs, use_multilayer)


# ---------------------------------------------------------------------------
# WavLM
# ---------------------------------------------------------------------------

class WavLMEncoder(AbsSSLEncoder):
    """WavLM teacher/student with LoRA.

    use_multilayer_loss controls how many layers are loaded (same logic as W2vBert2Encoder).
    """

    def __init__(
        self,
        model_tag: str = "microsoft/wavlm-large",
        target_layer: int = 6,
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        input_sr: int = 16000,
        freeze_base: bool = True,
        use_flash_attention: bool = True,
        multilayer_mode: str = "low",
        use_multilayer_loss: bool = False,
    ):
        super().__init__()
        from transformers import WavLMModel
        self._target_layer    = target_layer
        self._multilayer_mode = multilayer_mode
        self.input_sr         = input_sr

        n_load = _decide_n_load(use_multilayer_loss, multilayer_mode, target_layer)
        hf_kwargs = {}
        if n_load is not None:
            hf_kwargs["num_hidden_layers"] = n_load

        self.teacher = WavLMModel.from_pretrained(model_tag, **hf_kwargs)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.student = WavLMModel.from_pretrained(model_tag, **hf_kwargs)
        if freeze_base:
            for p in self.student.parameters():
                p.requires_grad = False

        total = len(self.student.encoder.layers)
        self._total_layers = total
        lo, hi = _layer_range(total, target_layer, multilayer_mode)
        self._lo, self._hi = lo, hi

        try:
            from peft import LoraConfig, inject_adapter_in_model
            self.student = inject_adapter_in_model(
                LoraConfig(lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           r=lora_rank, bias="lora_only",
                           target_modules=["output_dense"]),
                self.student,
            )
            for i, layer in enumerate(self.student.encoder.layers):
                if not (lo <= i < hi):
                    for p in layer.parameters():
                        p.requires_grad = False
            trainable = sum(p.numel() for p in self.student.parameters()
                            if p.requires_grad)
            logger.info(
                "WavLMEncoder: loaded=%d layers (n_load=%s), use_multilayer_loss=%s, "
                "mode=%s, LoRA on [%d:%d], trainable=%.2fM",
                total, n_load, use_multilayer_loss, multilayer_mode, lo, hi, trainable / 1e6,
            )
        except ImportError:
            logger.warning("peft not available; WavLMEncoder has no LoRA")

        self._ssl_dim    = self.student.config.hidden_size
        self._num_layers = hi - lo

    @property
    def ssl_dim(self) -> int: return self._ssl_dim
    @property
    def num_layers(self) -> int: return self._num_layers
    @property
    def total_layers(self) -> int: return self._total_layers

    def _run(self, model, ssl_inputs, use_multilayer=False):
        out = model(**ssl_inputs, output_hidden_states=True)
        if use_multilayer:
            return list(out.hidden_states[1:])[self._lo:self._hi]
        return out.hidden_states[self._target_layer]

    def forward(self, ssl_inputs, use_multilayer=False):
        result = self._run(self.student, ssl_inputs, use_multilayer)
        if use_multilayer:
            return result[-1], OrderedDict(pred_ssl_feat=result[-1], all_feats=result)
        return result, OrderedDict(pred_ssl_feat=result)

    @torch.no_grad()
    def extract_clean_features(self, ssl_inputs, use_multilayer=False):
        return self._run(self.teacher, ssl_inputs, use_multilayer)


# =============================================================================
# 2. Vocoder
# =============================================================================

class Snake(nn.Module):
    """Snake activation: x + sin^2(alpha*x) / alpha  (DAC / BigVGAN style)."""
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        a = self.alpha.abs() + 1e-8
        return x + torch.sin(a * x) ** 2 / a


class _ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1  = nn.ModuleList()
        self.convs2  = nn.ModuleList()
        self.snakes1 = nn.ModuleList()
        self.snakes2 = nn.ModuleList()
        for d in dilation:
            p = d * (kernel_size - 1) // 2
            self.convs1.append(weight_norm(
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=p)))
            self.convs2.append(weight_norm(
                nn.Conv1d(channels, channels, kernel_size,
                          dilation=1, padding=(kernel_size - 1) // 2)))
            self.snakes1.append(Snake(channels))
            self.snakes2.append(Snake(channels))

    def forward(self, x):
        for s1, c1, s2, c2 in zip(self.snakes1, self.convs1, self.snakes2, self.convs2):
            x = x + c2(s2(c1(s1(x))))
        return x

    def remove_weight_norm(self):
        for c in (*self.convs1, *self.convs2):
            remove_weight_norm(c)


class SpeechCleanerVocoder(nn.Module):
    """HiFi-GAN + Snake: SSL features -> 48 kHz waveform.
    Upsampling: [8, 5, 4, 3, 2] -> 960x (50 Hz SSL -> 48 kHz).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        upsample_initial_channels: int = 512,
        upsample_rates: List[int] = [8, 5, 4, 3, 2],
        upsample_kernel_sizes: List[int] = [16, 10, 8, 6, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()
        self.num_kernels   = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        ch = upsample_initial_channels
        self.conv_pre  = weight_norm(nn.Conv1d(input_dim, ch, 7, padding=3))
        self.ups       = nn.ModuleList()
        self.snakes_up = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for u, k in zip(upsample_rates, upsample_kernel_sizes):
            self.snakes_up.append(Snake(ch))
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(ch, ch // 2, k, u, padding=(k - u) // 2)))
            ch //= 2
            for kr, dr in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(_ResBlock(ch, kr, tuple(dr)))
        self.snake_post = Snake(ch)
        self.conv_post  = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

    def generate(self, ssl_feat: torch.Tensor) -> torch.Tensor:
        """[B, T_ssl, D] -> [B, T_wav]"""
        x = ssl_feat.transpose(1, 2)
        x = self.conv_pre(x)
        n = self.num_kernels
        for i, (snake, up) in enumerate(zip(self.snakes_up, self.ups)):
            x = up(snake(x))
            xs = None
            for j in range(n):
                out = self.resblocks[i * n + j](x)
                xs  = out if xs is None else xs + out
            x = xs / n
        return torch.tanh(self.conv_post(self.snake_post(x))).squeeze(1)

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for up in self.ups: remove_weight_norm(up)
        for rb in self.resblocks: rb.remove_weight_norm()



class SpeechCleanerSSLFeatureLoss(nn.Module):
    """Single-layer MSE loss (Sidon original)."""
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        T = min(pred.shape[1], target.shape[1])
        return F.mse_loss(pred[:, :T], target[:, :T], reduction=self.reduction)


class SpeechCleanerMultiLayerSSLLoss(nn.Module):
    """Uniform per-layer MSE for FP training.

    loss = (1/N) * sum_i MSE(h_i_noisy, h_i_clean)
   
    """
    def __init__(self, num_layers: int, reduction: str = "mean"):
        super().__init__()
        self.num_layers = num_layers
        self.reduction  = reduction

    def forward(
        self,
        pred_list: List[torch.Tensor],
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(pred_list) == len(target_list) == self.num_layers
        per_layer = []
        loss = pred_list[0].new_zeros(())
        for p, t in zip(pred_list, target_list):
            T   = min(p.shape[1], t.shape[1])
            l_i = F.mse_loss(p[:, :T], t[:, :T], reduction=self.reduction)
            loss = loss + l_i
            per_layer.append(l_i.detach())
        return loss / self.num_layers, torch.stack(per_layer)


class SpeechCleanerMelLoss(nn.Module):

    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self._sr = sample_rate
        if _DAC_AVAILABLE:
            self._stft_loss = MultiScaleSTFTLoss(window_lengths=[2048, 512])
            self._mel_loss  = MelSpectrogramLoss(
                n_mels=[5, 10, 20, 40, 80, 160, 320],
                window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
                mel_fmin=[0]*7, mel_fmax=[None]*7,
                pow=1.0, clamp_eps=1e-5, mag_weight=0.0,
            )
            self._wav_loss = DACWavLoss()
        else:
            self._fallback_mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=2048, hop_length=480,
                win_length=2048, n_mels=128, power=1.0,
                norm="slaney", mel_scale="slaney",
            )

    def _to_signal(self, wav):
        if wav.dim() == 2: wav = wav.unsqueeze(1)
        return AudioSignal(wav, self._sr)

    def forward(self, pred_wav, target_wav, return_components: bool = False):
        T = min(pred_wav.shape[-1], target_wav.shape[-1])
        pred_wav, target_wav = pred_wav[..., :T], target_wav[..., :T]
        if _DAC_AVAILABLE:
            ps, ts = self._to_signal(pred_wav), self._to_signal(target_wav)
            mel_loss  = self._mel_loss(ps, ts)
            if return_components:
                # Computed only for logging/diagnostics -- NOT in the
                # returned loss, matching Sidon's actual training code.
                stft_loss = self._stft_loss(ps, ts)
                wav_loss  = self._wav_loss(ps, ts)
                return mel_loss, dict(
                    mel_loss=mel_loss.detach(),
                    stft_loss=stft_loss.detach(),
                    wav_loss=wav_loss.detach(),
                )
            return mel_loss
        fb = self._fallback_mel.to(pred_wav.device)
        pm = torch.log(fb(pred_wav).clamp(1e-5))
        tm = torch.log(fb(target_wav).clamp(1e-5))
        Tm = min(pm.shape[-1], tm.shape[-1])
        return F.l1_loss(pm[..., :Tm], tm[..., :Tm])
        
class SpeechCleanerGeneratorLoss(nn.Module):
    """Matches Sidon's GANLoss.generator_loss: plain SUM over discriminator
    sub-outputs, NOT averaged by count.
    """
    def forward(self, fake_outs):
        loss_g = fake_outs[0].new_zeros(()) if fake_outs else 0.0
        for fo in fake_outs:
            loss_g = loss_g + torch.mean((fo - 1.0) ** 2)
        return loss_g


class SpeechCleanerDiscriminatorLoss(nn.Module):
    """Matches Sidon's GANLoss.discriminator_loss: plain SUM, not averaged."""
    def forward(self, real_outs, fake_outs):
        loss_d = real_outs[0].new_zeros(()) if real_outs else 0.0
        for ro, fo in zip(real_outs, fake_outs):
            loss_d = loss_d + torch.mean((ro - 1.0) ** 2) + torch.mean(fo ** 2)
        return loss_d


class SpeechCleanerFeatureMatchLoss(nn.Module):
    """Matches Sidon's feature_loss accumulation: plain SUM over all
    discriminator layers/feature maps, not averaged by count.
    """
    def __init__(self, weight: float = 2.0):
        super().__init__()
        self.weight = weight

    def forward(self, real_fmaps, fake_fmaps):
        loss = None
        for r_fmap, f_fmap in zip(real_fmaps, fake_fmaps):
            for rf, ff in zip(r_fmap, f_fmap):
                term = F.l1_loss(ff, rf.detach())
                loss = term if loss is None else loss + term
        if loss is None:
            loss = torch.zeros((), device=real_fmaps[0][0].device if real_fmaps else "cpu")
        return self.weight * loss




# =============================================================================
# 4. Stage 1 - Feature Predictor
# =============================================================================

class SpeechCleanerFPModel(AbsESPnetModel):
    """Feature Predictor (Stage 1).

    use_multilayer_loss=False: single-layer MSE (Sidon original)
    use_multilayer_loss=True:  uniform per-layer MSE over all selected layers
    """

    @typechecked
    def __init__(
        self,
        ssl_encoder: AbsSSLEncoder,
        use_multilayer_loss: bool = False,
    ):
        super().__init__()
        self.ssl_encoder         = ssl_encoder
        self.use_multilayer_loss = use_multilayer_loss

        if use_multilayer_loss:
            self.loss_fn = SpeechCleanerMultiLayerSSLLoss(num_layers=ssl_encoder.num_layers)
            logger.info("FPModel: multi-layer loss over %d layers", ssl_encoder.num_layers)
        else:
            self.loss_fn = SpeechCleanerSSLFeatureLoss()
            logger.info("FPModel: single-layer loss (layer %d)", ssl_encoder._target_layer)

    def forward(
        self,
        noisy_speech, noisy_speech_lengths,
        speech_ref1,  speech_ref1_lengths,
        noisy_speech_ssl=None, speech_ref1_ssl=None,
        **kwargs,
    ):
        if noisy_speech_ssl is None or speech_ref1_ssl is None:
            raise ValueError("ssl_inputs must be provided via collate_fn")

        B      = noisy_speech.shape[0]
        device = noisy_speech.device
        noisy_ssl = {k: v.to(device) for k, v in noisy_speech_ssl.items()}
        clean_ssl = {k: v.to(device) for k, v in speech_ref1_ssl.items()}

        if self.use_multilayer_loss:
            _, others   = self.ssl_encoder(noisy_ssl, use_multilayer=True)
            pred_list   = others["all_feats"]
            target_list = self.ssl_encoder.extract_clean_features(clean_ssl, use_multilayer=True)
            loss, per_layer = self.loss_fn(pred_list, target_list)
            stats = dict(loss=loss.detach(), loss_ssl=loss.detach())
            for i, l_i in enumerate(per_layer):
                stats[f"loss_layer{i+1}"] = l_i
        else:
            pred_feat, _ = self.ssl_encoder(noisy_ssl, use_multilayer=False)
            target_feat  = self.ssl_encoder.extract_clean_features(clean_ssl, use_multilayer=False)
            loss  = self.loss_fn(pred_feat, target_feat)
            stats = dict(loss=loss.detach(), loss_ssl=loss.detach())

        loss, stats, weight = force_gatherable((loss, stats, B), loss.device)
        return loss, stats, weight

    def collect_feats(self, **kwargs): return {}


# =============================================================================
# 5. Stage 2/3 - Vocoder GAN
# =============================================================================

class SpeechCleanerVocoderModel(AbsGANESPnetModel):
    """Vocoder GAN (Stages 2/3) — uses DAC Discriminator matching Sidon exactly.
 
    Loss interface (matches Sidon's GANLoss exactly):
      d_out = discriminator(wav)   # List[List[Tensor]]
        outer list: one per sub-discriminator
        inner list: [fmap0, fmap1, ..., logits]  (logits = last element)
      loss_D = Σ_sub [mean(fake_logits^2) + mean((1-real_logits)^2)]
      loss_G = Σ_sub [mean((1-fake_logits)^2)]
      loss_fm = Σ_sub Σ_layer L1(fake_fmap, real_fmap.detach())
    """
    
    @typechecked
    def __init__(
        self,
        ssl_encoder: AbsSSLEncoder,
        vocoder: nn.Module,
        use_predicted_feat: bool = False,
        use_multilayer_feat: bool = False,
        layer_weighting: Optional[str] = None,   # NEW
        mel_loss_weight: float = 15.0,
        adv_loss_weight: float = 2.0,
        fm_loss_weight: float = 1.0,
        sample_rate: int = 48000,
    ):
        super().__init__()
        self.ssl_encoder         = ssl_encoder
        self.vocoder             = vocoder
        self.use_predicted_feat  = use_predicted_feat
        self.use_multilayer_feat = use_multilayer_feat
        self.layer_weighting     = layer_weighting          # NEW
        self.mel_w   = mel_loss_weight
        self.adv_w   = adv_loss_weight
        self.fm_w    = fm_loss_weight
        self._sr     = sample_rate

        import dac
        self.discriminator = dac.model.discriminator.Discriminator(sample_rate=sample_rate)
        self.mel_loss = SpeechCleanerMelLoss(sample_rate=sample_rate)
        self._dnsmos_sess       = None
        self._dnsmos_polyfit_fn = None

        # NEW: layer_weights / layer_router 구성을 layer_weighting에 따라 분기
        self.layer_weights = None   # global_learnable
        self.layer_router  = None   # utterance_dynamic / frame_dynamic

        if use_multilayer_feat:
            N = ssl_encoder.num_layers
            D = ssl_encoder.ssl_dim
            if layer_weighting == "uniform":
                pass  # 
            elif layer_weighting == "global_learnable":
                self.layer_weights = nn.Parameter(torch.zeros(N))
            elif layer_weighting in ("utterance_dynamic", "frame_dynamic"):
                self.layer_router = nn.Linear(D, 1)
            else:
                raise ValueError(
                    f"use_multilayer_feat=True requires layer_weighting in "
                    f"['uniform','global_learnable','utterance_dynamic','frame_dynamic'], "
                    f"got {layer_weighting!r}"
                )
            logger.info(
                "VocoderModel: multilayer feat aggregation strategy=%s over %d layers",
                layer_weighting, N,
            )
    # ── Sidon-exact GAN loss helpers ──────────────────────────────────────
 
    def _discriminator_loss(
        self,
        fake_wav: torch.Tensor,
        real_wav: torch.Tensor,
    ) -> torch.Tensor:
        """Matches Sidon's GANLoss.discriminator_loss exactly."""
        d_fake = self.discriminator(fake_wav.detach().unsqueeze(1))
        d_real = self.discriminator(real_wav.unsqueeze(1))
        loss_d = fake_wav.new_zeros(())
        for out_fake, out_real in zip(d_fake, d_real):
            loss_d = loss_d + torch.mean(out_fake[-1] ** 2)
            loss_d = loss_d + torch.mean((1 - out_real[-1]) ** 2)
        return loss_d
 
    def _generator_loss(
        self,
        fake_wav: torch.Tensor,
        real_wav: torch.Tensor,
    ):
        """Matches Sidon's GANLoss.generator_loss exactly.
        Returns (loss_g, feature_loss) both as plain summations.
        """
        d_fake = self.discriminator(fake_wav.unsqueeze(1))
        d_real = self.discriminator(real_wav.detach().unsqueeze(1))
 
        loss_g = fake_wav.new_zeros(())
        for out_fake in d_fake:
            loss_g = loss_g + torch.mean((1 - out_fake[-1]) ** 2)
 
        feature_loss = fake_wav.new_zeros(())
        for out_fake, out_real in zip(d_fake, d_real):
            # feature maps = all elements except the last (logits)
            for fmap_fake, fmap_real in zip(out_fake[:-1], out_real[:-1]):
                feature_loss = feature_loss + F.l1_loss(fmap_fake, fmap_real.detach())
 
        return loss_g, feature_loss
 
    # ── SSL feature extraction ────────────────────────────────────────────
    def _weighted_sum(self, feat_list: List[torch.Tensor]) -> torch.Tensor:
        T = min(f.shape[1] for f in feat_list)
        feats = torch.stack([f[:, :T] for f in feat_list], dim=2)  # [B, T, N, D]
        N = feats.shape[2]

        if self.layer_weighting == "uniform":
            weights = feats.new_full((N,), 1.0 / N).view(1, 1, N)              # [1,1,N]

        elif self.layer_weighting == "global_learnable":
            weights = torch.softmax(self.layer_weights, dim=0).view(1, 1, N)   # [1,1,N]

        elif self.layer_weighting == "utterance_dynamic":
            pooled = feats.mean(dim=1)                          # [B, N, D]  (T 축 평균으로 제거)
            scores = self.layer_router(pooled).squeeze(-1)       # [B, N]
            weights = torch.softmax(scores, dim=-1).unsqueeze(1) # [B, 1, N] -> T축에 broadcast

        elif self.layer_weighting == "frame_dynamic":
            scores = self.layer_router(feats).squeeze(-1)        # [B, T, N]
            weights = torch.softmax(scores, dim=-1)               # [B, T, N]

        else:
            raise ValueError(self.layer_weighting)

        # 진단용: 배치/시간 평균 weight을 저장 (stats 로깅에 재사용)
        with torch.no_grad():
            w = weights.detach()
            w_flat = w.reshape(-1, w.shape[-1]) if w.dim() == 3 else w.view(1, -1)  # [B*T or B, N]
            self._last_layer_weights     = w_flat.mean(dim=0)   # [N] - 배치/시간 평균
            self._last_layer_weights_std = w_flat.std(dim=0)    # [N] - 배치/시간 표준편차 (아래 5번 참고)
            entropy = -(w_flat * (w_flat + 1e-8).log()).sum(dim=-1)  # [B*T or B]
            self._last_router_entropy = entropy.mean()

        out = (weights.unsqueeze(-1) * feats).sum(dim=2)    # [B, T, D]
        return out
 
    def _get_ssl_feat(self, noisy_ssl, clean_ssl) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if self.use_multilayer_feat:
                if self.use_predicted_feat:
                    with torch.no_grad():
                        _, others = self.ssl_encoder(noisy_ssl, use_multilayer=True)
                        feat_list = others["all_feats"]
                else:
                    feat_list = self.ssl_encoder.extract_clean_features(
                        clean_ssl, use_multilayer=True)
                feat = self._weighted_sum(feat_list)
            else:
                if self.use_predicted_feat:
                    with torch.no_grad():
                        feat, _ = self.ssl_encoder(noisy_ssl, use_multilayer=False)
                else:
                    feat = self.ssl_encoder.extract_clean_features(
                        clean_ssl, use_multilayer=False)
        return feat.float().contiguous()
 
    @staticmethod
    def _trim(a, b):
        T = min(a.shape[-1], b.shape[-1])
        return a[..., :T], b[..., :T]
 
    # ── DNSMOS (validation only) ──────────────────────────────────────────
 
    def _load_score_v2(self):
        path = os.path.join(os.getcwd(), "local", "score_v2.py")
        spec = _ilu.spec_from_file_location("score_v2", path)
        mod  = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
 
    def _compute_dnsmos_ovrl(self, wav_48k: torch.Tensor) -> torch.Tensor:
        if self._dnsmos_sess is None:
            import onnxruntime as ort
            score_v2 = self._load_score_v2()
            onnx_path = score_v2._download_dnsmos_onnx(".dnsmos_cache")
            self._dnsmos_sess       = ort.InferenceSession(onnx_path)
            self._dnsmos_polyfit_fn = score_v2._dnsmos_polyfit
 
        N_SAMPLES = 144160
        wav_16k = AF.resample(wav_48k.float().cpu(), 48000, 16000)
        scores = []
        for i in range(wav_16k.shape[0]):
            w = wav_16k[i].numpy()
            if len(w) < N_SAMPLES:
                if len(w) < 1:
                    w = np.zeros(N_SAMPLES, dtype=np.float32)  # 극단적 안전장치
                else:
                    reps = int(np.ceil(N_SAMPLES / len(w)))
                    w = np.tile(w, reps)
            w = w[:N_SAMPLES].astype(np.float32)
            try:
                out = self._dnsmos_sess.run(None, {"input_1": w[np.newaxis, :]})[0][0]
                _, _, ovr = self._dnsmos_polyfit_fn(*out)
                scores.append(float(np.clip(ovr, 1.0, 5.0)))
            except Exception as e:
                logger.warning("DNSMOS scoring failed: %s", e)
        mean_score = float(np.mean(scores)) if scores else 3.0
        return torch.tensor(mean_score, device=wav_48k.device, dtype=torch.float32) 
    # ── STOI (validation only, fast ~50ms per batch) ──────────────────────
 
    def _compute_stoi(
        self,
        fake_wav: torch.Tensor,
        real_wav: torch.Tensor,
    ) -> torch.Tensor:
        """Standard STOI at 16kHz using clean reference.
 
        Works at 16kHz (resampled from 48kHz). Much faster than DNSMOS.
        Uses pystoi if available, otherwise returns 0 silently.
        """
        try:
            from pystoi import stoi as _stoi
        except ImportError:
            logger.warning_once("pystoi not installed; STOI metric skipped. "
                                "pip install pystoi")
            return torch.zeros((), device=fake_wav.device)
 
        # Resample 48k -> 16k for STOI computation
        fake_16k = AF.resample(fake_wav.float().cpu(), 48000, 16000)
        real_16k = AF.resample(real_wav.float().cpu(), 48000, 16000)
 
        scores = []
        for i in range(fake_16k.shape[0]):
            f = fake_16k[i].squeeze().numpy()
            r = real_16k[i].squeeze().numpy()
            T = min(len(f), len(r))
            if T < 256:
                continue
            try:
                s = _stoi(r[:T], f[:T], 16000, extended=False)
                scores.append(float(s))
            except Exception as e:
                logger.debug("STOI failed on one sample: %s", e)
 
        mean_stoi = float(np.mean(scores)) if scores else 0.0
        return torch.tensor(mean_stoi, device=fake_wav.device, dtype=torch.float32)
 
    # ── Forward ───────────────────────────────────────────────────────────
 
    def forward(
        self,
        noisy_speech, noisy_speech_lengths,
        speech_ref1,  speech_ref1_lengths,
        noisy_speech_ssl=None, speech_ref1_ssl=None,
        forward_generator: bool = True,
        **kwargs,
    ):
        device    = noisy_speech.device
        noisy_ssl = {k: v.to(device) for k, v in noisy_speech_ssl.items()} \
            if noisy_speech_ssl else None
        clean_ssl = {k: v.to(device) for k, v in speech_ref1_ssl.items()} \
            if speech_ref1_ssl else None
        if forward_generator:
            return self._forward_G(noisy_ssl, clean_ssl, speech_ref1)
        return self._forward_D(noisy_ssl, clean_ssl, speech_ref1)
 
    def _forward_G(self, noisy_ssl, clean_ssl, speech_ref1):
        ssl_feat           = self._get_ssl_feat(noisy_ssl, clean_ssl)
        fake_wav           = self.vocoder.generate(ssl_feat)
        real_wav, fake_wav = self._trim(speech_ref1, fake_wav)
 
        # Mel reconstruction loss (Sidon: mel_loss only, weight=15)
        l_mel = self.mel_loss(fake_wav, real_wav)
 
        # Adversarial + feature matching (Sidon GANLoss.generator_loss)
        l_adv, l_fm = self._generator_loss(fake_wav, real_wav)
 
        loss_G = self.mel_w * l_mel + self.adv_w * l_adv + self.fm_w * l_fm
 
        stats = dict(
            loss_G=loss_G.detach(),
            loss_mel=l_mel.detach(),
            loss_adv_G=l_adv.detach(),
            loss_fm=l_fm.detach(),
        )
 
        # ── Validation-only metrics ──────────────────────────────────────
        if not self.training:
            with torch.no_grad():
                # DNSMOS (slow, ~500ms/batch — keep for checkpoint selection)
                dnsmos_score = self._compute_dnsmos_ovrl(fake_wav.detach())
                stats["dnsmos_ovrl"] = dnsmos_score
 
                # STOI (fast, ~50ms/batch — reliable intelligibility proxy)
                stoi_score = self._compute_stoi(fake_wav.detach(), real_wav.detach())
                stats["stoi"] = stoi_score
        else:
            stats["dnsmos_ovrl"] = loss_G.new_zeros(())
            stats["stoi"]        = loss_G.new_zeros(())
        # ────────────────────────────────────────────────────────────────
 
        # if self.use_multilayer_feat and getattr(self, "_last_layer_weights", None) is not None:
        #     for i, wi in enumerate(self._last_layer_weights):
        #         stats[f"ssl_weight_layer{i+1}"] = wi
        if self.use_multilayer_feat:
            if getattr(self, "_last_layer_weights", None) is not None:
                for i, wi in enumerate(self._last_layer_weights):
                    stats[f"ssl_weight_layer{i+1}"] = wi
            if getattr(self, "_last_layer_weights_std", None) is not None:
                for i, si in enumerate(self._last_layer_weights_std):
                    stats[f"ssl_weight_layer{i+1}_std"] = si
            if getattr(self, "_last_router_entropy", None) is not None:
                import math
                stats["router_entropy"] = self._last_router_entropy
                stats["router_entropy_max"] = torch.tensor(
                    math.log(len(self._last_layer_weights)), device=self._last_router_entropy.device)

        B = speech_ref1.shape[0]
        loss_G, stats_G, weight = force_gatherable((loss_G, stats, B), loss_G.device)
        return dict(loss=loss_G, stats=stats_G, weight=weight, optim_idx=0)
 
    def _forward_D(self, noisy_ssl, clean_ssl, speech_ref1):
        with torch.no_grad():
            ssl_feat = self._get_ssl_feat(noisy_ssl, clean_ssl)
            fake_wav = self.vocoder.generate(ssl_feat)
 
        real_wav, fake_wav = self._trim(speech_ref1, fake_wav)
 
        # Sidon GANLoss.discriminator_loss
        loss_D = self._discriminator_loss(fake_wav, real_wav)
 
        B = speech_ref1.shape[0]
        loss_D, stats_D, weight = force_gatherable(
            (loss_D, dict(loss_D=loss_D.detach()), B), loss_D.device)
        return dict(loss=loss_D, stats=stats_D, weight=weight, optim_idx=1)
 
    def collect_feats(self, **kwargs): return {}
 
    def train(self, mode: bool = True):
        """Keep ssl_encoder always in eval mode."""
        super().train(mode)
        self.ssl_encoder.eval()
        return self
 