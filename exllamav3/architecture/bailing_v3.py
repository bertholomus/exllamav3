from __future__ import annotations
from typing_extensions import override
import atexit
import os
import re
import torch
import torch.nn.functional as F

from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..util.tensor import get_for_device, to2
from ..modules import (
    RMSNorm, Embedding, TransformerBlock, MLAttention, GatedMLP, Linear,
    BlockSparseMLP, GatedDeltaNet, GDNState,
)
from ..modules.gated_rmsnorm import GatedRMSNorm
from ..modules.gated_delta_net import _proj_dtype, _gate_dtype
from ..modules.gated_delta_net_fn import causal_conv1d_update, gated_delta_rule_fn
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence


# BailingMoeV3 (Ling-3.1-flash), model_type "bailing_hybrid".
#
# 72 decoder layers: 63 KDA (GatedDeltaNet, full-rank gates) and 9 gated-MLA at
# layers 7,15,...,71; dense MLP 0..3, grouped-sigmoid MoE 4..71 (512 experts + 1
# shared); untied bf16 head, vocab 157,184. MTP layer 72 is a DeepSeek-V3-shaped
# nextn layer (eh_proj/enorm/hnorm input layer + one MLA+MoE block + its own
# final_layernorm) sharing the trunk's embedding and lm_head.
#
# Exactness items, named below by their labels:
#   R1  kv_b_proj is never a budgeted Linear: MLAttention reads it raw (w_uk_flat /
#       w_uv_flat) and carries it unquantized through conversion. The MTP layer's
#       attention.g_proj is the only gate coming from an FP8 source tensor (dequanted
#       to bf16 in the BF16 input checkpoint); both stay out of the -rcp recipe and -mb.
#   R4  KDA with full-rank f_proj/g_proj (no_kda_lora), per-channel decay
#       g = lower_bound * sigmoid(exp(A_log) * (f + dt_bias)) with lower_bound -5.0,
#       beta = sigmoid(fp32 b) * beta_scale kept fp32 into the chunked KDA kernel
#       (no bf16 cache; the recurrent decode kernel's bf16 beta contract is honored
#       at the kernel boundary only -- see BailingGatedDeltaNet.forward).
#   R5  MLA with the head-wise g_proj gate applied per head before the dense
#       projection, GPT-J interleaved rope on the 64-dim slice at theta 6e6, softmax
#       scale 192^-0.5 over q_nope+q_rope, kv_a_layernorm on the 512 latent only.
#   R6  Grouped router: fp32 logits from fp32-upcast inputs (TF32 asserted off),
#       group masking with -inf (not 0.0), unbiased sigmoid weights + 1e-20, x 2.5.
#   R7  SwiGLU clamps via act_limit (min(silu(g), L) * clip(u, -L, L) in the
#       kernels): routed experts layers 65-71, shared experts 64-71, never layer 72.
#   Also an optional tokens-per-expert-per-layer counter (EXL3_LING_EXPERT_COUNTER=1),
#       logging min / p5 / median per layer at process exit.
#
# Not wired here (loud failures by design):
#   - TP export/import for the two subclasses (the base classes reconstruct plain
#     GatedDeltaNet / MLAttention, which would silently drop the full-rank gates
#     and the head gate); tp_export raises on both until that is written.
#   - BailingMoeV3MTPModel.attach_to (draft-time reuse of the trunk embedding and
#     lm_head) -- not needed for conversion, which loads and quantizes the MTP
#     side model module by module.
#
# Reference: inclusionAI's modeling_bailing_moe_v3.py (Ling-3.1-flash, revision
# 7b864daa; Apache License 2.0, Copyright 2025 Antgroup and The HuggingFace Inc.
# team) and an exact fp32 forward written from it.

# Decoder layers that use gated MLA instead of KDA. The checkpoint config carries
# no layer_types key; this window is an architectural constant of BailingMoeV3.
MLA_LAYERS = frozenset({7, 15, 23, 31, 39, 47, 55, 63, 71})
# MTP layer index: MLA-structured, carries eh_proj/enorm/hnorm/final_layernorm and
# the full MoE; never clamped (R7 -- neither limit list covers it).
MTP_LAYER = 72
FIRST_MOE_LAYER = 4  # config first_k_dense_replace


def _swiglu_limit(lst: list | None, idx: int) -> float:
    """Per-layer swiglu clamp; absent entries (including the MTP layer) are 0 = off."""
    return lst[idx] if lst and idx < len(lst) else 0.0


# ---------------------------------------------------------------------------
# Router (R6)
# ---------------------------------------------------------------------------

def _tf32_problems() -> list[str]:
    bad = []
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0") not in ("", "0"):
        bad.append("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE is set")
    if torch.backends.cuda.matmul.allow_tf32:
        bad.append("torch.backends.cuda.matmul.allow_tf32")
    if torch.backends.cudnn.allow_tf32:
        bad.append("torch.backends.cudnn.allow_tf32")
    if torch.get_float32_matmul_precision() != "highest":
        bad.append("float32_matmul_precision != highest")
    return bad


def _assert_no_tf32():
    """TF32 off in every reference and calibration process (router logits, calibration Hessians)."""
    bad = _tf32_problems()
    if bad:
        raise RuntimeError("bailing_v3: TF32 is on (" + "; ".join(bad) + "). The exact fp32 router needs it off.")


def set_tf32_off():
    """Called from BailingMoeV3Config.__init__. The env override is read when cuBLAS starts, so it is only
    checked: launch the process without it."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    _assert_no_tf32()


def routing_bailing_ds3(bsz, cfg, y, params):
    """
    Ling's grouped sigmoid router, on the routing_ds3 contract (same RoutingCFG), with
    the two R6 exactness fixes over block_sparse_mlp_routing.routing_ds3:

      - logits in fp32 from fp32-upcast inputs, TF32 off: the reference and the oracle
        both score the router in fp32 (router_dtype fp32).
      - group masking fills -inf instead of 0.0: scores_for_choice is sigmoid(logits)
        + expert_bias, which can be <= 0, and a 0.0-masked slot would then beat a real
        unmasked pick. -inf keeps every masked expert strictly below every unmasked one.

    Weights stay the unbiased sigmoid scores (the bias only steers selection),
    renormalized with a 1e-20 floor and scaled by routed_scaling_factor.
    """
    _assert_no_tf32()
    activate_all_experts = params.get("activate_all_experts")
    router_logits = torch.matmul(y.float(), cfg.gate_tensor.float())

    scores = router_logits.sigmoid()
    scores_for_choice = scores.view(-1, cfg.num_experts)
    if cfg.e_score_correction_bias is not None:
        scores_for_choice = scores_for_choice + cfg.e_score_correction_bias.unsqueeze(0)

    group_scores = (
        scores_for_choice.view(-1, cfg.n_group, cfg.num_experts // cfg.n_group)
        .topk(2, dim = -1)[0]
        .sum(dim = -1)
    )
    group_idx = torch.topk(group_scores, k = cfg.topk_group, dim = -1, sorted = False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, cfg.n_group, cfg.num_experts // cfg.n_group)
        .reshape(-1, cfg.num_experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

    topk_indices = torch.topk(
        scores_for_choice,
        k = cfg.num_experts if activate_all_experts else cfg.num_experts_per_tok,
        dim = -1,
        sorted = False
    )[1]
    topk_weights = scores.gather(1, topk_indices)
    denominator = topk_weights.sum(dim = -1, keepdim = True) + 1e-20
    topk_weights /= denominator
    topk_weights = topk_weights * cfg.routed_scaling_factor
    return topk_indices, topk_weights


# ---------------------------------------------------------------------------
# Tokens-per-expert counter (EXL3_LING_EXPERT_COUNTER=1)
# ---------------------------------------------------------------------------

_EXPERT_COUNTER_ENABLED = os.environ.get("EXL3_LING_EXPERT_COUNTER", "0") != "0"
_EXPERT_COUNTS: dict[str, torch.Tensor] = {}


def report_expert_counters():
    """Log the tokens-per-expert distribution per layer: min / 5th percentile / median
    over the layer's experts. Accumulated on real routing only (not the all-experts
    calibration capture, not tp_warmup random routing)."""
    if not _EXPERT_COUNTS:
        print(" -- expert counter: no routed layers recorded")
        return
    print(" -- tokens-per-expert per layer (min / p5 / median over experts):")

    def layer_ord(key: str):
        m = re.search(r"layers\.(\d+)\.", key)
        return int(m.group(1)) if m else -1

    for key in sorted(_EXPERT_COUNTS, key = layer_ord):
        c = _EXPERT_COUNTS[key].to(torch.float64).cpu()
        print(f"     {key:44s} min {c.min().item():8.0f}   "
              f"p5 {torch.quantile(c, 0.05).item():10.1f}   median {c.median().item():10.1f}")


if _EXPERT_COUNTER_ENABLED:
    atexit.register(report_expert_counters)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class BailingMoeV3Config(Config):
    arch_string = "BailingMoeV3ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        set_tf32_off()
        super().__init__(
            directory,
            {"text": BailingMoeV3Model, "mtp": BailingMoeV3MTPModel},
            **kwargs
        )

        # Gated MLA (layers 7,15,...,71)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.q_lora_rank = self.read_cfg(int, "q_lora_rank", no_default)
        self.kv_lora_rank = self.read_cfg(int, "kv_lora_rank", no_default)
        self.qk_nope_head_dim = self.read_cfg(int, "qk_nope_head_dim", no_default)
        self.qk_rope_head_dim = self.read_cfg(int, "qk_rope_head_dim", no_default)
        self.v_head_dim = self.read_cfg(int, "v_head_dim", no_default)
        assert self.read_cfg(int, "qk_head_dim", 0) == self.qk_nope_head_dim + self.qk_rope_head_dim, \
            "config qk_head_dim does not match qk_nope_head_dim + qk_rope_head_dim"
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # Reported head dim, for allocation and logging; the cache stores the latent instead
        self.head_dim = self.qk_head_dim
        # R5: the softmax scale follows the unabsorbed head dim (192 here); no YaRN
        # mscale -- the checkpoint has rope_scaling null
        self.sm_scale = self.qk_head_dim ** -0.5
        assert self.read_cfg(dict, "rope_scaling", None) is None, \
            "BailingMoeV3 ships rope_scaling null; extensional rope is not supported"

        # RoPE: GPT-J interleaved (rope_interleave true), theta 6e6, on the 64-dim rope
        # slice only. The reference pins full rotation of that slice
        # (modeling_bailing_moe_v3.py sets partial_rotary_factor = 1.0); the
        # checkpoint's rotary_dim 64 is what the kernels read, so require it here
        # rather than silently honoring partial_rotary_factor 0.5
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.GPTJ,
            override_head_dim = self.qk_rope_head_dim,
        )
        assert self.rope_settings.rotary_dim == self.qk_rope_head_dim, \
            f"BailingMoeV3 rotates the full {self.qk_rope_head_dim}-dim rope slice; " \
            f"config rotary_dim is {self.rope_settings.rotary_dim}"

        # Layer schedule: KDA everywhere except the fixed MLA window
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.layer_types = [
            "full_attention" if idx in MLA_LAYERS else "linear_attention"
            for idx in range(self.num_hidden_layers)
        ]

        # KDA linear attention. Ling ships no_kda_lora (full-rank f_proj/g_proj); the
        # low-rank f_a/f_b, g_a/g_b bottleneck path does not apply (R4)
        self.linear_num_heads = self.num_q_heads
        self.linear_head_dim = self.read_cfg(int, "head_dim", no_default)
        self.linear_conv_kernel_size = self.read_cfg(int, "short_conv_kernel_size", no_default)
        self.linear_lower_bound = self.read_cfg(float, "kda_lower_bound", no_default)
        self.assert_cfg(bool, "kda_safe_gate", True, True)
        self.assert_cfg(bool, "no_kda_lora", True, False)
        self.assert_cfg(bool, "use_kda_lora", False, True)
        self.assert_cfg(bool, "mtp_use_kda", False, True)
        # in-kernel q/k l2norm (eps 1e-6, q * head_dim^-0.5) matches the reference
        self.assert_cfg(bool, "use_qk_norm", True, True)

        # MLP / MoE
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", no_default)
        self.num_shared_experts = self.read_cfg(int, "num_shared_experts", no_default)
        self.num_experts = self.read_cfg(int, "num_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", no_default)
        self.routed_scaling_factor = self.read_cfg(float, "routed_scaling_factor", no_default)
        self.first_k_dense_replace = self.read_cfg(int, "first_k_dense_replace", no_default)
        self.mlp_layer_types = [
            "dense" if idx < self.first_k_dense_replace else "sparse"
            for idx in range(self.num_hidden_layers)
        ]
        self.n_group = self.read_cfg(int, "n_group", no_default)
        self.topk_group = self.read_cfg(int, "topk_group", no_default)
        self.assert_cfg(str, "score_function", "sigmoid", True)
        self.assert_cfg(str, "topk_method", "noaux_tc", True)
        self.assert_cfg(bool, "norm_topk_prob", True, True)
        self.assert_cfg(bool, "moe_router_enable_expert_bias", True, True)
        # R6: the router scores in exact fp32
        self.assert_cfg(str, "router_dtype", "fp32", True)
        self.assert_cfg(bool, "scale_router_input", False, True)
        self.assert_cfg(bool, "up_proj_norm", False, True)
        shared_inter = self.read_cfg(int, "moe_shared_expert_intermediate_size", None)
        if shared_inter is not None:
            assert shared_inter == self.moe_intermediate_size * self.num_shared_experts, \
                "moe_shared_expert_intermediate_size does not match the shared-expert " \
                "width the modules below are built with"

        # R7: per-layer swiglu clamps. The checkpoint's config: routed experts clamped on
        # layers 65-71, shared experts on 64-71, and neither list covers the MTP layer
        self.swiglu_limit_list = self.read_cfg(list, "expert_swiglu_limit_list", None)
        self.shared_swiglu_limit_list = self.read_cfg(
            list, "share_expert_swiglu_limit_list", None)

        # Norms / head
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        assert not self.tie_word_embeddings, \
            "BailingMoeV3 stores an untied lm_head; tied embeddings are not supported"
        self.assert_cfg(str, "gated_attention_proj_granularity_type", "head_wise", True)
        self.num_mtp_layers = self.read_cfg(int, "num_nextn_predict_layers", 0)


# ---------------------------------------------------------------------------
# KDA linear attention, full-rank gates (R4)
# ---------------------------------------------------------------------------

class BailingGatedDeltaNet(GatedDeltaNet):
    """
    Ling KDA layer at exllamav3's GatedDeltaNet tensor contract, with the full-rank
    f_proj/g_proj the no_kda_lora checkpoint stores (exllamav3's KDA mode hard-codes the
    rank-128 f_a/f_b, g_a/g_b chain).

    Per layer (key prefix <layer>.attention):
      q/k/v_proj [8192,4096] -> fused qkv_proj Linear (alt_key), g_proj [8192,4096]
      (per-channel output gate, held by the base's z_proj slot), f_proj [8192,4096]
      (per-channel decay gate input), o_proj [4096,8192], b_proj [64,4096] bf16,
      A_log [64] f32, dt_bias [8192] f32, q/k/v_conv1d [8192,1,4], o_norm [128].

    Forward is exllamav3's KDA torch path with three changes: the f and g gates are
    single full-rank projections; beta stays fp32 into the chunked KDA kernel (the
    base casts it to bf16, R4); and the fused CUDA-graph decode path is never taken
    (it is written against the low-rank gate chain).
    """

    def __init__(
        self,
        config: BailingMoeV3Config,
        key: str,
        layer_idx: int,
        hidden_size: int,
        k_head_dim: int,
        v_head_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        rms_norm_eps: float,
        conv_kernel_size: int,
        gate_lower_bound: float,
        qmap: str | None,
        out_dtype: torch.dtype | None = None,
        select_hq_bits: int = 0,
    ):
        # The base picks the norm's gate activation from its own KDA flag (key_f_a),
        # which stays None here, so prebuild the o_norm with the sigmoid gate the KDA
        # reference uses
        norm = GatedRMSNorm(
            config, f"{key}.o_norm", rms_norm_eps, out_dtype = torch.half,
            gate_activation = "sigmoid",
        )
        super().__init__(
            config, key, layer_idx, hidden_size, k_head_dim, v_head_dim,
            num_k_heads, num_v_heads, rms_norm_eps, conv_kernel_size,
            key_a_log = "A_log",
            key_dt_bias = "dt_bias",
            key_conv1d = "conv1d",
            key_conv1d_q = "q_conv1d",
            key_conv1d_k = "k_conv1d",
            key_conv1d_v = "v_conv1d",
            key_qkv = "qkv_proj",
            key_qkv_alt = ["q_proj", "k_proj", "v_proj"],
            # The base builds the gated-norm gate as z_proj when key_z is set; key_z
            # "g_proj" gives that Linear exactly Ling's checkpoint name and shape, and
            # satisfies the base's split-projection assert outside its low-rank KDA mode
            key_z = "g_proj",
            key_o = "o_proj",
            gate_lower_bound = gate_lower_bound,
            norm = norm,
            qmap = qmap,
            out_dtype = out_dtype,
            select_hq_bits = select_hq_bits,
        )
        self.module_name = "BailingGatedDeltaNet"

        # Full-rank decay gate (f), budgeted (recipe class "kda"); the output gate is
        # the base's z_proj above (checkpoint name ...attention.g_proj). b_proj is the
        # fp32 beta logit projection (unquantized)
        self.kda = True
        self.f_proj = Linear(
            config, f"{key}.f_proj", hidden_size, self.k_dim,
            qmap = qmap + ".input", out_dtype = torch.float,
            select_hq_bits = select_hq_bits, qgroup = key + ".fg",
        )
        self.register_submodule(self.f_proj)

        self.b_proj = Linear(
            config, f"{key}.b_proj", hidden_size, self.num_v_heads,
            qmap = None, out_dtype = torch.float, pad_to = 1,
        )
        self.register_submodule(self.b_proj)

    @override
    def optimizer_targets(self):
        # qkv, f, g and o all read the same block input, so they share one calibration
        # group; the base would drop f (and its own list has no Ling g slot)
        return [[
            self.qkv_proj.optimizer_targets(),
            self.f_proj.optimizer_targets(),
            self.z_proj.optimizer_targets(),
            self.o_proj.optimizer_targets(),
        ]]

    @override
    def tp_export(self, plan, producer):
        # The base reconstructs plain GatedDeltaNet on the worker and rebuilds the gates
        # from key_f_a/key_f_b/key_g_a/key_g_b -- all None here. Exporting would hand the
        # worker a gateless KDA layer; fail loudly until the subclass round-trip exists
        raise NotImplementedError(
            "BailingGatedDeltaNet: no TP export yet -- the worker would rebuild a "
            "gateless plain GatedDeltaNet. Carry the full-rank f_proj/g_proj (and this "
            "class) through tp_export/tp_import before TP placement."
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        save_history = params.get("recurrent_history", False)

        # Post load, fuse conv1d weights if needed (same deferred fill as the base)
        if self.conv1d_weight is None:
            self.conv1d_weight = torch.cat([
                self.conv1d_q_weight,
                self.conv1d_k_weight,
                self.conv1d_v_weight,
            ], dim = 0)
            self.conv1d_q_weight = None
            self.conv1d_k_weight = None
            self.conv1d_v_weight = None
        if self.conv1d_weight_flat is None:
            self.conv1d_weight_flat = self.conv1d_weight.squeeze(1).contiguous()

        # Previous state (same contract as the base forward)
        rsg = params.get("recurrent_states")
        if rsg:
            recurrent_slots = get_for_device(params, "recurrent_slots", self.device)
            layer_instance = (self.layer_idx, params.get("layer_instance", 0))
            if rsg[0].exported:
                rsl = self.tp_recurrent_lookup[rsg[0].cache]
            else:
                rsl = rsg[0].cache.get_recurrent_layer(layer_instance)
            conv_state, recurrent_state = rsl.get_state_tensors()
            save_state = True
        else:
            recurrent_slots = None
            conv_state, recurrent_state = None, None
            save_state = False
            save_history = False  # no SD without prior state, for simplicity

        # Torch path only: the fused C++/graph decode path is written against the
        # rank-128 f_a/f_b, g_a/g_b chain this module does not have
        qkv = self.qkv_proj.forward(x, params, out_dtype = torch.float)  # the reference rounds the projection to bf16 once; fp16 first would land 1 ulp off
        mixed_qkv = qkv.transpose(1, 2).to(torch.bfloat16).contiguous()
        del qkv

        # Full-rank per-channel output gate for the gated norm (the base's z, held by
        # z_proj under the checkpoint's ...attention.g_proj name; the gated norm takes a
        # bf16 or fp32 gate, so this stays fp32)
        z = self.z_proj.forward(x, params) \
            .view(bsz, seqlen, self.num_v_heads, self.v_head_dim)

        # R4: beta = sigmoid(fp32 b) * beta_scale, kept fp32 -- the base casts to bf16
        # here, which rounds the beta the delta rule consumes
        beta = torch.sigmoid(
            self.b_proj.forward(x, params, out_dtype = _gate_dtype).float() * self.beta_scale
        )

        # Per-k-channel log decay from the full-rank decay gate: "safe gate" form with
        # the configured lower bound (-5.0), else softplus (per the Kimi/GLM reference)
        f = self.f_proj.forward(x, params, out_dtype = _gate_dtype)
        gf = f.float() if f.dtype != torch.float else f.clone()
        del f
        gf = gf.add_(self.dt_bias.float().view(1, 1, -1)) \
            .view(bsz, seqlen, self.num_v_heads, self.k_head_dim)
        decay = torch.exp(self.a_log.float()).view(1, 1, self.num_v_heads, 1)
        if self.gate_lower_bound is not None:
            g = torch.sigmoid(gf.mul_(decay)).mul_(self.gate_lower_bound)
        else:
            g = F.softplus(gf, threshold = 20.0).mul_(-decay)
        del gf

        # Short conv (SiLU) and the delta rule
        mixed_qkv = causal_conv1d_update(
            mixed_qkv = mixed_qkv,
            conv_state = conv_state,
            recurrent_slots = recurrent_slots,
            conv1d_weight = self.conv1d_weight_flat,
            conv1d_bias = self.conv1d_bias,
            history = save_history,
            params = params,
            token_major = False,
        )

        # The chunked KDA kernel (prefill, the calibration and parity path) takes beta
        # as-is; the recurrent decode kernel's contract is bf16 beta
        # (gdn.cu TORCH_CHECK_DTYPE(beta, kBFloat16)), so the single rounding happens
        # exactly there and nowhere earlier. An fp32-beta kernel variant would remove it
        decode_beta = beta if seqlen >= self.num_v_heads and not save_history \
            else beta.to(torch.bfloat16)
        core_attn_out = gated_delta_rule_fn(
            mixed_qkv = mixed_qkv,
            beta = decode_beta,
            g = g,
            recurrent_state = recurrent_state,
            recurrent_slots = recurrent_slots,
            history = save_history,
            save_state = save_state,
            num_k_heads = self.num_k_heads,
            num_v_heads = self.num_v_heads,
            k_dim = self.k_dim,
            v_dim = self.v_dim,
            k_head_dim = self.k_head_dim,
            v_head_dim = self.v_head_dim,
            params = params,
            channelwise_g = self.kda,
        )
        del mixed_qkv, decode_beta, beta, g

        # Gated output norm: rms_norm(y) * sigmoid(g_proj(x)), per channel over DV
        core_attn_out = self.norm.forward(core_attn_out, params, gate = z)
        core_attn_out = core_attn_out.view(bsz, seqlen, self.num_v_heads * self.v_head_dim)
        del z

        # Output projection
        x = self.o_proj.forward(core_attn_out, params)

        # TP reduction
        if self.tp_reduce:
            self.tp_collect(params["backend"], x)

        return to2(x, out_dtype, self.out_dtype)


# ---------------------------------------------------------------------------
# Gated MLA with the head-wise output gate (R5)
# ---------------------------------------------------------------------------

class BailingMLAttention(MLAttention):
    """
    Fork MLAttention at Ling's tensor names (o_proj key "dense", kv_b raw) plus the
    head-wise output gate: g_proj [num_q_heads, hidden] bf16, sigmoid in fp32, applied
    to each head's output slice before the dense projection. The gate is registered as
    an unquantized submodule; the o_proj attribute is wrapped so every eager path
    (mha-form, dense, sparse -- all reshape to (bsz, seqlen, H*v) first) is gated.
    """

    def __init__(
        self,
        *args,
        gate_out_features: int = 0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.gate_proj = None
        self._gate_x = None
        self._o_proj_linear = self.o_proj
        if gate_out_features and self.num_q_heads:
            pre = (kwargs.get("submodules") or {}).get("gate_proj")
            self.gate_proj = pre if pre is not None else Linear(
                self.config, f"{self.key}.g_proj",
                self.hidden_size, gate_out_features,
                qmap = None, out_dtype = torch.half, pad_to = 1,
            )
            self.register_submodule(self.gate_proj)
            owner = self

            class _GatedOProj:
                """Stands in for the o_proj attribute on the attention module. The real
                Linear stays registered in the module tree (load, calibration and
                quantization walk that), and every other attribute access delegates."""

                def forward(self, o, params):
                    return owner._gated_o_proj(o, params)

                def __getattr__(self, name):
                    return getattr(owner._o_proj_linear, name)

            self.o_proj = _GatedOProj()

    def _gated_o_proj(self, o: torch.Tensor, params: dict) -> torch.Tensor:
        # R5: o_h *= bf16(sigmoid(fp32(g_proj(x)))) per head before the dense projection;
        # x is the attention block input stashed by forward()
        bsz, seqlen, _ = o.shape
        g = self.gate_proj.forward(self._gate_x, params)  # (bsz, seqlen, H), fp16
        gate = torch.sigmoid(g.float()).to(torch.bfloat16).to(o.dtype)
        o = (o.view(bsz, seqlen, self.num_q_heads, self.v_head_dim)
             * gate.unsqueeze(-1)).reshape(bsz, seqlen, -1)
        return self._o_proj_linear.forward(o, params)

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        if self.gate_proj is not None:
            self._gate_x = x
            try:
                return super().forward(x, params, out_dtype)
            finally:
                self._gate_x = None
        return super().forward(x, params, out_dtype)

    @override
    def bc_mla_step(self, *args, **kwargs):
        # The graph-captured decode block has no gate input path; return None so the
        # dispatch falls back to the eager path and the gate stays exact (R5)
        if self.gate_proj is not None:
            return None
        return super().bc_mla_step(*args, **kwargs)

    @override
    def make_tp_allocation(self, options: dict) -> list:
        if self.gate_proj is not None:
            raise NotImplementedError(
                "BailingMLAttention: no TP plan yet -- the gate must be carried through "
                "tp_export/tp_import (the base reconstructs a gateless plain "
                "MLAttention)."
            )
        return super().make_tp_allocation(options)

    @override
    def tp_export(self, plan, producer):
        if self.gate_proj is not None:
            raise NotImplementedError(
                "BailingMLAttention: no TP export yet -- the base would reconstruct a "
                "gateless plain MLAttention on the worker."
            )
        return super().tp_export(plan, producer)


# ---------------------------------------------------------------------------
# Block-sparse MoE with Ling's router (R6) and the expert counter
# ---------------------------------------------------------------------------

class BailingBlockSparseMLP(BlockSparseMLP):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.router_type == "ds3":
            # routing_ds3 masks with 0.0 and scores in whatever dtype the input carries;
            # routing_bailing_ds3 is the exact variant (R6)
            self.routing_fn = self._routing_bailing_ds3
        if _EXPERT_COUNTER_ENABLED and self.routing_gate is not None:
            self._wrap_expert_counter()

    def _routing_bailing_ds3(self, bsz, cfg, y, params):
        selected_experts, routing_weights = routing_bailing_ds3(bsz, cfg, y, params)
        # The quantized expert paths take the weights in half, the dtype every upstream router
        # returns: exl3_moe reads them as half with no dtype check, exl3_moe_gather requires half
        # (our first run stopped there at layer 4), and receiving TP ranks allocate half.
        # The unquantized forward (calibration capture, parity checks) keeps the fp32 weights
        if self.is_quantized:
            routing_weights = routing_weights.half()
        return selected_experts, routing_weights

    def _wrap_expert_counter(self):
        real_fn = self.routing_fn
        layer_key = self.key
        num_experts = self.num_experts

        def counted(bsz, cfg, y, params):
            selected_experts, routing_weights = real_fn(bsz, cfg, y, params)
            # Real routing only: the all-experts calibration capture and tp_warmup's
            # random selections say nothing about rare-expert coverage
            if not params.get("activate_all_experts") and not params.get("tp_warmup"):
                counts = torch.bincount(
                    selected_experts.reshape(-1), minlength = num_experts
                ).to(torch.long)
                acc = _EXPERT_COUNTS.get(layer_key)
                if acc is None or acc.device != counts.device:
                    acc = torch.zeros(num_experts, dtype = torch.long, device = counts.device)
                    _EXPERT_COUNTS[layer_key] = acc
                acc += counts
            return selected_experts, routing_weights

        self.routing_fn = counted


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BailingMoeV3Model(Model):
    config_class = BailingMoeV3Config

    def __init__(
        self,
        config: BailingMoeV3Config,
        key_prefix: str = "model",
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = f"{key_prefix}.word_embeddings",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            key = f"{key_prefix}.layers.{idx}"
            # R7: routed experts clamp on the expert list, shared experts on the shared
            # list; entries past the end (the MTP layer) are unclamped
            act_limit = _swiglu_limit(config.swiglu_limit_list, idx)
            shared_limit = _swiglu_limit(config.shared_swiglu_limit_list, idx)

            if config.layer_types[idx] == "linear_attention":
                attn = BailingGatedDeltaNet(
                    config = config,
                    key = f"{key}.attention",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    k_head_dim = config.linear_head_dim,
                    v_head_dim = config.linear_head_dim,
                    num_k_heads = config.linear_num_heads,
                    num_v_heads = config.linear_num_heads,
                    rms_norm_eps = config.rms_norm_eps,
                    conv_kernel_size = config.linear_conv_kernel_size,
                    gate_lower_bound = config.linear_lower_bound,
                    qmap = "block.attn",
                    out_dtype = torch.float,
                    select_hq_bits = 2,
                )
            else:
                attn = BailingMLAttention(
                    config = config,
                    key = f"{key}.attention",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    num_q_heads = config.num_q_heads,
                    kv_lora_rank = config.kv_lora_rank,
                    qk_nope_head_dim = config.qk_nope_head_dim,
                    qk_rope_head_dim = config.qk_rope_head_dim,
                    v_head_dim = config.v_head_dim,
                    rope_settings = config.rope_settings,
                    q_lora_rank = config.q_lora_rank,
                    sm_scale = config.sm_scale,
                    rms_norm_eps = config.rms_norm_eps,
                    qmap = "block.attn",
                    out_dtype = torch.float,
                    select_hq_bits = 2,
                    key_o = "dense",
                    gate_out_features = config.num_q_heads,
                )

            if config.mlp_layer_types[idx] == "dense":
                mlp = GatedMLP(
                    config = config,
                    key = f"{key}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = "up_proj",
                    key_gate = "gate_proj",
                    key_down = "down_proj",
                    activation_fn = "silu",
                    act_limit = _swiglu_limit(config.swiglu_limit_list, idx),
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    select_hq_bits = 1,
                )
            else:
                mlp = BailingBlockSparseMLP(
                    config = config,
                    key = f"{key}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.moe_intermediate_size,
                    num_experts = config.num_experts,
                    num_experts_per_tok = config.num_experts_per_tok,
                    key_up = "experts.{expert_idx}.up_proj",
                    key_gate = "experts.{expert_idx}.gate_proj",
                    key_down = "experts.{expert_idx}.down_proj",
                    key_routing_gate = "gate",
                    key_e_score_bias = "gate.expert_bias",
                    activation_fn = "silu",
                    act_limit = act_limit,
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    router_type = "ds3",
                    routed_scaling_factor = config.routed_scaling_factor,
                    n_group = config.n_group,
                    topk_group = config.topk_group,
                    qbits_key = "bits",
                    shared_experts = GatedMLP(
                        config = config,
                        key = f"{key}.mlp.shared_experts",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        activation_fn = "silu",
                        act_limit = shared_limit,
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        select_hq_bits = 2,
                    ) if config.num_shared_experts else None,
                )

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = attn,
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = mlp,
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        self.modules += [
            RMSNorm(
                config = config,
                key = f"{key_prefix}.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = "lm_head",
                qbits_key = "head_bits",
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True}
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        # Activate all experts during the H capture pass in quantization
        self.calibration_all_experts = True

        self.caps.update({
            "supports_tp": True,
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
            "linear_attn": True,
        })
        self.recurrent_state_cls = GDNState

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        prepare_for_recurrence(input_ids, params, self)
        return input_ids

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError(
            "Use the Ling chat template via the tokenizer, not default_chat_prompt"
        )


class BailingMoeV3MTPModel(Model):
    """
    MTP head: model.layers.72 as a DeepSeek-V3-shaped nextn layer (hnorm/enorm/eh_proj
    input layer + one gated-MLA + full-MoE block + the layer's own final_layernorm),
    sharing the trunk's embedding and lm_head. Mirrors glm5_next_mtp.py. The MTP
    component's quantized Linears are all qbits_key "mtp_bits" (convert.py -mb), and
    the layer's attention.g_proj is unquantized like the trunk gates (R1).
    """

    config_class = BailingMoeV3Config

    def __init__(
        self,
        config: BailingMoeV3Config,
        key_prefix: str = "model",
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        first_mtp_layer = config.num_hidden_layers  # 72
        # Ling ships exactly one nextn layer; the block's layer_idx is its index within
        # THIS component (the MTP cache is indexed by it), not the trunk layer number
        assert config.num_mtp_layers == 1, \
            f"BailingMoeV3MTPModel builds one nextn layer, config has {config.num_mtp_layers}"

        from ..modules.arch_specific.qwen3_5_mtp import Qwen3_5MTPInputLayer
        self.input_layer = Qwen3_5MTPInputLayer(
            config = config,
            key = f"{key_prefix}.layers.{first_mtp_layer}.input",
            key_pre_fc_norm_hidden = f"{key_prefix}.layers.{first_mtp_layer}.hnorm",
            key_pre_fc_norm_embedding = f"{key_prefix}.layers.{first_mtp_layer}.enorm",
            key_fc = f"{key_prefix}.layers.{first_mtp_layer}.eh_proj",
            hidden_size = config.hidden_size,
            rms_norm_eps = config.rms_norm_eps,
            native_draft_len = 1,
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
            constant_bias = 0.0,
        )
        self.modules = [self.input_layer]
        self.first_block_idx = len(self.modules)

        idx = first_mtp_layer
        key = f"{key_prefix}.layers.{idx}"
        # R7: neither clamp list covers the MTP layer -- both lookups fall past the end
        act_limit = _swiglu_limit(config.swiglu_limit_list, idx)
        shared_limit = _swiglu_limit(config.shared_swiglu_limit_list, idx)

        attn = BailingMLAttention(
            config = config,
            key = f"{key}.attention",
            layer_idx = 0,
            hidden_size = config.hidden_size,
            num_q_heads = config.num_q_heads,
            kv_lora_rank = config.kv_lora_rank,
            qk_nope_head_dim = config.qk_nope_head_dim,
            qk_rope_head_dim = config.qk_rope_head_dim,
            v_head_dim = config.v_head_dim,
            rope_settings = config.rope_settings,
            q_lora_rank = config.q_lora_rank,
            sm_scale = config.sm_scale,
            rms_norm_eps = config.rms_norm_eps,
            qmap = "block.attn",
            out_dtype = torch.float,
            select_hq_bits = 2,
            key_o = "dense",
            gate_out_features = config.num_q_heads,
            qbits_key = "mtp_bits",
        )
        mlp = BailingBlockSparseMLP(
            config = config,
            key = f"{key}.mlp",
            hidden_size = config.hidden_size,
            intermediate_size = config.moe_intermediate_size,
            num_experts = config.num_experts,
            num_experts_per_tok = config.num_experts_per_tok,
            key_up = "experts.{expert_idx}.up_proj",
            key_gate = "experts.{expert_idx}.gate_proj",
            key_down = "experts.{expert_idx}.down_proj",
            key_routing_gate = "gate",
            key_e_score_bias = "gate.expert_bias",
            activation_fn = "silu",
            act_limit = act_limit,
            qmap = "block.mlp",
            interm_dtype = torch.half,
            out_dtype = torch.float,
            router_type = "ds3",
            routed_scaling_factor = config.routed_scaling_factor,
            n_group = config.n_group,
            topk_group = config.topk_group,
            qbits_key = "mtp_bits",
            shared_experts = GatedMLP(
                config = config,
                key = f"{key}.mlp.shared_experts",
                hidden_size = config.hidden_size,
                intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                key_up = "up_proj",
                key_gate = "gate_proj",
                key_down = "down_proj",
                activation_fn = "silu",
                act_limit = shared_limit,
                qmap = "block.mlp",
                interm_dtype = torch.half,
                out_dtype = torch.float,
                qbits_key = "mtp_bits",
                select_hq_bits = 2,
            ) if config.num_shared_experts else None,
        )
        self.modules += [
            TransformerBlock(
                config = config,
                key = key,
                layer_idx = 0,
                attn_norm = RMSNorm(
                    config = config,
                    key = f"{key}.input_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                attn = attn,
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"{key}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                mlp = mlp,
            )
        ]

        self.last_kv_module_idx = len(self.modules) - 1

        # The draft shares the trunk's lm_head, so the pre-head norm is the layer's own
        # final_layernorm
        self.final_norm = RMSNorm(
            config = config,
            key = f"{key}.final_layernorm",
            rms_norm_eps = config.rms_norm_eps,
            out_dtype = torch.half,
        )
        self.modules.append(self.final_norm)

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": 3,
        })
        self.calibration_all_experts = True

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # The MTP input layer embeds; prepare_for_attn still wires the attention params
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")

    def attach_to(self, target):
        # Not written yet: bind the trunk embedding + lm_head like glm5_next_mtp.attach_to
        # (borrowed refs for the draft-time forward and sample_from_state). Conversion
        # does not need this: the MTP side model is quantized module by module.
        raise NotImplementedError(
            "BailingMoeV3MTPModel.attach_to is not written yet (draft-time reuse of the trunk "
            "embedding + lm_head)"
        )

    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long

    def default_load_params(self, max_chunk_size):
        return {}
