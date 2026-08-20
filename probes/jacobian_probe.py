"""
Inference-only Jacobian diagnostic probes for LongLive causal memory.

Four probes (all read-only; model weights never updated):
  A: J_c  = d KV_new / d prompt_embeds          (prompt sensitivity)
  B: J_x  = d KV_new / d context_latent          (visual sensitivity)
  C: J_o  = d y_t    / d historical KV            (output observability)
  D: J_m  = d KV_new / d historical KV            (memory persistence)

Architecture notes for LongLive / CausalWanModel:
  - Each transformer block runs: self-attention → cross-attention → FFN
  - New K/V written to self-attn cache = f(x at block input, before cross-attn)
  - So for block 0: new K/V has zero gradient to prompt (cross-attn runs AFTER)
  - For block L>0: new K/V does depend on prompt through cross-attn in blocks 0..L-1
  - Text conditioning is T5 embeddings → text_embedding MLP → cross-attn K/V
  - Flash attention is NOT available; SDPA (differentiable) is used for inference
  - Context-update step: model(denoised_pred, context_timestep≈0) writes new K/V to cache

Gradient flow for Probe A/B:
  - We reset crossattn_cache["is_init"]=False in probe clone so cross-attn recomputes
    K/V from prompt_probe, enabling gradient flow.  Values are numerically identical.
  - We hook CausalWanSelfAttention to capture new_k/new_v before cache commit.
  - VJPs computed from captured new_k/new_v back to leaf tensors.

Gradient flow for Probe C (J_o):
  - Monkey-patch target block's self_attn.forward to substitute k_hist/v_hist leaf
    tensors for the historical portion of the KV passed to SDPA.
  - Gradient flows: y_t → model head → block outputs → patched attention → k_hist/v_hist.

Cache invariance:
  - All probes use deep-cloned caches; original kv_cache is verified unchanged.
"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

from mpmath import j
import numpy as np
import torch
import torch.autograd as autograd

from wan.modules.causal_model import causal_rope_apply
from wan.modules.attention import attention as wan_attention

@dataclass
class ProbeRecord:
    global_chunk: int
    prompt_segment_id: int
    relative_to_switch: int
    denoising_timestep: int
    layer_id: int
    probe_seed: int

    gamma_c: float = field(default=0.0)
    gamma_x: float = field(default=0.0)
    control_ratio: float = field(default=0.0)

    e_c_per_proj: List[float] = field(default_factory=list)
    e_x_per_proj: List[float] = field(default_factory=list)
    entanglement_proxy: float = float("nan")

    jo_k_norm_by_slot: List[float] = field(default_factory=list)
    jo_v_norm_by_slot: List[float] = field(default_factory=list)
    jo_combined_by_slot: List[float] = field(default_factory=list)
    jm_combined_by_slot: List[float] = field(default_factory=list)


def _vjp_frob_sq(
    output_tensors: List[torch.Tensor],
    leaf_tensors: List[torch.Tensor],
    num_proj: int,
    gen: torch.Generator,
) -> Tuple[List[List[float]], List[float]]:
    n_leaves = len(leaf_tensors)
    per_proj: List[List[float]] = [[] for _ in range(n_leaves)]

    for m in range(num_proj):
        scalar = sum(
            (t.float() * torch.randn(t.shape, dtype=torch.float32, device=t.device, generator=gen)).sum()
            for t in output_tensors
        )
        retain = (m < num_proj - 1)
        grads = autograd.grad(
            scalar, leaf_tensors,
            retain_graph=retain,
            allow_unused=True,
        )
        for j, g in enumerate(grads):
            per_proj[j].append(float(g.float().norm() ** 2) if g is not None else 0.0)
        
    mean_sq = [float(np.mean(pp)) for pp in per_proj]
    return per_proj, mean_sq


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _snapshot_cache(kv_cache: List[dict]) -> List[dict]:
    return [{
        "k": blk["k"].detach().clone(),
        "v": blk["v"].detach().clone(),
        "global_end_index": blk["global_end_index"].detach().clone(),
        "local_end_index": blk["local_end_index"].detach().clone(),
    } for blk in kv_cache]


def _snapshot_crossattn(crossattn_cache: List[dict]) -> List[dict]:
    return [{
        "k": blk["k"].detach().clone(),
        "v": blk["v"].detach().clone(),
        "is_init": blk["is_init"],
    } for blk in crossattn_cache]


def _clone_cache(kv_cache: List[dict]) -> List[dict]:
    return [{
        "k": blk["k"].detach().clone(),
        "v": blk["v"].detach().clone(),
        "global_end_index": blk["global_end_index"].detach().clone(),
        "local_end_index": blk["local_end_index"].detach().clone(),
    } for blk in kv_cache]


def _clone_crossattn(crossattn_cache: List[dict], is_init_override: Optional[bool] = None) -> List[dict]:
    result = []
    for blk in crossattn_cache:
        is_init = blk["is_init"] if is_init_override is None else is_init_override
        result.append({
            "k": blk["k"].detach().clone(),
            "v": blk["v"].detach().clone(),
            "is_init": is_init,
        })
    return result


def _check_cache_unchanged(kv_cache: List[dict], snap: List[dict], label: str = "kv_cache", tol: float = 1e-5) -> bool:
    ok = True
    for i, (blk, s) in enumerate(zip(kv_cache, snap)):
        dk = float((blk["k"] - s["k"]).abs().max())
        dv = float((blk["v"] - s["v"]).abs().max())
        if dk > tol or dv > tol:
            print(f"[PROBE WARNING] {label}[{i}] modified! max_diff_k={dk:.2e} max_diff_v={dv:.2e}")
            ok = False
    return ok


def _check_crossattn_unchanged(crossattn_cache: List[dict], snap: List[dict],
                                label: str = "crossattn", tol: float = 1e-5) -> bool:
    ok = True
    for i, (blk, s) in enumerate(zip(crossattn_cache, snap)):
        dk = float((blk["k"] - s["k"]).abs().max())
        dv = float((blk["v"] - s["v"]).abs().max())
        if dk > tol or dv > tol:
            print(f"[PROBE WARNING] {label}[{i}] modified! max_diff_k={dk:.2e} max_diff_v={dv:.2e}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Main JacobianProbe class
# ---------------------------------------------------------------------------

class JacobianProbe:
    """Inference-time Jacobian probe for LongLive causal memory analysis.

    Args:
        enabled:          If False all methods are no-ops.
        layers:           Transformer layer indices to probe.
        num_layers_total: Total number of transformer blocks.
        num_proj:         Number of random VJP projections (M).
        probe_seed:       Seed for probe random vectors (isolated from generation RNG).
        output_dir:       Directory for saving results.
    """

    def __init__(
        self,
        enabled: bool = True,
        layers: Optional[List[int]] = None,
        num_layers_total: int = 30,
        num_proj: int = 2,
        probe_seed: int = 114514,
        output_dir: str = "outputs/probes/default",
    ):
        self.enabled = enabled
        self.num_layers_total = num_layers_total
        if layers is None:
            L = num_layers_total
            layers = sorted({0, L // 3, 2 * L // 3, L -1})
        self.probe_layers = layers
        self.num_proj = num_proj
        self.probe_seed = probe_seed
        self.output_dir = output_dir
        self._gen = torch.Generator(device="cuda")
        self._gen.manual_seed(probe_seed)
        self.records: List[ProbeRecord] = []
        if enabled:
            os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public: collect one probe point
    # ------------------------------------------------------------------
    def collect_probe_at_block(
        self,
        *,
        global_chunk: int,
        prompt_segment_id: int,
        relative_to_switch: str,
        denoising_timestep: int,
        generator_model,                # WanDiffusionWrapper
        context_latent: torch.Tensor,   # denoised_pred: [B, F, C, H, W]
        conditional_dict: dict,
        context_timestep: torch.Tensor,
        kv_cache: List[dict],
        crossattn_cache: List[dict],
        current_start: int,
        noisy_input: Optional[torch.Tensor] = None,
        denoising_ts: Optional[torch.Tensor] = None,
        do_probe_c: bool = False,
        do_probe_x: bool = False,
        do_probe_o: bool = True,
        do_probe_m: bool = False,
    ) -> None:
        if not self.enabled:
            return
        
        kv_snap = _snapshot_cache(kv_cache)
        ca_snap = _snapshot_crossattn(crossattn_cache)

        for layer_id in self.probe_layers:
            rec = ProbeRecord(
                global_chunk=global_chunk,
                prompt_segment_id=prompt_segment_id,
                relative_to_switch=relative_to_switch,
                denoising_timestep=denoising_timestep,
                layer_id=layer_id,
                probe_seed=self.probe_seed,
            )

            try:
                if do_probe_c or do_probe_x:
                    gc, gx, cr, ec, ex, ent = self._probe_ab(
                        generator_model=generator_model,
                        context_latent=context_latent,
                        conditional_dict=conditional_dict,
                        context_timestep=context_timestep,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start,
                        layer_id=layer_id,
                        do_c=do_probe_c,
                        do_x=do_probe_x,
                    )
                    rec.gamma_c = gc
                    rec.gamma_x = gx
                    rec.control_ratio = cr
                    rec.e_c_per_proj = ec
                    rec.e_x_per_proj = ex
                    rec.entanglement_proxy = ent
            except Exception as e:
                print(f"[Probe A/B] layer={layer_id} chunk={global_chunk} error: {e}")
            
            try:
                if do_probe_o and noisy_input is not None and denoising_ts is not None:
                    jk, jv, jc = self._probe_c(
                        generator_model=generator_model,
                        noisy_input=noisy_input,
                        conditional_dict=conditional_dict,
                        denoising_ts=denoising_ts,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start,
                        layer_id=layer_id,
                    )
                    rec.jo_k_norm_by_slot = jk
                    rec.jo_v_norm_by_slot = jv
                    rec.jo_combined_by_slot = jc
            except Exception as e:
                print(f"[Probe C] layer={layer_id} chunk={global_chunk} error: {e}")
            
            try:
                if do_probe_m:
                    jm = self._probe_d(
                        generator_model=generator_model,
                        context_latent=context_latent,
                        conditional_dict=conditional_dict,
                        context_timestep=context_timestep,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start,
                        layer_id=layer_id,
                    )
                    rec.jm_combined_by_slot = jm
            except Exception as e:
                print(f"[Probe D] layer={layer_id} chunk={global_chunk} error: {e}")

            self.records.append(rec)

        _check_cache_unchanged(kv_cache, kv_snap, label="kv_cache")
        _check_crossattn_unchanged(crossattn_cache, ca_snap, label="crossattn_cache")
    
    def _probe_ab(
        self, *, generator_model, context_latent, conditional_dict,
        context_timestep, kv_cache, crossattn_cache, current_start,
        layer_id, do_c, do_x,
    ):
        eps = 1e-8
        raw_embeds = conditional_dict["prompt_embeds"]

        with torch.inference_mode(False), torch.enable_grad():
            x_probe: Optional[torch.Tensor] = None
            prompt_probe: Optional[torch.Tensor] = None

            if do_x:
                x_probe = context_latent.detach().clone().float().requires_grad_(True)

            if do_c:
                stacked = torch.stack([e.detach().clone().float() for e in raw_embeds])
                prompt_probe = stacked.requires_grad_(True)
        
            # Capture new_k / new_v from self-attn hook
            new_kv: Dict[str, Optional[torch.Tensor]] = {"k": None, "v": None}
            target_sa = generator_model.model.blocks[layer_id].self_attn

            def _hook(module, input, output):
                if isinstance(output, tuple) and len(output) == 2:
                    _, info = output
                    if isinstance(info, tuple) and len(info) == 3:
                        _, _, upd = info
                        if upd is not None:
                            new_kv["k"] = upd.get("new_k")
                            new_kv["v"] = upd.get("new_v")
            
            handle = target_sa.register_forward_hook(_hook)

            # Clone caches: reset crossattn is_init=False so gradients flow through
            # cross-attention K/V computation (values are numerically identical since
            # prompt_probe has the same values as the original embeds).
            probe_kv = _clone_cache(kv_cache)
            probe_ca = _clone_crossattn(crossattn_cache, is_init_override=False)

            if do_c and prompt_probe is not None:
                # Provide prompt as list of per-batch tensors (matching original format)
                # Cast to context_latent dtype for the forward (bfloat16 in practice)
                _dtype = context_latent.dtype
                probe_embeds = [prompt_probe[i].to(_dtype) for i in range(prompt_probe.shape[0])]
            else:
                probe_embeds = [e.detach() for e in raw_embeds]

            probe_cond = {"prompt_embeds": probe_embeds}
            
            # Input: use x_probe (float32 → cast to model dtype inside forward)
            _input = x_probe.to(context_latent.dtype) if do_x else context_latent.detach()
        
        
            try:
                generator_model(
                    noisy_image_or_video=_input,
                    conditional_dict=probe_cond,
                    timestep=context_timestep,
                    kv_cache=probe_kv,
                    crossattn_cache=probe_ca,
                    current_start=current_start,
                )
            finally:
                handle.remove()

            nk, nv = new_kv["k"], new_kv["v"]

            if nk is None or nv is None:
                print(f"  [Probe A/B] layer={layer_id}: hook did not capture new K/V (recompute step?)")
                return 0.0, 0.0, 0.5, [], [], float("nan")

            # Accumulate norms in fp32
            nk_f = nk.float()
            nv_f = nv.float()
            kv_norm = float((nk_f.norm() ** 2 + nv_f.norm() ** 2) ** 0.5) + eps

            # Build list of leaf tensors and output tensors for VJP
            leaf_list = []
            if do_c and prompt_probe is not None:
                leaf_list.append(prompt_probe)
            if do_x and x_probe is not None:
                leaf_list.append(x_probe)
            
            if not leaf_list:
                return 0.0, 0.0, 0.5, [], [], float("nan")
            
            with torch.enable_grad():
                per_proj, mean_sq = _vjp_frob_sq([nk_f, nv_f], leaf_list, self.num_proj, self._gen)

            ci = 0
            e_c_list, e_x_list = [], []

            gamma_c, gamma_x = 0.0, 0.0

            if do_c and prompt_probe is not None:
                e_c_list = per_proj[ci]
                jc_frob = float(np.mean(e_c_list)) ** 0.5
                c_norm = float(prompt_probe.float().norm()) + eps
                gamma_c = jc_frob * c_norm / kv_norm
                ci += 1

            if do_x and x_probe is not None:
                e_x_list = per_proj[ci]
                jx_frob = float(np.mean(e_x_list)) ** 0.5
                x_norm = float(x_probe.float().norm()) + eps
                gamma_x = jx_frob * x_norm / kv_norm

            control_ratio = gamma_c / (gamma_c + gamma_x + eps)

            ent = float("nan")
            if len(e_c_list) >= 2 and len(e_x_list) >= 2:
                ec_arr = np.array(e_c_list, dtype=np.float64)
                ex_arr = np.array(e_x_list, dtype=np.float64)
                if ec_arr.std() > 0 and ex_arr.std() > 0:
                    ent = float(np.corrcoef(ec_arr, ex_arr)[0, 1])

            return gamma_c, gamma_x, control_ratio, e_c_list, e_x_list, ent

    # ------------------------------------------------------------------
    # Probe C: d y_t / d historical KV  (J_o)
    # ------------------------------------------------------------------

    def _probe_c(
        self, *, generator_model, noisy_input, conditional_dict,
        denoising_ts, kv_cache, crossattn_cache, current_start, layer_id,
    ):
        """
        Probe C:
            J_o = d pred_x0 / d historical KV

        Principle:
        - Do NOT reimplement self-attention / cache rolling.
        - Inject differentiable historical K/V into a cloned target-layer cache.
        - Let LongLive's original self-attention handle:
                RoPE
                recomputation
                sink tokens
                rolling eviction
                local attention window
                current K/V insertion
        - Disable final cache commit because the probe is read-only.
        """

        model = generator_model.model
        block_cache = kv_cache[layer_id]

        local_end = int(block_cache["local_end_index"].item())
        if local_end <= 0:
            return [], [], []

        # ------------------------------------------------------------
        # IMPORTANT:
        # LongLive inference globally disables grad.
        # Everything that creates the probe graph must happen here.
        # ------------------------------------------------------------
        with torch.inference_mode(False), torch.enable_grad():

            base_k = block_cache["k"]
            base_v = block_cache["v"]

            # --------------------------------------------------------
            # 1. Historical KV become FP32 leaf tensors
            # --------------------------------------------------------
            k_hist = (base_k[:, :local_end].detach().clone().float().requires_grad_(True))
            v_hist = (base_v[:, :local_end].detach().clone().float().requires_grad_(True))

            # --------------------------------------------------------
            # 2. Clone the complete cache normally
            # --------------------------------------------------------
            probe_kv = _clone_cache(kv_cache)
            probe_ca = _clone_crossattn(crossattn_cache)

            # --------------------------------------------------------
            # 3. Replace ONLY target layer cache with tensors connected
            #    to k_hist / v_hist.
            #
            #    Tail is kept because cache capacity can be > local_end.
            # --------------------------------------------------------
            k_tail = base_k[:, local_end:].detach().clone()
            v_tail = base_v[:, local_end:].detach().clone()

            probe_kv[layer_id]["k"] = torch.cat([k_hist.to(base_k.dtype), k_tail,],dim=1,)
            probe_kv[layer_id]["v"] = torch.cat([v_hist.to(base_v.dtype), v_tail,],dim=1,)

            # The pointers stay numerically identical to real inference.
            probe_kv[layer_id]["global_end_index"] = (block_cache["global_end_index"].detach().clone())
            probe_kv[layer_id]["local_end_index"] = (block_cache["local_end_index"].detach().clone())

            # --------------------------------------------------------
            # Debug sanity checks
            # --------------------------------------------------------
            assert k_hist.requires_grad
            assert v_hist.requires_grad

            assert probe_kv[layer_id]["k"].requires_grad
            assert probe_kv[layer_id]["v"].requires_grad

            if layer_id == self.probe_layers[0]:
                print(
                    f"[Probe C debug] "
                    f"layer={layer_id} "
                    f"local_end={local_end} "
                    f"capacity={base_k.shape[1]} "
                    f"global_end={int(block_cache['global_end_index'].item())} "
                    f"k_cache_grad={probe_kv[layer_id]['k'].requires_grad}"
                )

            probe_cond = {"prompt_embeds": [e.detach() for e in conditional_dict["prompt_embeds"]]}

            # --------------------------------------------------------
            # 4. DO NOT commit cache updates.
            #
            # LongLive computes all block outputs first and only afterwards
            # calls _apply_cache_updates(). The head does not need these
            # committed cache values, so suppressing the final commit does
            # not change pred_x0 for this forward.
            # --------------------------------------------------------
            orig_apply_cache_updates = model._apply_cache_updates

            # Gradient checkpointing is unnecessary for this diagnostic
            # and changes LongLive's execution branch.
            had_gc_attr = hasattr(model, "gradient_checkpointing")
            old_gradient_checkpointing = (
                model.gradient_checkpointing if had_gc_attr else None
            )

            try:
                model._apply_cache_updates = (
                    lambda kv_cache_arg, cache_update_infos: None
                )

                if had_gc_attr:
                    model.gradient_checkpointing = False

                _, pred_x0 = generator_model(
                    noisy_image_or_video=noisy_input.detach(),
                    conditional_dict=probe_cond,
                    timestep=denoising_ts,
                    kv_cache=probe_kv,
                    crossattn_cache=probe_ca,
                    current_start=current_start,
                )

            finally:
                model._apply_cache_updates = orig_apply_cache_updates

                if had_gc_attr:
                    model.gradient_checkpointing = old_gradient_checkpointing

            # --------------------------------------------------------
            # 5. VJP: random projection of output
            # --------------------------------------------------------
            y_f = pred_x0.float()

            if not y_f.requires_grad:
                raise RuntimeError(
                    f"Probe C output has no grad graph at layer={layer_id}: "
                    f"requires_grad={y_f.requires_grad}, "
                    f"grad_fn={y_f.grad_fn}"
                )

            # IMPORTANT:
            # use a generator on the SAME device as y_f.
            # self._gen may be a CPU Generator.
            probe_gen = torch.Generator(device=y_f.device)
            probe_gen.manual_seed(self.probe_seed + 100003 * layer_id)

            rv = torch.randn(y_f.shape,dtype=torch.float32,device=y_f.device,generator=probe_gen)

            scalar = (y_f * rv).sum()

            grads = autograd.grad(
                scalar,
                [k_hist, v_hist],
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )

            gk, gv = grads

            # --------------------------------------------------------
            # 6. Per-cache-position observability
            # --------------------------------------------------------
            jo_k = []
            jo_v = []
            jo_comb = []

            for slot in range(local_end):
                nk = (
                    float(gk[:, slot].float().norm())
                    if gk is not None else 0.0
                )

                nv = (
                    float(gv[:, slot].float().norm())
                    if gv is not None else 0.0
                )

                jo_k.append(nk)
                jo_v.append(nv)
                jo_comb.append(
                    math.sqrt(nk * nk + nv * nv)
                )

            return jo_k, jo_v, jo_comb

    # ------------------------------------------------------------------
    # Probe D: d KV_new / d historical KV  (J_m)
    # ------------------------------------------------------------------
    def _probe_d(
        self, *, generator_model, context_latent, conditional_dict,
        context_timestep, kv_cache, crossattn_cache, current_start, layer_id,
    ):
        block_cache = kv_cache[layer_id]
        local_end = int(block_cache["local_end_index"].item())
        if local_end == 0:
            return []

        k_hist_d = block_cache["k"][:, :local_end].detach().clone().float().requires_grad_(True)
        v_hist_d = block_cache["v"][:, :local_end].detach().clone().float().requires_grad_(True)

        new_kv_d: Dict[str, Optional[torch.Tensor]] = {"k": None, "v": None}
        
        target_sa = generator_model.model.blocks[layer_id].self_attn
        _orig = target_sa.forward

        def _patch_d(
            x_in, seq_lens, grid_sizes, freqs, block_mask,
            kv_cache_arg=None, current_start_arg=0,
            cache_start=None, sink_recache_after_switch=False
        ):
            b, s = x_in.shape
            n, d = target_sa.num_heads, target_sa.head_dim

            q = target_sa.norm_q(target_sa.q(x_in)).view(b, s, n, d)
            k_new = target_sa.norm_k(target_sa.k(x_in)).view(b, s, n, d)
            v_new = target_sa.v(x_in).view(b, s, n, d)

            if kv_cache_arg is None:
                return _orig(x_in, seq_lens, grid_sizes, freqs, block_mask)
            
            frame_seqlen = int(math.prod(grid_sizes[0][1:].tolist()))
            cur_frame = current_start_arg // frame_seqlen
            roped_q = causal_rope_apply(q, grid_sizes, freqs, start_frame=cur_frame).type_as(v_new)
            roped_k_new = causal_rope_apply(k_new, grid_sizes, freqs, start_frame=cur_frame).type_as(v_new)

            # Store new K/V (depend on x_in, not on k_hist_d directly,
            # but the attention output does depend on k_hist_d)
            new_kv_d["k"] = roped_k_new
            new_kv_d["v"] = v_new

            k_full = torch.cat([k_hist_d.to(roped_k_new.dtype), roped_k_new], dim=1)
            v_full = torch.cat([v_hist_d.to(v_new.dtype), v_new], dim=1)

            max_sz = target_sa.max_attention_size
            if k_full.shape[1] > max_sz:
                k_full = k_full[:, -max_sz:]
                v_full = v_full[:, -max_sz:]

            x_out = wan_attention(roped_q, k_full, v_full)
            x_out = x_out.flatten(2)
            x_out = target_sa.o(x_out)

            current_end = current_start_arg + roped_q.shape[1]
            local_end_idx = local_end + s
            dummy_info = {"action": "direct_insert", "new_k": roped_k_new, "new_v": v_new,
                          "local_start_index": local_end, "local_end_index": local_end_idx,
                          "write_start_index": local_end, "write_end_index": local_end_idx,
                          "current_end": current_end, "is_recompute": False}
            return x_out, (current_end, local_end_idx, dummy_info)

        target_sa.forward = _patch_d

        probe_kv = _clone_cache(kv_cache)
        probe_ca = _clone_crossattn(crossattn_cache)
        probe_cond = {"prompt_embeds": [e.detach() for e in conditional_dict["prompt_embeds"]]}

        try:
            with torch.enable_grad():
                generator_model(
                    noisy_image_or_video=context_latent.detach(),
                    conditional_dict=probe_cond,
                    timestep=context_timestep,
                    kv_cache=probe_kv,
                    crossattn_cache=probe_ca,
                    current_start=current_start,
                )
        finally:
            target_sa.forward = _orig

        nk_d = new_kv_d.get("k")
        nv_d = new_kv_d.get("v")
        if nk_d is None:
            return []

        # VJP
        with torch.enable_grad():
            rk = torch.randn(nk_d.shape, dtype=torch.float32, device=nk_d.device, generator=self._gen)
            rv = torch.randn(nv_d.shape, dtype=torch.float32, device=nv_d.device, generator=self._gen)
            scalar = (nk_d.float() * rk).sum() + (nv_d.float() * rv).sum()
            grads = autograd.grad(scalar, [k_hist_d, v_hist_d], allow_unused=True)

        gk = grads[0]
        gv = grads[1]

        jm = []
        for slot in range(local_end):
            nk = float(gk[:, slot].float().norm()) if gk is not None else 0.0
            nv = float(gv[:, slot].float().norm()) if gv is not None else 0.0
            jm.append((nk ** 2 + nv ** 2) ** 0.5)
        return jm

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def finalize(self) -> str:
        if not self.enabled:
            return ""
        path = os.path.join(self.output_dir, "probe_records.json")
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)
        print(f"[JacobianProbe] Saved {len(self.records)} records → {path}")
        return path
