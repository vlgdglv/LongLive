#!/usr/bin/env python3
"""
probes/probe_historical_routing.py

Offline analysis of historical attention routing in a saved rollout seed directory.

Part A (no model):
  Input/output geometry gain via CKA and KNN, grouped by prompt-boundary proximity.

Part B (model required):
  Replay context-update forwards to build KV history, capture post-RoPE Q/K/V via
  attention-function interception, compute manual softmax routing weights, and derive
  routing stability, sparsity, V-contribution, and prompt-switch metrics.

Usage (Part A only):
  python probes/probe_historical_routing.py \
      --input_dir outputs/probes/latent_1 \
      --output_dir outputs/probes/latent_1/routing_analysis

Usage (Part A + B):
  python probes/probe_historical_routing.py \
      --input_dir outputs/probes/latent_1 \
      --output_dir outputs/probes/latent_1/routing_analysis \
      --layers 0,10,20,29 \
      --config_path configs/longvid_local.yaml \
      --generator_ckpt outputs/checkpoints/model.pt
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants matching the 1.3B Wan model used in LongLive
# ---------------------------------------------------------------------------
FRAME_SEQ_LEN = 1560       # tokens per latent frame
NUM_HEADS     = 12
HEAD_DIM      = 128
NUM_TF_BLOCKS = 30          # transformer blocks
POOL_H        = 8
POOL_W        = 14
POOL_DIM      = POOL_H * POOL_W   # 112 features after spatial pool

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _glob_latent_files(input_dir: str) -> List[Path]:
    """Return all latents_X_Y.pt files sorted by start frame index."""
    pattern = re.compile(r"latents_(\d+)_(\d+)\.pt$")
    files = []
    for p in Path(input_dir).iterdir():
        m = pattern.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda t: t[0])
    return [p for _, p in files]


def _load_latent_file(path: Path) -> Tuple[int, str, dict]:
    """
    Load one latents_X_Y.pt file. Returns (block_frame_start, embed_path, step_dict).

    step_dict maps step_idx (int) → {"step_idx", "input_timestep",
                                      "input_before_forward", "pred_x0"}.
    Handles both multi-step format (outer dict keyed by int) and single-step
    format (outer dict is the step data directly).
    """
    m = re.search(r"latents_(\d+)_\d+\.pt$", path.name)
    block_frame_start = int(m.group(1)) if m else 0

    data = torch.load(path, map_location="cpu", weights_only=False)
    latents = data["latents"]
    embed_path = data.get("embed_path", "")

    keys = list(latents.keys())
    if keys and isinstance(keys[0], int):
        # Multi-step: {step_idx: {step_idx, input_timestep, ...}, ...}
        step_dict = latents
    else:
        # Single-step: {step_idx, input_timestep, ...}
        step_dict = {latents["step_idx"]: latents}

    return block_frame_start, embed_path, step_dict


def load_all_blocks(input_dir: str) -> List[dict]:
    """
    Return list of block info dicts, sorted by block_frame_start. Each has:
      block_frame_start, embed_path, prompt_id (inferred), step_dict
    """
    files = _glob_latent_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No latents_X_Y.pt files found in {input_dir}")

    blocks = []
    for path in files:
        bfs, embed_path, step_dict = _load_latent_file(path)
        blocks.append({
            "block_frame_start": bfs,
            "embed_path": embed_path,
            "step_dict": step_dict,
        })

    # Infer prompt_id from embed_path (e.g. "prompt0_embeds.pt" → 0)
    _id_map: dict = {}
    for blk in blocks:
        ep = os.path.basename(blk["embed_path"])
        m = re.search(r"prompt(\d+)", ep)
        raw = int(m.group(1)) if m else 0
        if raw not in _id_map:
            _id_map[raw] = len(_id_map)
        blk["prompt_id"] = _id_map[raw]

    return blocks


def infer_switch_indices(blocks: List[dict]) -> List[int]:
    """Return block indices (into `blocks`) where prompt_id changes."""
    switches = []
    for i in range(1, len(blocks)):
        if blocks[i]["prompt_id"] != blocks[i - 1]["prompt_id"]:
            switches.append(i)
    return switches


def block_group(block_idx: int, switch_indices: List[int]) -> str:
    """Classify a block as 'switch', 'switch+1', 'switch+2', or 'normal'."""
    for sw in switch_indices:
        if block_idx == sw - 1:
            return "switch"
        if block_idx == sw:
            return "switch+1"
        if block_idx == sw + 1:
            return "switch+2"
    return "normal"


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def spatial_pool_flat(t: torch.Tensor) -> np.ndarray:
    """
    Pool a latent tensor to a compact spatial representation.
    t: [B, F, C, H, W] or [F, C, H, W] — uses frame-mean, then channel-mean.
    Returns float32 numpy array of shape (POOL_DIM,).
    """
    if t.ndim == 5:
        t = t[0]      # drop batch
    # t: [F, C, H, W]
    t = t.float()
    t = t.mean(0)     # [C, H, W]
    t = t.mean(0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]
    t = F.adaptive_avg_pool2d(t, (POOL_H, POOL_W))  # [1, 1, 8, 14]
    return t.squeeze().cpu().numpy().astype(np.float32).ravel()


def _gram(X: np.ndarray) -> np.ndarray:
    """Centered Gram matrix for rows of X (N, d)."""
    G = X @ X.T
    n = G.shape[0]
    mu_row = G.mean(axis=1, keepdims=True)
    mu_col = G.mean(axis=0, keepdims=True)
    mu_all = G.mean()
    return G - mu_row - mu_col + mu_all


def centered_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """CKA between row matrices X (N, d1) and Y (N, d2)."""
    Kx = _gram(X)
    Ky = _gram(Y)
    num = np.sum(Kx * Ky)
    denom = np.sqrt(np.sum(Kx * Kx) * np.sum(Ky * Ky))
    return float(num / (denom + 1e-10))


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix for rows of X (N, d)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms
    return Xn @ Xn.T


def knn_preservation(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    """
    Fraction of k-NN neighbours preserved between row-matrices X and Y (same N rows).
    """
    N = X.shape[0]
    if N <= k:
        return 1.0
    Cx = cosine_matrix(X)
    Cy = cosine_matrix(Y)
    np.fill_diagonal(Cx, -np.inf)
    np.fill_diagonal(Cy, -np.inf)
    nx = np.argsort(-Cx, axis=1)[:, :k]
    ny = np.argsort(-Cy, axis=1)[:, :k]
    overlap = sum(
        len(set(nx[i]) & set(ny[i])) / k for i in range(N)
    )
    return float(overlap / N)


def gini(arr: np.ndarray) -> float:
    """Gini coefficient for a non-negative 1-D array."""
    arr = np.sort(np.abs(arr))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)


# ---------------------------------------------------------------------------
# SegmentMap – tracks which cache positions belong to which AR block
# ---------------------------------------------------------------------------

@dataclass
class SegmentInfo:
    block_frame_start: int
    cache_start: int   # inclusive
    cache_end:   int   # exclusive
    prompt_id:   int


class SegmentMap:
    """
    Mirrors the KV cache write pattern of CausalWanSelfAttention so we can
    map attention K positions back to originating AR blocks.
    """

    def __init__(self, cache_size: int, frame_seq_length: int, sink_tokens: int = 0):
        self.cache_size       = cache_size
        self.frame_seq_length = frame_seq_length
        self.sink_tokens      = sink_tokens
        self.segments:  List[SegmentInfo] = []
        self.local_end: int  = 0   # next write position (before current block)

    def commit_block(self, block_frame_start: int, n_frames: int, prompt_id: int) -> None:
        """Record that `n_frames` frames were committed to the KV cache."""
        tokens       = n_frames * self.frame_seq_length
        local_end_old = self.local_end

        if local_end_old + tokens <= self.cache_size:
            # Direct insert: no rolling
            self.segments.append(SegmentInfo(
                block_frame_start=block_frame_start,
                cache_start=local_end_old,
                cache_end=local_end_old + tokens,
                prompt_id=prompt_id,
            ))
            self.local_end = local_end_old + tokens
        else:
            # Rolling: evict oldest num_evicted non-sink tokens
            num_evicted = local_end_old + tokens - self.cache_size
            # Shift all non-sink segments left by num_evicted
            surviving = []
            for seg in self.segments:
                if seg.cache_start < self.sink_tokens:
                    # Sink segment – positions unchanged
                    surviving.append(seg)
                elif seg.cache_end <= self.sink_tokens + num_evicted:
                    # Fully evicted
                    pass
                else:
                    new_start = max(seg.cache_start - num_evicted, self.sink_tokens)
                    new_end   = seg.cache_end - num_evicted
                    surviving.append(SegmentInfo(
                        block_frame_start=seg.block_frame_start,
                        cache_start=new_start,
                        cache_end=new_end,
                        prompt_id=seg.prompt_id,
                    ))
            # New block always lands at the end of the cache
            new_start = self.cache_size - tokens
            surviving.append(SegmentInfo(
                block_frame_start=block_frame_start,
                cache_start=new_start,
                cache_end=self.cache_size,
                prompt_id=prompt_id,
            ))
            self.segments  = surviving
            self.local_end = self.cache_size

    def cache_pos_to_info(self, pos: int) -> Optional[SegmentInfo]:
        for seg in self.segments:
            if seg.cache_start <= pos < seg.cache_end:
                return seg
        return None

    def k_index_to_frame(
        self,
        k_idx: int,
        current_tokens: int,
        max_attention_size: int,
    ) -> Tuple[Optional[SegmentInfo], int]:
        """
        Like k_index_to_info but also returns frame_within_block (0-based).
        Returns (None, -1) when the position is outside the historical window.
        Must be called BEFORE commit_block for the current block.
        """
        local_end_old = self.local_end
        local_end_new = local_end_old + current_tokens

        if self.sink_tokens > 0:
            local_budget = max_attention_size - self.sink_tokens
            if local_budget <= 0:
                if k_idx < self.sink_tokens:
                    cache_pos = k_idx
                else:
                    return None, -1
            else:
                local_start_for_window = max(self.sink_tokens, local_end_new - local_budget)
                if k_idx < self.sink_tokens:
                    cache_pos = k_idx
                else:
                    cache_pos = local_start_for_window + (k_idx - self.sink_tokens)
        else:
            window_start = max(0, local_end_new - max_attention_size)
            cache_pos    = window_start + k_idx

        if cache_pos >= local_end_old:
            return None, -1

        seg = self.cache_pos_to_info(cache_pos)
        if seg is None:
            return None, -1

        frame_within_block = (cache_pos - seg.cache_start) // self.frame_seq_length
        return seg, frame_within_block


    def k_index_to_info(self,
        k_idx: int,
        current_tokens: int,
        max_attention_size: int,
    ) -> Optional[SegmentInfo]:
        """
        Map position k_idx inside the captured K tensor (historical part only,
        excluding the current block's own tokens at the end) to a SegmentInfo.

        Must be called BEFORE commit_block for the current block.
        """
        local_end_old = self.local_end
        local_end_new = local_end_old + current_tokens  # after writing current tokens

        if self.sink_tokens > 0:
            local_budget = max_attention_size - self.sink_tokens
            if local_budget <= 0:
                if k_idx < self.sink_tokens:
                    return self.cache_pos_to_info(k_idx)
                return None
            local_start_for_window = max(self.sink_tokens, local_end_new - local_budget)
            if k_idx < self.sink_tokens:
                cache_pos = k_idx
            else:
                cache_pos = local_start_for_window + (k_idx - self.sink_tokens)
        else:
            window_start = max(0, local_end_new - max_attention_size)
            cache_pos    = window_start + k_idx

        if cache_pos >= local_end_old:
            return None   # current block's own keys
        return self.cache_pos_to_info(cache_pos)


# ---------------------------------------------------------------------------
# Part A: Input/output geometry gain
# ---------------------------------------------------------------------------

def _build_rep_matrix(
    blocks: List[dict], step_idx: int, key: str
) -> Optional[np.ndarray]:
    """
    Build representation matrix [N, POOL_DIM] from all blocks at the given
    denoising step_idx. key is 'input_before_forward' or 'pred_x0'.
    Returns None if the step is not available in any block.
    """
    rows = []
    for blk in blocks:
        sd = blk["step_dict"]
        if step_idx not in sd:
            continue
        t = sd[step_idx][key]
        rows.append(spatial_pool_flat(t))
    if not rows:
        return None
    return np.stack(rows, axis=0)


def run_part_a(
    blocks: List[dict],
    output_dir: str,
) -> dict:
    """
    Compute geometry gain metrics (CKA and KNN) for Part A.
    Returns a dict of metrics for saving.
    """
    print("\n[Part A] Input/output geometry gain analysis")
    switch_indices = infer_switch_indices(blocks)
    N = len(blocks)

    # Gather all available step indices
    all_step_ids: List[int] = []
    for blk in blocks:
        for sid in blk["step_dict"]:
            if sid not in all_step_ids:
                all_step_ids.append(sid)
    all_step_ids.sort()

    if not all_step_ids:
        print("[Part A] No step data found.")
        return {}

    final_step = max(all_step_ids)

    # Final pred_x0 used as the anchor geometry
    H_x0_final = _build_rep_matrix(blocks, final_step, "pred_x0")
    if H_x0_final is None:
        print("[Part A] Final pred_x0 unavailable.")
        return {}

    cka_noisy, cka_x0, knn_noisy, knn_x0 = [], [], [], []
    timesteps_used = []

    for sid in all_step_ids:
        H_noisy = _build_rep_matrix(blocks, sid, "input_before_forward")
        H_x0    = _build_rep_matrix(blocks, sid, "pred_x0")
        if H_noisy is None or H_x0 is None:
            continue

        # CKA against final pred_x0
        cka_n = centered_linear_cka(H_noisy, H_x0_final)
        cka_p = centered_linear_cka(H_x0,    H_x0_final)
        knn_n = knn_preservation(H_noisy, H_x0_final)
        knn_p = knn_preservation(H_x0,    H_x0_final)

        cka_noisy.append(cka_n)
        cka_x0.append(cka_p)
        knn_noisy.append(knn_n)
        knn_x0.append(knn_p)

        ts = blocks[0]["step_dict"].get(sid, {}).get("input_timestep", sid)
        timesteps_used.append(float(ts))
        print(f"  step={sid}  t={ts:.1f}  CKA_noisy={cka_n:.4f}  CKA_x0={cka_p:.4f}"
              f"  CKA_gain={cka_p-cka_n:+.4f}")

    # --- Prompt-boundary grouping (final step only) ---
    groups = [block_group(i, switch_indices) for i in range(N)]
    unique_groups = ["normal", "switch", "switch+1", "switch+2"]
    group_cka_gain: Dict[str, List[float]] = {g: [] for g in unique_groups}
    group_knn_gain: Dict[str, List[float]] = {g: [] for g in unique_groups}

    for i, blk in enumerate(blocks):
        sd = blk["step_dict"]
        if final_step not in sd:
            continue
        h_n = _build_rep_matrix([blk], final_step, "input_before_forward")
        h_x = _build_rep_matrix([blk], final_step, "pred_x0")
        if h_n is None or h_x is None:
            continue
        # Single-block CKA is trivially 1; use pairwise cosine distance instead
        # For boundary analysis: compare this block's noisy vs x0 vs final x0
        # using the entire population anchor H_x0_final
        g = groups[i]
        cka_g_n = centered_linear_cka(
            np.concatenate([h_n, H_x0_final[i:i+1]], axis=0),
            np.concatenate([h_n, H_x0_final[i:i+1]], axis=0)
        )  # self-CKA for sanity; real gain is cross-block
        # Use cosine similarity as a per-block scalar instead
        cos_n = float(np.dot(h_n[0], H_x0_final[i]) /
                      (np.linalg.norm(h_n[0]) * np.linalg.norm(H_x0_final[i]) + 1e-8))
        cos_x = float(np.dot(h_x[0], H_x0_final[i]) /
                      (np.linalg.norm(h_x[0]) * np.linalg.norm(H_x0_final[i]) + 1e-8))
        group_cka_gain[g].append(cos_x - cos_n)
        group_knn_gain[g].append(cos_x)

    print("\n[Part A] Per-group cosine gain (noisy→x0 vs final x0):")
    group_summary = {}
    for g in unique_groups:
        vals = group_cka_gain[g]
        if vals:
            mu, sd_ = float(np.mean(vals)), float(np.std(vals))
            print(f"  {g:12s}: mean_gain={mu:+.4f}  std={sd_:.4f}  n={len(vals)}")
            group_summary[g] = {"mean_cos_gain": mu, "std": sd_, "n": len(vals)}
        else:
            group_summary[g] = {"mean_cos_gain": None, "std": None, "n": 0}

    # --- Plots ---
    if cka_noisy:
        _plot_geometry_gain(
            timesteps_used, cka_noisy, cka_x0, knn_noisy, knn_x0,
            group_summary, output_dir
        )

    return {
        "timesteps": timesteps_used,
        "cka_noisy_vs_final": cka_noisy,
        "cka_x0_vs_final":    cka_x0,
        "knn_noisy_vs_final": knn_noisy,
        "knn_x0_vs_final":    knn_x0,
        "group_cosine_gain":  group_summary,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_geometry_gain(
    timesteps, cka_noisy, cka_x0, knn_noisy, knn_x0,
    group_summary: dict, output_dir: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    xs = list(range(len(timesteps)))
    xlabels = [f"{int(t)}" for t in timesteps]

    ax = axes[0]
    ax.plot(xs, cka_noisy, "o--", label="noisy_input")
    ax.plot(xs, cka_x0,    "s-",  label="pred_x0")
    ax.set_xticks(xs); ax.set_xticklabels(xlabels)
    ax.set_xlabel("timestep"); ax.set_ylabel("CKA vs final x0")
    ax.set_title("CKA geometry: noisy_input vs pred_x0")
    ax.legend()

    ax = axes[1]
    groups_ordered = ["normal", "switch", "switch+1", "switch+2"]
    gains = [group_summary[g]["mean_cos_gain"] or 0 for g in groups_ordered]
    ax.bar(groups_ordered, gains)
    ax.set_xlabel("Block group"); ax.set_ylabel("Mean cosine gain (x0 - noisy)")
    ax.set_title("Geometry gain by prompt proximity")
    ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "input_output_geometry_gain.png"), dpi=120)
    plt.close(fig)
    print(f"[Part A] Saved input_output_geometry_gain.png")


# ---------------------------------------------------------------------------
# Part B: attention capture mechanism
# ---------------------------------------------------------------------------

class _AttentionCapture:
    """
    Context manager that temporarily replaces wan.modules.causal_model.attention
    with a call-count wrapper.  The i-th call during a forward corresponds to
    transformer layer i.  Records q, k, v for target layer indices.
    """

    def __init__(self, target_layers: List[int]):
        self.target_layers = set(target_layers)
        self.captures: Dict[int, dict] = {}
        self._call_count = 0
        self._orig = None
        self._module = None

    def __enter__(self):
        import wan.modules.causal_model as _cm
        self._module = _cm
        self._orig   = _cm.attention
        self._call_count = 0
        self.captures.clear()

        capture = self

        def _capturing(q, k, v, **kwargs):
            layer_id = capture._call_count
            capture._call_count += 1
            if layer_id in capture.target_layers:
                capture.captures[layer_id] = {
                    "q": q.detach().float().cpu(),
                    "k": k.detach().float().cpu(),
                    "v": v.detach().float().cpu(),
                }
            return capture._orig(q, k, v, **kwargs)

        _cm.attention = _capturing
        return self

    def __exit__(self, *_):
        if self._module is not None and self._orig is not None:
            self._module.attention = self._orig


def _compute_routing(
    q: torch.Tensor,    # [1, L_q, H, D]
    k_hist: torch.Tensor,  # [1, L_hist, H, D]
    v_hist: torch.Tensor,  # [1, L_hist, H, D]
    chunk_q: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-position routing weights averaged over query tokens and heads.

    Returns:
      avg_routing_by_head: [H, L_hist]  — per-head average routing mass
      v_weighted_norm:     [L_hist]     — ||weighted V|| per K position (head-averaged)
    """
    L_q    = q.shape[1]
    L_hist = k_hist.shape[1]
    H, D   = q.shape[2], q.shape[3]
    scale  = D ** -0.5

    # Move to CUDA if available for speed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q      = q[0].to(device)        # [L_q, H, D]
    k      = k_hist[0].to(device)   # [L_hist, H, D]
    v      = v_hist[0].to(device)   # [L_hist, H, D]

    # Accumulate routing over Q in chunks (memory bound otherwise)
    routing_sum = torch.zeros(H, L_hist, device=device)  # [H, L_hist]
    for q_s in range(0, L_q, chunk_q):
        q_c = q[q_s:q_s + chunk_q]               # [chunk, H, D]
        # scores: [H, chunk, L_hist]
        scores = torch.einsum("qhd,khd->hqk", q_c, k) * scale
        attn   = scores.softmax(dim=-1)           # [H, chunk, L_hist]
        routing_sum += attn.sum(dim=1)            # [H, L_hist]

    avg_routing = (routing_sum / L_q).cpu().numpy()   # [H, L_hist]

    # V contribution: routing_mean * ||v_j|| averaged over heads
    avg_r    = avg_routing.mean(0)                      # [L_hist]
    v_norms  = v.norm(dim=-1).mean(dim=1).cpu().numpy() # [L_hist, H] → mean over H → [L_hist]
    v_weighted = avg_r * v_norms

    return avg_routing, v_weighted


# ---------------------------------------------------------------------------
# Part B: model loading and cache management
# ---------------------------------------------------------------------------

def _load_model(args):
    """
    Load generator model, matching the exact loading sequence used in
    interactive_inference_probe.py:
      1. Create WanDiffusionWrapper (loads pretrained architecture weights)
      2. Load base checkpoint from state["generator_ema"] or state["generator"]
      3. Apply LoRA wrapper + weights if config.adapter / config.lora_ckpt are set
      4. Cast to bfloat16 and move to CUDA

    Returns (generator, local_attn_size, sink_size).
    """
    try:
        from omegaconf import OmegaConf
    except ImportError:
        raise ImportError("omegaconf required for Part B: pip install omegaconf")

    cfg = OmegaConf.load(args.config_path)
    model_kwargs = OmegaConf.to_container(cfg.model_kwargs, resolve=True)

    from utils.wan_wrapper import WanDiffusionWrapper
    generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)
    generator.eval().requires_grad_(False)

    # --- Base checkpoint (mirrors interactive_inference_probe.py lines 79-97) ---
    ckpt_path = args.generator_ckpt or getattr(cfg, "generator_ckpt", None)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        use_ema = getattr(cfg, "use_ema", False)
        if use_ema and "generator_ema" in state:
            raw = state["generator_ema"]
            # Clean FSDP wrapper prefix
            raw = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in raw.items()}
            missing, unexpected = generator.load_state_dict(raw, strict=False)
        elif "generator" in state:
            generator.load_state_dict(state["generator"])
        else:
            # Fallback: assume the file IS the generator state dict
            generator.load_state_dict(state, strict=False)
        print(f"[Part B] Loaded base generator from {ckpt_path}")
    else:
        print(f"[Part B] WARNING: no base checkpoint loaded (ckpt_path={ckpt_path})")

    # --- LoRA (mirrors interactive_inference_probe.py lines 99-133) ---
    adapter_cfg = getattr(cfg, "adapter", None)
    if adapter_cfg is not None:
        try:
            import peft
            from utils.lora_utils import configure_lora_for_model
        except ImportError as e:
            raise ImportError(f"peft / lora_utils required for LoRA loading: {e}")

        print(f"[Part B] Applying LoRA adapter: {adapter_cfg}")
        generator.model = configure_lora_for_model(
            generator.model,
            model_name="generator",
            lora_config=adapter_cfg,
            is_main_process=True,
        )

        lora_ckpt_path = args.lora_ckpt or getattr(cfg, "lora_ckpt", None)
        if lora_ckpt_path and os.path.exists(lora_ckpt_path):
            lora_state = torch.load(lora_ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(lora_state, dict) and "generator_lora" in lora_state:
                peft.set_peft_model_state_dict(generator.model, lora_state["generator_lora"])
            else:
                peft.set_peft_model_state_dict(generator.model, lora_state)
            print(f"[Part B] Loaded LoRA weights from {lora_ckpt_path}")
        else:
            print(f"[Part B] WARNING: no LoRA checkpoint loaded (lora_ckpt_path={lora_ckpt_path})")

    # Cast to bfloat16 and move to CUDA (mirrors lines 137-141)
    generator = generator.to(dtype=torch.bfloat16)
    if torch.cuda.is_available():
        generator = generator.cuda()
    generator.eval().requires_grad_(False)

    local_attn_size = model_kwargs.get("local_attn_size", -1)
    sink_size       = model_kwargs.get("sink_size", 0)
    return generator, local_attn_size, sink_size


def _init_kv_cache(
    n_blocks: int,
    cache_size: int,
    device: torch.device,
    dtype=torch.bfloat16,
) -> List[dict]:
    return [
        {
            "k":                 torch.zeros(1, cache_size, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype),
            "v":                 torch.zeros(1, cache_size, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype),
            "global_end_index":  torch.tensor(0, device=device),
            "local_end_index":   torch.tensor(0, device=device),
            "write_info":        (0, 0),
        }
        for _ in range(n_blocks)
    ]


def _init_crossattn_cache(
    n_blocks: int,
    device: torch.device,
    dtype=torch.bfloat16,
    context_seq_len: int = 512,
) -> List[dict]:
    # Matches CausalInferencePipeline._initialize_crossattn_cache:
    # k/v shape [B, 512, 12, 128], is_init=False
    return [
        {
            "k":      torch.zeros(1, context_seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype),
            "v":      torch.zeros(1, context_seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype),
            "is_init": False,
        }
        for _ in range(n_blocks)
    ]


def _snapshot_kv(kv_cache: List[dict]) -> List[dict]:
    return [
        {
            "k":                c["k"].clone(),
            "v":                c["v"].clone(),
            "global_end_index": c["global_end_index"].clone(),
            "local_end_index":  c["local_end_index"].clone(),
            "write_info":       c["write_info"],
        }
        for c in kv_cache
    ]


def _restore_kv(kv_cache: List[dict], snapshot: List[dict]) -> None:
    for c, s in zip(kv_cache, snapshot):
        c["k"].copy_(s["k"])
        c["v"].copy_(s["v"])
        c["global_end_index"].copy_(s["global_end_index"])
        c["local_end_index"].copy_(s["local_end_index"])
        c["write_info"] = s["write_info"]


def _snapshot_crossattn(ca_cache: List[dict]) -> List[dict]:
    return [
        {"k": c["k"].clone(), "v": c["v"].clone(), "is_init": c["is_init"]}
        for c in ca_cache
    ]


def _restore_crossattn(ca_cache: List[dict], snapshot: List[dict]) -> None:
    for c, s in zip(ca_cache, snapshot):
        c["k"].copy_(s["k"])
        c["v"].copy_(s["v"])
        c["is_init"] = s["is_init"]


# ---------------------------------------------------------------------------
# Part B: prompt embedding loader and recache helper
# ---------------------------------------------------------------------------

def _load_prompt_embeds(
    blocks: List[dict], input_dir: str, device, dtype
) -> Dict[int, dict]:
    """
    Load saved prompt{N}_embeds.pt files. Returns {prompt_id: conditional_dict}
    where conditional_dict = {"prompt_embeds": tensor}.

    Tries the stored embed_path first (absolute path from collection time),
    then falls back to input_dir/prompt{N}_embeds.pt.
    """
    result: Dict[int, dict] = {}
    for blk in blocks:
        pid = blk["prompt_id"]
        if pid in result:
            continue
        # Try absolute path from collection, then relative to input_dir
        candidates = [
            blk["embed_path"],
            os.path.join(input_dir, os.path.basename(blk["embed_path"])),
            os.path.join(input_dir, f"prompt{pid}_embeds.pt"),
        ]
        loaded = None
        for ep in candidates:
            if ep and os.path.exists(ep):
                loaded = torch.load(ep, map_location="cpu", weights_only=False)
                print(f"  [Part B] prompt_id={pid} embeddings loaded from {ep}")
                break
        if loaded is None:
            print(f"  [Part B] WARNING: prompt_id={pid} embeddings not found; using zeros")
            loaded = torch.zeros(1, 512, 4096)
        if isinstance(loaded, torch.Tensor):
            loaded = loaded.to(device=device, dtype=dtype)
        result[pid] = {"prompt_embeds": loaded}
    return result


def _do_recache(
    generator,
    kv_cache: List[dict],
    crossattn_cache: List[dict],
    output_frames: Dict[int, torch.Tensor],  # bfs → [1, n_frames, C, H, W] on CPU
    current_start_frame: int,
    new_cond: dict,
    local_attn_size: int,
    frame_seq_length: int,
    num_frame_per_block: int,
    context_noise: int,
    device,
    dtype,
    global_sink: bool = False,
) -> None:
    """
    Mirror InteractiveCausalInferencePipeline._recache_after_switch exactly.

    Zeros KV content (but NOT indices), zeros crossattn, replays prior clean
    frames with the new prompt so cross-attention is re-conditioned, then
    zeros crossattn again.
    """
    # 1. Zero KV content (indices intentionally kept — matches original)
    if not global_sink:
        for cache in kv_cache:
            cache["k"].zero_()
            cache["v"].zero_()

    # 2. Zero crossattn
    for blk in crossattn_cache:
        blk["k"].zero_()
        blk["v"].zero_()
        blk["is_init"] = False

    if current_start_frame == 0:
        return

    # 3. Determine recache window
    if local_attn_size == -1:
        num_recache = current_start_frame
    else:
        num_recache = min(local_attn_size, current_start_frame)
    recache_start = current_start_frame - num_recache

    # 4. Stack saved pred_x0 frames (CPU→GPU)
    all_frames = torch.cat(
        [output_frames[bfs] for bfs in sorted(output_frames)], dim=1
    )  # [1, total_frames, C, H, W]
    frames_slice = all_frames[:, recache_start:current_start_frame].to(device=device, dtype=dtype)
    actual_n = frames_slice.shape[1]

    print(f"  [recache] start={recache_start} end={current_start_frame} n={actual_n}")

    # 5. Block mask (same call as original _recache_after_switch)
    block_mask = generator.model._prepare_blockwise_causal_attn_mask(
        device=device,
        num_frames=actual_n,
        frame_seqlen=frame_seq_length,
        num_frame_per_block=num_frame_per_block,
        local_attn_size=local_attn_size,
    )
    generator.model.block_mask = block_mask

    ctx_ts = torch.ones([1, actual_n], device=device, dtype=torch.long) * context_noise

    # 6. Recache forward
    with torch.no_grad():
        generator(
            noisy_image_or_video=frames_slice,
            conditional_dict=new_cond,
            timestep=ctx_ts,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=recache_start * frame_seq_length,
            sink_recache_after_switch=not global_sink,
        )

    # 7. Zero crossattn again after recache (matches original)
    for blk in crossattn_cache:
        blk["k"].zero_()
        blk["v"].zero_()
        blk["is_init"] = False


# ---------------------------------------------------------------------------
# Part B: routing probe
# ---------------------------------------------------------------------------

def run_part_b(
    blocks: List[dict],
    input_dir: str,
    output_dir: str,
    generator,
    local_attn_size: int,
    sink_size: int,
    target_layers: List[int],
    context_noise: int = 0,
    probe_every: int = 1,
    global_sink: bool = False,
) -> dict:
    """
    Replay context-update forwards with real prompt embeddings and prompt-switch
    recache to build an accurate KV history, then probe denoising forwards for
    each probe block to extract self-attention routing weights.
    """
    print("\n[Part B] Historical routing probe")

    device = next(generator.parameters()).device
    dtype  = next(generator.parameters()).dtype

    frame_seq_len  = FRAME_SEQ_LEN
    N              = len(blocks)
    switch_indices = infer_switch_indices(blocks)

    # Load real prompt embeddings
    cond_by_pid = _load_prompt_embeds(blocks, input_dir, device, dtype)

    # Infer num_frame_per_block from saved latents
    first_sd   = list(blocks[0]["step_dict"].values())[0]
    num_fpb    = first_sd["pred_x0"].shape[1]

    # Determine KV cache size
    if local_attn_size > 0:
        cache_size = local_attn_size * frame_seq_len
    else:
        total_frames = max(
            blk["block_frame_start"] + list(blk["step_dict"].values())[0]["pred_x0"].shape[1]
            for blk in blocks
        )
        cache_size = total_frames * frame_seq_len

    sink_tokens   = sink_size * frame_seq_len
    max_attn_size = cache_size

    kv_cache        = _init_kv_cache(NUM_TF_BLOCKS, cache_size, device, dtype)
    crossattn_cache = _init_crossattn_cache(NUM_TF_BLOCKS, device, dtype)
    seg_map         = SegmentMap(cache_size, frame_seq_len, sink_tokens)

    # Buffer of clean frames generated so far, used for prompt-switch recache.
    # Keyed by block_frame_start, value is [1, n_frames, C, H, W] on CPU.
    output_frames: Dict[int, torch.Tensor] = {}

    # Routing results: {block_idx: {layer_id: [routing_array_per_denoising_step]}}
    routing_by_step: Dict[int, Dict[int, List[np.ndarray]]] = {}

    prev_pid = blocks[0]["prompt_id"]

    for block_idx, blk in enumerate(blocks):
        bfs = blk["block_frame_start"]
        pid = blk["prompt_id"]
        sd  = blk["step_dict"]

        final_step     = max(sd.keys())
        n_frames       = sd[final_step]["pred_x0"].shape[1]
        current_tokens = n_frames * frame_seq_len
        cond           = cond_by_pid[pid]

        # ---------------------------------------------------------------
        # Prompt switch: replicate _recache_after_switch before anything else
        # ---------------------------------------------------------------
        if pid != prev_pid:
            print(f"  [Part B] Prompt switch at block={block_idx} (frame={bfs}): "
                  f"pid {prev_pid}→{pid}")
            _do_recache(
                generator       = generator,
                kv_cache        = kv_cache,
                crossattn_cache = crossattn_cache,
                output_frames   = output_frames,
                current_start_frame = bfs,
                new_cond        = cond,
                local_attn_size = local_attn_size,
                frame_seq_length= frame_seq_len,
                num_frame_per_block = num_fpb,
                context_noise   = context_noise,
                device          = device,
                dtype           = dtype,
                global_sink     = global_sink,
            )
            # SegmentMap: after recache, KV content refreshed but positions
            # unchanged.  No structural change to seg_map needed.
            prev_pid = pid

        probe_this = (block_idx % probe_every == 0)

        # Pre-build bfs→index lookup once per probe block (used in inner layer loop)
        bfs_to_idx: Dict[int, int] = {
            b["block_frame_start"]: j for j, b in enumerate(blocks[:block_idx])
        }

        # ---------------------------------------------------------------
        # Step 1: probe denoising forward (capture routing, NO cache commit)
        # ---------------------------------------------------------------
        if probe_this and target_layers:
            for sid in sorted(sd.keys()):
                entry    = sd[sid]
                noisy_in = entry["input_before_forward"].to(device=device, dtype=dtype)
                t_val    = float(entry["input_timestep"])
                timestep = (
                    torch.ones([noisy_in.shape[0], n_frames], device=device, dtype=torch.long)
                    * int(t_val)
                )

                snap_kv = _snapshot_kv(kv_cache)
                snap_ca = _snapshot_crossattn(crossattn_cache)
                with torch.no_grad(), _AttentionCapture(target_layers) as cap:
                    generator(
                        noisy_image_or_video=noisy_in,
                        conditional_dict=cond,
                        timestep=timestep,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=bfs * frame_seq_len,
                    )
                _restore_kv(kv_cache, snap_kv)       # do NOT commit denoising to cache
                _restore_crossattn(crossattn_cache, snap_ca)

                for layer_id, tensors in cap.captures.items():
                    L_kv   = tensors["k"].shape[1]
                    L_hist = L_kv - current_tokens
                    if L_hist <= 0:
                        continue
                    k_hist = tensors["k"][:, :L_hist]
                    v_hist = tensors["v"][:, :L_hist]
                    q      = tensors["q"]

                    avg_routing, _ = _compute_routing(q, k_hist, v_hist)

                    # Map K positions to VISIBLE AR blocks and frames (single pass).
                    # Blocks evicted from rolling KV cache are NOT mapped.
                    routing_per_pos = avg_routing.mean(0)          # [L_hist] head-averaged

                    visible_mass:  Dict[int, float]         = {}   # block_idx → mass
                    frame_mass:    Dict[Tuple, float]       = {}   # (block_idx, frame_w) → mass
                    sink_set:      set                      = set()
                    rolling_set:   set                      = set()

                    for k_pos in range(L_hist):
                        seg_info, frame_w = seg_map.k_index_to_frame(
                            k_pos, current_tokens, max_attn_size
                        )
                        if seg_info is None:
                            continue
                        tgt = bfs_to_idx.get(seg_info.block_frame_start)
                        if tgt is None:
                            continue
                        m = float(routing_per_pos[k_pos])
                        visible_mass[tgt]               = visible_mass.get(tgt, 0.0) + m
                        frame_mass[(tgt, frame_w)]      = frame_mass.get((tgt, frame_w), 0.0) + m
                        if seg_info.cache_start < sink_tokens:
                            sink_set.add(tgt)
                        else:
                            rolling_set.add(tgt)

                    visible_block_ids = sorted(visible_mass.keys())
                    compact_routing   = np.array(
                        [visible_mass[j] for j in visible_block_ids], dtype=np.float64
                    )

                    # Frame-level compact routing (sorted by block then frame)
                    visible_frame_coords = sorted(frame_mass.keys())
                    frame_compact = np.array(
                        [frame_mass[fc] for fc in visible_frame_coords], dtype=np.float64
                    )

                    # Sink / rolling mass fractions
                    total_mass  = compact_routing.sum() + 1e-10
                    sink_mass_v = sum(visible_mass[b] for b in sink_set if b in visible_mass)
                    roll_mass_v = max(total_mass - sink_mass_v - 1e-10, 0.0)

                    # Normalized attention entropy over visible blocks [0, 1]
                    p_blk    = compact_routing / total_mass
                    ent_raw  = -float(np.sum(p_blk * np.log(p_blk + 1e-12)))
                    ent_norm = ent_raw / np.log(max(len(p_blk), 2))

                    # Head agreement: fraction of heads whose argmax maps to the same block
                    head_top_kpos = avg_routing.argmax(axis=1)       # [H] argmax over L_hist
                    head_top_blks = []
                    for _h in range(avg_routing.shape[0]):
                        _si, _ = seg_map.k_index_to_frame(
                            int(head_top_kpos[_h]), current_tokens, max_attn_size
                        )
                        if _si is not None:
                            _ht = bfs_to_idx.get(_si.block_frame_start)
                            head_top_blks.append(_ht)
                    if head_top_blks:
                        from collections import Counter as _Counter
                        head_agreement = _Counter(head_top_blks).most_common(1)[0][1] / len(head_top_blks)
                    else:
                        head_agreement = 0.0

                    routing_by_step.setdefault(block_idx, {}).setdefault(layer_id, []).append({
                        "routing":              compact_routing,
                        "visible_block_ids":    visible_block_ids,
                        "sink_block_ids":       sorted(sink_set),
                        "rolling_block_ids":    sorted(rolling_set & set(visible_block_ids)),
                        # frame-level
                        "frame_routing":        frame_compact,
                        "visible_frame_coords": visible_frame_coords,
                        # scalar analysis
                        "sink_mass_frac":       float(sink_mass_v / total_mass),
                        "rolling_mass_frac":    float(roll_mass_v / total_mass),
                        "attention_entropy":    float(ent_norm),
                        "head_agreement":       float(head_agreement),
                    })

            # KV visibility diagnostics – logged once per probe block
            _pdata = routing_by_step.get(block_idx, {})
            if _pdata:
                _lid0  = min(_pdata.keys())
                _e0    = _pdata[_lid0][-1]
                _vis   = _e0["visible_block_ids"]
                _snk   = _e0["sink_block_ids"]
                _rol   = _e0["rolling_block_ids"]
                _tail  = _rol[-min(6, len(_rol)):]
                print(
                    f"  [Part B] block={block_idx} (frame={bfs})  "
                    f"total_generated={block_idx}  "
                    f"visible={len(_vis)} (sink={len(_snk)}, rolling={len(_rol)})  "
                    f"layers={sorted(_pdata.keys())}"
                )
                if _snk:
                    print(f"           sink_blocks   = {_snk}")
                if _rol:
                    suffix = "..." if len(_rol) > 6 else ""
                    print(f"           rolling_blocks= {_tail}{suffix}")

        # ---------------------------------------------------------------
        # Step 2: context-update forward with real embeddings (commits to cache)
        # ---------------------------------------------------------------
        pred_x0_final = sd[final_step]["pred_x0"].to(device=device, dtype=dtype)
        ctx_ts = (
            torch.ones([pred_x0_final.shape[0], n_frames], device=device, dtype=torch.long)
            * context_noise
        )
        with torch.no_grad():
            generator(
                noisy_image_or_video=pred_x0_final,
                conditional_dict=cond,
                timestep=ctx_ts,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=bfs * frame_seq_len,
            )

        # Store clean frame for future recache windows (CPU to save GPU memory)
        output_frames[bfs] = pred_x0_final.detach().cpu()

        # Update segment map to mirror the cache commit
        seg_map.commit_block(bfs, n_frames, pid)

    # -------------------------------------------------------------------
    # Aggregate metrics from routing_by_step
    # -------------------------------------------------------------------
    metrics = _aggregate_routing_metrics(blocks, routing_by_step, switch_indices)
    lag_profile = _aggregate_lag_profile(blocks, routing_by_step, num_fpb)
    _plot_routing_metrics(metrics, lag_profile, blocks, routing_by_step,
                          switch_indices, output_dir, num_fpb=num_fpb)

    return metrics


def _compute_sparsity_metrics(
    routing: np.ndarray,
    topk: Tuple[int, ...] = (1, 2, 3, 5, 10),
) -> dict:
    """
    All sparsity / concentration metrics for one routing distribution.
    `topk` controls which cumulative-mass thresholds are computed.
    Default (1,2,3,5,10) covers both block-level (max=3) and frame-level (max=9).
    """
    n     = len(routing)
    total = routing.sum() + 1e-10
    p     = routing / total

    g_raw  = gini(routing)
    g_norm = g_raw / (1.0 - 1.0 / n) if n > 1 else 0.0
    n_eff  = float(1.0 / (np.sum(p ** 2) + 1e-10))

    p_desc = np.sort(p)[::-1]
    def _topk(k: int) -> float:
        return float(p_desc[:k].sum()) if k <= n else 1.0

    def _active(thr: float) -> int:
        return int((p > thr).sum())

    return {
        "gini_raw":          g_raw,
        "gini_norm":         g_norm,
        "n_eff":             n_eff,
        **{f"top{k}_mass": _topk(k) for k in topk},
        "total_hist_blocks": n,
        "active_1e-3":       _active(1e-3),
        "active_1e-2":       _active(1e-2),
    }


def _aggregate_routing_metrics(
    blocks: List[dict],
    routing_by_step: Dict[int, Dict[int, List[dict]]],
    switch_indices: List[int],
    # K values calibrated to LongLive: 3 visible history blocks, 9 visible history frames
    block_topk: Tuple[int, ...] = (1, 2, 3, 5, 10),
    frame_topk: Tuple[int, ...] = (1, 3, 5, 9),
) -> dict:
    """
    Aggregate all routing metrics from per-block per-layer routing dicts.

    Block-level top-k: K={1,2,3} are meaningful (max 3 visible history blocks).
    Frame-level top-k: K={1,3,5,9} (9 = all visible history frames: 3 sink + 6 rolling).
    """
    _ = switch_indices

    # ---- block-level lists ----
    stability_list     = []
    sparsity_list      = []
    switch_same_list   = []
    switch_prev_list   = []
    sparsity_norm_list = []
    n_eff_list         = []
    block_topk_lists: Dict[int, list] = {k: [] for k in block_topk}

    # ---- frame-level lists ----
    frame_sparsity_norm_list = []
    frame_n_eff_list         = []
    frame_topk_lists: Dict[int, list] = {k: [] for k in frame_topk}

    # ---- scalar analysis lists ----
    head_agreement_list = []
    sink_frac_list      = []
    rolling_frac_list   = []
    entropy_list        = []

    # ---- KV visibility counts ----
    visible_count_list = []
    sink_count_list    = []
    rolling_count_list = []
    total_gen_list     = []
    active_1e3_list    = []
    active_1e2_list    = []

    for block_idx, layer_dict in routing_by_step.items():
        pid = blocks[block_idx]["prompt_id"]
        for layer_id, step_routings in layer_dict.items():

            # --- stability across denoising steps ---
            if len(step_routings) < 2:
                corr = 1.0
            else:
                corrs = []
                for i in range(len(step_routings) - 1):
                    a_r = step_routings[i]["routing"]
                    b_r = step_routings[i + 1]["routing"]
                    if len(a_r) != len(b_r) or a_r.std() < 1e-8 or b_r.std() < 1e-8:
                        continue
                    corrs.append(float(np.corrcoef(a_r, b_r)[0, 1]))
                corr = float(np.mean(corrs)) if corrs else 1.0

            final_entry   = step_routings[-1]
            final_routing = final_entry["routing"]
            visible_ids   = final_entry["visible_block_ids"]
            sink_ids      = final_entry["sink_block_ids"]
            rolling_ids   = final_entry["rolling_block_ids"]

            # ---- block-level sparsity ----
            sp = _compute_sparsity_metrics(final_routing, topk=block_topk)

            total     = final_routing.sum() + 1e-10
            same_mass = sum(final_routing[i] for i, vid in enumerate(visible_ids)
                            if blocks[vid]["prompt_id"] == pid)
            prev_mass = total - same_mass - 1e-10

            stability_list.append(    (block_idx, layer_id, corr))
            sparsity_list.append(     (block_idx, layer_id, sp["gini_raw"]))
            switch_same_list.append(  (block_idx, layer_id, same_mass / total))
            switch_prev_list.append(  (block_idx, layer_id, prev_mass / total))
            sparsity_norm_list.append((block_idx, layer_id, sp["gini_norm"]))
            n_eff_list.append(        (block_idx, layer_id, sp["n_eff"]))
            for k in block_topk:
                block_topk_lists[k].append((block_idx, layer_id, sp[f"top{k}_mass"]))

            # ---- frame-level sparsity ----
            frame_compact = final_entry.get("frame_routing", np.array([]))
            if len(frame_compact) > 0:
                sp_f = _compute_sparsity_metrics(frame_compact, topk=frame_topk)
                frame_sparsity_norm_list.append((block_idx, layer_id, sp_f["gini_norm"]))
                frame_n_eff_list.append(        (block_idx, layer_id, sp_f["n_eff"]))
                for k in frame_topk:
                    frame_topk_lists[k].append((block_idx, layer_id, sp_f[f"top{k}_mass"]))

            # ---- scalar analysis ----
            head_agreement_list.append((block_idx, layer_id, final_entry.get("head_agreement", 0.0)))
            sink_frac_list.append(     (block_idx, layer_id, final_entry.get("sink_mass_frac", 0.0)))
            rolling_frac_list.append(  (block_idx, layer_id, final_entry.get("rolling_mass_frac", 0.0)))
            entropy_list.append(       (block_idx, layer_id, final_entry.get("attention_entropy", 0.0)))

            # ---- KV visibility counts ----
            visible_count_list.append( (block_idx, layer_id, len(visible_ids)))
            sink_count_list.append(    (block_idx, layer_id, len(sink_ids)))
            rolling_count_list.append( (block_idx, layer_id, len(rolling_ids)))
            total_gen_list.append(     (block_idx, layer_id, block_idx))
            active_1e3_list.append(    (block_idx, layer_id, sp["active_1e-3"]))
            active_1e2_list.append(    (block_idx, layer_id, sp["active_1e-2"]))

    return {
        # backward-compat block-level
        "routing_stability":         stability_list,
        "routing_sparsity":          sparsity_list,
        "routing_same_prompt":       switch_same_list,
        "routing_prev_prompt":       switch_prev_list,
        "routing_sparsity_norm":     sparsity_norm_list,
        "routing_n_eff":             n_eff_list,
        **{f"routing_top{k}_mass": block_topk_lists[k] for k in block_topk},
        # frame-level
        "routing_frame_sparsity_norm": frame_sparsity_norm_list,
        "routing_frame_n_eff":         frame_n_eff_list,
        **{f"routing_frame_top{k}_mass": frame_topk_lists[k] for k in frame_topk},
        # scalar analysis
        "routing_head_agreement":    head_agreement_list,
        "routing_sink_mass_frac":    sink_frac_list,
        "routing_rolling_mass_frac": rolling_frac_list,
        "routing_entropy":           entropy_list,
        # KV visibility
        "routing_visible_block_count":  visible_count_list,
        "routing_sink_block_count":     sink_count_list,
        "routing_rolling_block_count":  rolling_count_list,
        "routing_total_hist_blocks":    total_gen_list,
        "routing_active_1e-3":          active_1e3_list,
        "routing_active_1e-2":          active_1e2_list,
    }


def _aggregate_lag_profile(
    blocks: List[dict],
    routing_by_step: Dict[int, Dict[int, List[dict]]],
    num_fpb: int = 3,
) -> dict:
    """
    Build lag-decay profiles from frame-level routing dicts.

    Block lag:  lag_b = current_block_idx - history_block_idx  (1 = adjacent)
    Frame lag:  lag_f = lag_b * num_fpb - frame_within_block   (1 = most recent)
                For the rolling region only. Sink tracked separately.

    Returns:
        rolling_block_lag_mean: {lag_b: mean_mass_fraction}
        rolling_frame_lag_mean: {lag_f: mean_mass_fraction}
        mean_sink_mass_frac:    float
    """
    block_lag_accum: Dict[int, List[float]] = {}
    frame_lag_accum: Dict[int, List[float]] = {}
    sink_fracs: List[float] = []

    for block_idx, layer_dict in routing_by_step.items():
        for layer_id, step_routings in layer_dict.items():
            final_entry = step_routings[-1]
            sink_ids    = set(final_entry["sink_block_ids"])
            rolling_ids = set(final_entry["rolling_block_ids"])

            # --- block lag profile (rolling only) ---
            final_routing = final_entry["routing"]
            visible_ids   = final_entry["visible_block_ids"]
            total_mass    = final_routing.sum() + 1e-10

            for i, vid in enumerate(visible_ids):
                if vid not in rolling_ids:
                    continue
                lag_b = block_idx - vid
                if lag_b <= 0:
                    continue
                frac = float(final_routing[i]) / total_mass
                block_lag_accum.setdefault(lag_b, []).append(frac)

            # --- frame lag profile (rolling only) ---
            frame_compact = final_entry.get("frame_routing", np.array([]))
            frame_coords  = final_entry.get("visible_frame_coords", [])
            if len(frame_compact) > 0 and len(frame_coords) == len(frame_compact):
                ftotal = frame_compact.sum() + 1e-10
                for fi, (fb_idx, fw) in enumerate(frame_coords):
                    if fb_idx not in rolling_ids:
                        continue
                    lag_b = block_idx - fb_idx
                    if lag_b <= 0:
                        continue
                    # lag=1 = most recent frame in the adjacent block's last slot
                    lag_f = lag_b * num_fpb - fw
                    if lag_f <= 0:
                        lag_f = 1
                    frac_f = float(frame_compact[fi]) / ftotal
                    frame_lag_accum.setdefault(lag_f, []).append(frac_f)

            # --- sink mass fraction ---
            sink_fracs.append(final_entry.get("sink_mass_frac", 0.0))

    rolling_block_lag_mean = {
        lag: float(np.mean(vs)) for lag, vs in sorted(block_lag_accum.items())
    }
    rolling_frame_lag_mean = {
        lag: float(np.mean(vs)) for lag, vs in sorted(frame_lag_accum.items())
    }
    mean_sink_mass_frac = float(np.mean(sink_fracs)) if sink_fracs else 0.0

    return {
        "rolling_block_lag_mean": rolling_block_lag_mean,
        "rolling_frame_lag_mean": rolling_frame_lag_mean,
        "mean_sink_mass_frac":    mean_sink_mass_frac,
    }


# ---------------------------------------------------------------------------
# Part B: plots
# ---------------------------------------------------------------------------

def _plot_scalar_metric(
    data: List[Tuple],   # (block_idx, layer_id, scalar)
    ylabel: str, title: str, fname: str,
    switch_indices: List[int], N: int,
    ylim: Optional[Tuple] = None,
) -> None:
    by_layer: Dict[int, Tuple[List, List]] = {}
    for block_idx, layer_id, val in data:
        by_layer.setdefault(layer_id, ([], []))
        by_layer[layer_id][0].append(block_idx)
        by_layer[layer_id][1].append(val)

    fig, ax = plt.subplots(figsize=(10, 4))
    for layer_id, (xs, ys) in sorted(by_layer.items()):
        ax.plot(xs, ys, "o-", label=f"Layer {layer_id}", markersize=4)
    for i, sw in enumerate(switch_indices):
        ax.axvline(sw, color="red", linestyle=":", alpha=0.6,
                   label="prompt switch" if i == 0 else "")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Block index"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(fontsize=8); ax.set_xlim(0, max(N - 1, 1))
    plt.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"[Part B] Saved {os.path.basename(fname)}")


def _plot_routing_metrics(
    metrics: dict,
    lag_profile: dict,
    blocks: List[dict],
    routing_by_step: Dict,
    switch_indices: List[int],
    output_dir: str,
    num_fpb: int = 3,
) -> None:
    N = len(blocks)

    def _sw_label(i):
        return "prompt switch" if i == 0 else ""

    def _xy(key, lid):
        xs = [bi for bi, li, _ in metrics[key] if li == lid]
        ys = [v  for _,  li, v in metrics[key] if li == lid]
        return xs, ys

    layers_present = sorted({lid for _, lid, _ in metrics.get("routing_visible_block_count", [])})

    # ------------------------------------------------------------------
    # 1) Routing stability
    # ------------------------------------------------------------------
    _plot_scalar_metric(
        metrics["routing_stability"],
        ylabel="Step-to-step routing correlation",
        title="Routing Stability across Denoising Steps",
        fname=os.path.join(output_dir, "routing_stability.png"),
        switch_indices=switch_indices, N=N,
    )

    # ------------------------------------------------------------------
    # 2) Raw Gini
    # ------------------------------------------------------------------
    _plot_scalar_metric(
        metrics["routing_sparsity"],
        ylabel="Gini (raw, over visible KV blocks)",
        title="Routing Sparsity — raw Gini",
        fname=os.path.join(output_dir, "routing_sparsity_raw.png"),
        switch_indices=switch_indices, N=N, ylim=(0, 1),
    )

    # ------------------------------------------------------------------
    # 3) Normalized Gini
    # ------------------------------------------------------------------
    _plot_scalar_metric(
        metrics["routing_sparsity_norm"],
        ylabel="Normalized Gini  G / (1 − 1/n_visible)",
        title="Routing Sparsity — normalized Gini (over visible KV blocks only)",
        fname=os.path.join(output_dir, "routing_sparsity.png"),
        switch_indices=switch_indices, N=N, ylim=(0, 1),
    )

    # ------------------------------------------------------------------
    # 4) Block-level N_eff
    # ------------------------------------------------------------------
    _plot_scalar_metric(
        metrics["routing_n_eff"],
        ylabel="N_eff = 1/Σp²  (visible KV blocks)",
        title="Effective Number of Attended Visible KV Blocks",
        fname=os.path.join(output_dir, "routing_n_eff.png"),
        switch_indices=switch_indices, N=N,
    )

    # ------------------------------------------------------------------
    # 5) KV visibility: total generated vs visible vs sink vs rolling
    # ------------------------------------------------------------------
    for lid in layers_present:
        fig, ax = plt.subplots(figsize=(10, 4))
        txs, tys  = _xy("routing_total_hist_blocks",     lid)
        vxs, vys  = _xy("routing_visible_block_count",   lid)
        sxs, sys_ = _xy("routing_sink_block_count",      lid)
        rxs, rys  = _xy("routing_rolling_block_count",   lid)
        ax.plot(txs, tys,  "k:",  label="total generated",       linewidth=1.5)
        ax.plot(vxs, vys,  "b-",  label="visible (sink+rolling)", linewidth=2)
        ax.plot(sxs, sys_, "g--", label="sink blocks",            linewidth=1.5)
        ax.plot(rxs, rys,  "r--", label="rolling blocks",         linewidth=1.5)
        for i, sw in enumerate(switch_indices):
            ax.axvline(sw, color="purple", linestyle=":", alpha=0.6, label=_sw_label(i))
        ax.set_xlabel("AR block index"); ax.set_ylabel("Number of AR blocks")
        ax.set_title(f"KV Visibility (layer {lid})")
        ax.legend(fontsize=8); ax.set_xlim(0, max(N - 1, 1))
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"routing_kv_visibility_L{lid}.png"), dpi=120)
        plt.close(fig)
        print(f"[Part B] Saved routing_kv_visibility_L{lid}.png")

    # ------------------------------------------------------------------
    # 6) Active KV blocks among visible
    # ------------------------------------------------------------------
    for lid in layers_present:
        fig, ax = plt.subplots(figsize=(10, 4))
        vxs, vys    = _xy("routing_visible_block_count", lid)
        a3_xs, a3_ys = _xy("routing_active_1e-3",        lid)
        a2_xs, a2_ys = _xy("routing_active_1e-2",        lid)
        ax.plot(vxs,   vys,   "b--", label="visible KV blocks",  linewidth=1.5)
        ax.plot(a3_xs, a3_ys, "o-",  label="active (mass>1e-3)", markersize=4)
        ax.plot(a2_xs, a2_ys, "s-",  label="active (mass>1e-2)", markersize=4)
        for i, sw in enumerate(switch_indices):
            ax.axvline(sw, color="purple", linestyle=":", alpha=0.6, label=_sw_label(i))
        ax.set_xlabel("AR block index"); ax.set_ylabel("Number of visible KV blocks")
        ax.set_title(f"Active KV Blocks among Visible History (layer {lid})")
        ax.legend(fontsize=8); ax.set_xlim(0, max(N - 1, 1))
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"routing_kv_utilisation_L{lid}.png"), dpi=120)
        plt.close(fig)
        print(f"[Part B] Saved routing_kv_utilisation_L{lid}.png")

    # ------------------------------------------------------------------
    # 7) Block-level top-k routing coverage
    #    K={1,2,3} meaningful; 3 = 100% coverage → shows convergence point
    # ------------------------------------------------------------------
    block_topk_keys = [k for k in (1, 2, 3, 5, 10)
                       if f"routing_top{k}_mass" in metrics]
    for lid in layers_present:
        fig, ax = plt.subplots(figsize=(10, 4))
        for k in block_topk_keys:
            xs, ys = _xy(f"routing_top{k}_mass", lid)
            ax.plot(xs, ys, "o-", label=f"top-{k} blocks", markersize=4)
        for i, sw in enumerate(switch_indices):
            ax.axvline(sw, color="purple", linestyle=":", alpha=0.6, label=_sw_label(i))
        ax.set_xlabel("AR block index")
        ax.set_ylabel("Cumulative routing mass (visible blocks)")
        ax.set_title(f"Block-level Top-k Routing Coverage (layer {lid})\n"
                     f"max visible history = 3 blocks (1 sink + 2 rolling)")
        ax.legend(fontsize=8); ax.set_xlim(0, max(N - 1, 1)); ax.set_ylim(0, 1)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"routing_topk_L{lid}.png"), dpi=120)
        plt.close(fig)
        print(f"[Part B] Saved routing_topk_L{lid}.png")

    # ------------------------------------------------------------------
    # 8) Frame-level top-k routing coverage
    #    K={1,3,5,9}; 9 = all visible history frames (3 sink + 6 rolling)
    # ------------------------------------------------------------------
    frame_topk_keys = [k for k in (1, 3, 5, 9)
                       if f"routing_frame_top{k}_mass" in metrics
                       and metrics[f"routing_frame_top{k}_mass"]]
    if frame_topk_keys:
        for lid in layers_present:
            fig, ax = plt.subplots(figsize=(10, 4))
            for k in frame_topk_keys:
                xs, ys = _xy(f"routing_frame_top{k}_mass", lid)
                if xs:
                    ax.plot(xs, ys, "o-", label=f"top-{k} frames", markersize=4)
            for i, sw in enumerate(switch_indices):
                ax.axvline(sw, color="purple", linestyle=":", alpha=0.6, label=_sw_label(i))
            ax.set_xlabel("AR block index")
            ax.set_ylabel("Cumulative routing mass (visible frames)")
            ax.set_title(f"Frame-level Top-k Routing Coverage (layer {lid})\n"
                         f"local_attn=12fr, sink=3fr → 9 visible history frames")
            ax.legend(fontsize=8); ax.set_xlim(0, max(N - 1, 1)); ax.set_ylim(0, 1)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"routing_frame_topk_L{lid}.png"), dpi=120)
            plt.close(fig)
            print(f"[Part B] Saved routing_frame_topk_L{lid}.png")

    # ------------------------------------------------------------------
    # 9) Frame-level N_eff
    # ------------------------------------------------------------------
    if metrics.get("routing_frame_n_eff"):
        for lid in layers_present:
            xs, ys = _xy("routing_frame_n_eff", lid)
            if xs:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(xs, ys, "o-", markersize=4)
                ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5, label="N_eff=1 (hard routing)")
                ax.axhline(9.0, color="grey", linestyle=":",  alpha=0.5, label="N_eff=9 (uniform over 9fr)")
                for i, sw in enumerate(switch_indices):
                    ax.axvline(sw, color="purple", linestyle=":", alpha=0.6, label=_sw_label(i))
                ax.set_xlabel("AR block index")
                ax.set_ylabel("Frame N_eff = 1/Σp²")
                ax.set_title(f"Effective Number of Attended Frames (layer {lid})")
                ax.legend(fontsize=8); ax.set_xlim(0, max(N - 1, 1))
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, f"routing_frame_n_eff_L{lid}.png"), dpi=120)
                plt.close(fig)
                print(f"[Part B] Saved routing_frame_n_eff_L{lid}.png")

    # ------------------------------------------------------------------
    # 10) Lag-decay profile (rolling region only; sink annotated)
    # ------------------------------------------------------------------
    if lag_profile:
        blk_lag = lag_profile.get("rolling_block_lag_mean", {})
        frm_lag = lag_profile.get("rolling_frame_lag_mean", {})
        sink_frac = lag_profile.get("mean_sink_mass_frac", 0.0)

        if blk_lag or frm_lag:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            # Block-lag panel
            ax = axes[0]
            if blk_lag:
                lags = sorted(blk_lag.keys())
                vals = [blk_lag[l] for l in lags]
                ax.bar(lags, vals, color="steelblue", alpha=0.8)
            ax.set_xlabel("Block lag (current − history block)")
            ax.set_ylabel("Mean routing mass fraction (rolling only)")
            ax.set_title(f"Block-level Lag-Decay Profile\n"
                         f"mean sink mass = {sink_frac:.3f}")
            ax.text(0.98, 0.95, f"sink={sink_frac:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

            # Frame-lag panel
            ax = axes[1]
            if frm_lag:
                lags_f = sorted(frm_lag.keys())
                vals_f = [frm_lag[l] for l in lags_f]
                ax.bar(lags_f, vals_f, color="darkorange", alpha=0.8)
            ax.set_xlabel(f"Frame lag (lag_f=1 → most recent; {num_fpb} frames/block)")
            ax.set_ylabel("Mean routing mass fraction (rolling only)")
            ax.set_title(f"Frame-level Lag-Decay Profile\n"
                         f"mean sink mass = {sink_frac:.3f}")
            ax.text(0.98, 0.95, f"sink={sink_frac:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, "routing_lag_profile.png"), dpi=120)
            plt.close(fig)
            print("[Part B] Saved routing_lag_profile.png")

    # ------------------------------------------------------------------
    # 11) Sink vs rolling mass fraction per layer
    # ------------------------------------------------------------------
    if metrics.get("routing_sink_mass_frac"):
        for lid in layers_present:
            sxs, sys_ = _xy("routing_sink_mass_frac",    lid)
            rxs, rys  = _xy("routing_rolling_mass_frac", lid)
            if sxs:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(sxs, sys_, "g-",  label="sink mass fraction",    linewidth=2)
                ax.plot(rxs, rys,  "r--", label="rolling mass fraction",  linewidth=2)
                for i, sw in enumerate(switch_indices):
                    ax.axvline(sw, color="purple", linestyle=":", alpha=0.6, label=_sw_label(i))
                ax.set_xlabel("AR block index")
                ax.set_ylabel("Routing mass fraction")
                ax.set_title(f"Sink vs Rolling Attention Mass (layer {lid})\n"
                             f"sink = long-range memory; rolling = recent context")
                ax.legend(fontsize=8); ax.set_xlim(0, max(N - 1, 1)); ax.set_ylim(0, 1)
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, f"routing_sink_vs_rolling_L{lid}.png"), dpi=120)
                plt.close(fig)
                print(f"[Part B] Saved routing_sink_vs_rolling_L{lid}.png")

    # ------------------------------------------------------------------
    # 12) Head agreement per layer
    # ------------------------------------------------------------------
    if metrics.get("routing_head_agreement"):
        for lid in layers_present:
            xs, ys = _xy("routing_head_agreement", lid)
            if xs:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(xs, ys, "o-", color="purple", markersize=4)
                ax.axhline(1.0 / 3, color="grey", linestyle="--", alpha=0.5,
                           label="random (1/3 visible blocks)")
                ax.axhline(1.0,     color="grey", linestyle=":",  alpha=0.5,
                           label="perfect agreement")
                for i, sw in enumerate(switch_indices):
                    ax.axvline(sw, color="purple", linestyle=":", alpha=0.6, label=_sw_label(i))
                ax.set_xlabel("AR block index")
                ax.set_ylabel("Head agreement (fraction of heads → same top block)")
                ax.set_title(f"Routing Head Agreement (layer {lid})\n"
                             f"high = heads agree on which history block to attend")
                ax.legend(fontsize=8); ax.set_xlim(0, max(N - 1, 1)); ax.set_ylim(0, 1)
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, f"routing_head_agreement_L{lid}.png"), dpi=120)
                plt.close(fig)
                print(f"[Part B] Saved routing_head_agreement_L{lid}.png")

    # ------------------------------------------------------------------
    # 13) Normalized attention entropy per layer
    # ------------------------------------------------------------------
    if metrics.get("routing_entropy"):
        for lid in layers_present:
            xs, ys = _xy("routing_entropy", lid)
            if xs:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(xs, ys, "o-", color="darkcyan", markersize=4)
                ax.axhline(0.0, color="grey", linestyle="--", alpha=0.5, label="H=0 (hard routing)")
                ax.axhline(1.0, color="grey", linestyle=":",  alpha=0.5, label="H=1 (uniform)")
                for i, sw in enumerate(switch_indices):
                    ax.axvline(sw, color="purple", linestyle=":", alpha=0.6, label=_sw_label(i))
                ax.set_xlabel("AR block index")
                ax.set_ylabel("Normalized entropy H / log(n_visible)")
                ax.set_title(f"Attention Entropy (layer {lid})\n"
                             f"low = sparse/concentrated routing")
                ax.legend(fontsize=8); ax.set_xlim(0, max(N - 1, 1)); ax.set_ylim(0, 1)
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, f"routing_entropy_L{lid}.png"), dpi=120)
                plt.close(fig)
                print(f"[Part B] Saved routing_entropy_L{lid}.png")

    # ------------------------------------------------------------------
    # 14) Prompt-switch routing mass fractions
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    by_layer: Dict[int, Tuple[List, List, List]] = {}
    for block_idx, layer_id, same in metrics["routing_same_prompt"]:
        by_layer.setdefault(layer_id, ([], [], []))[0].append(block_idx)
        by_layer[layer_id][1].append(same)
    for block_idx, layer_id, prev in metrics["routing_prev_prompt"]:
        by_layer.setdefault(layer_id, ([], [], []))[2].append(prev)
    for layer_id, (bidxs, sames, prevs) in sorted(by_layer.items()):
        ax.plot(bidxs, sames, label=f"L{layer_id} same-prompt", linewidth=1.5)
        ax.plot(bidxs, prevs, "--", label=f"L{layer_id} prev-prompt", linewidth=1.0)
    for i, sw in enumerate(switch_indices):
        ax.axvline(sw, color="purple", linestyle=":", alpha=0.6, label=_sw_label(i))
    ax.set_xlabel("AR block index")
    ax.set_ylabel("Routing mass fraction (among visible KV blocks)")
    ax.set_title("Prompt-switch Routing: same vs previous prompt")
    ax.legend(fontsize=7); ax.set_xlim(0, max(N - 1, 1))
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "prompt_switch_routing.png"), dpi=120)
    plt.close(fig)
    print("[Part B] Saved prompt_switch_routing.png")

    # ------------------------------------------------------------------
    # 15) Routing heatmap — NaN (grey) for evicted blocks
    # ------------------------------------------------------------------
    if routing_by_step:
        probe_blocks = sorted(routing_by_step.keys())
        first_lid = (min(routing_by_step[probe_blocks[0]].keys())
                     if routing_by_step[probe_blocks[0]] else None)
        if first_lid is not None:
            valid = [bi for bi in probe_blocks if routing_by_step[bi].get(first_lid)]
            max_prior = max(valid) if valid else 0
            mat = np.full((len(valid), max_prior), np.nan)
            for row_i, bi in enumerate(valid):
                entry   = routing_by_step[bi][first_lid][-1]
                compact = entry["routing"]
                vis_ids = entry["visible_block_ids"]
                for j, vid in enumerate(vis_ids):
                    if vid < max_prior:
                        mat[row_i, vid] = float(compact[j])

            fig, ax = plt.subplots(figsize=(max(8, max_prior // 4), max(4, len(valid) // 4)))
            cmap = matplotlib.cm.viridis.copy()
            cmap.set_bad(color="lightgrey")
            im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="upper")
            for sw in switch_indices:
                ax.axvline(sw - 0.5, color="red", linestyle="--", alpha=0.7)
            ax.set_xlabel("Prior block index (grey = evicted from KV cache)")
            ax.set_ylabel("Current probe block")
            ax.set_title(
                f"Routing heatmap (layer {first_lid}, final denoising step)\n"
                f"grey = block evicted from rolling window"
            )
            plt.colorbar(im, ax=ax, label="Routing mass")
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, "routing_transfer_coverage.png"), dpi=120)
            plt.close(fig)
            print("[Part B] Saved routing_transfer_coverage.png")


# ---------------------------------------------------------------------------
# Output: metrics CSV and summary JSON
# ---------------------------------------------------------------------------

def _save_metrics(
    part_a: dict, part_b: dict, output_dir: str
) -> None:
    summary = {}

    if part_a:
        summary["part_a"] = {
            "n_timesteps": len(part_a.get("timesteps", [])),
            "cka_gain_final": float(
                part_a["cka_x0_vs_final"][-1] - part_a["cka_noisy_vs_final"][-1]
            ) if part_a.get("cka_x0_vs_final") else None,
            "knn_gain_final": float(
                part_a["knn_x0_vs_final"][-1] - part_a["knn_noisy_vs_final"][-1]
            ) if part_a.get("knn_x0_vs_final") else None,
            "group_cosine_gain": part_a.get("group_cosine_gain", {}),
        }

    def _mean_scalar(lst):
        return float(np.mean([v for _, _, v in lst])) if lst else None

    if part_b:
        summary["part_b"] = {
            "mean_stability":                _mean_scalar(part_b.get("routing_stability", [])),
            "mean_sparsity_raw":             _mean_scalar(part_b.get("routing_sparsity", [])),
            "mean_sparsity_norm":            _mean_scalar(part_b.get("routing_sparsity_norm", [])),
            "mean_n_eff":                    _mean_scalar(part_b.get("routing_n_eff", [])),
            "mean_top1_mass":                _mean_scalar(part_b.get("routing_top1_mass", [])),
            "mean_top5_mass":                _mean_scalar(part_b.get("routing_top5_mass", [])),
            "mean_top10_mass":               _mean_scalar(part_b.get("routing_top10_mass", [])),
            "mean_visible_block_count":      _mean_scalar(part_b.get("routing_visible_block_count", [])),
            "mean_sink_block_count":         _mean_scalar(part_b.get("routing_sink_block_count", [])),
            "mean_rolling_block_count":      _mean_scalar(part_b.get("routing_rolling_block_count", [])),
            "mean_active_blocks_1e-3":       _mean_scalar(part_b.get("routing_active_1e-3", [])),
            "mean_active_blocks_1e-2":       _mean_scalar(part_b.get("routing_active_1e-2", [])),
        }

    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {json_path}")

    # CSV rows — one row per (block, layer, metric)
    csv_path = os.path.join(output_dir, "metrics.csv")
    rows = []

    if part_a:
        for i, t in enumerate(part_a.get("timesteps", [])):
            rows.append({
                "part":       "A",
                "metric":     "cka_vs_final",
                "timestep":   t,
                "noisy":      part_a["cka_noisy_vs_final"][i] if i < len(part_a["cka_noisy_vs_final"]) else "",
                "x0":         part_a["cka_x0_vs_final"][i]    if i < len(part_a["cka_x0_vs_final"]) else "",
            })

    if part_b:
        # scalar metrics — (block, layer, value)
        _b_scalars = [
            ("routing_stability",    "stability"),
            ("routing_sparsity",     "sparsity_raw"),
            ("routing_sparsity_norm","sparsity_norm"),
            ("routing_n_eff",        "n_eff"),
            ("routing_top1_mass",    "top1_mass"),
            ("routing_top3_mass",    "top3_mass"),
            ("routing_top5_mass",    "top5_mass"),
            ("routing_top10_mass",   "top10_mass"),
            ("routing_same_prompt",  "same_prompt_mass"),
            ("routing_prev_prompt",  "prev_prompt_mass"),
            # KV visibility
            ("routing_total_hist_blocks",    "total_hist_blocks_generated"),
            ("routing_visible_block_count",  "visible_block_count"),
            ("routing_sink_block_count",     "sink_block_count"),
            ("routing_rolling_block_count",  "rolling_block_count"),
            ("routing_active_1e-3",  "active_blocks_1e-3"),
            ("routing_active_1e-2",  "active_blocks_1e-2"),
        ]
        for key, metric_name in _b_scalars:
            for block_idx, layer_id, val in part_b.get(key, []):
                rows.append({"part": "B", "metric": metric_name,
                             "block": block_idx, "layer": layer_id, "value": val})

    if rows:
        fieldnames = sorted({k for r in rows for k in r})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir",  required=True,
                   help="Directory containing latents_X_Y.pt and kv_X.pt files")
    p.add_argument("--output_dir", required=True,
                   help="Where to write figures and metric files")
    p.add_argument("--layers",     default="0,15,29",
                   help="Comma-separated transformer layer indices to probe in Part B")
    p.add_argument("--probe_every", type=int, default=1,
                   help="Probe routing every N-th block (Part B)")
    p.add_argument("--config_path", default=None,
                   help="Path to model config YAML (required for Part B)")
    p.add_argument("--generator_ckpt", default=None,
                   help="Path to generator base checkpoint (overrides config.generator_ckpt)")
    p.add_argument("--lora_ckpt", default=None,
                   help="Path to LoRA checkpoint (overrides config.lora_ckpt)")
    p.add_argument("--context_noise", type=int, default=0,
                   help="Timestep used for context-update forward (default 0)")
    p.add_argument("--skip_part_b", action="store_true",
                   help="Skip Part B even if config_path is given")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Input  : {args.input_dir}")
    print(f"Output : {args.output_dir}")

    # Load all block metadata
    blocks = load_all_blocks(args.input_dir)
    print(f"Found {len(blocks)} AR blocks; "
          f"{len(infer_switch_indices(blocks))} prompt switch(es) detected")

    # --- Part A ---
    part_a = run_part_a(blocks, args.output_dir)

    # --- Part B ---
    part_b = {}
    if args.config_path and not args.skip_part_b:
        target_layers = [int(x) for x in args.layers.split(",")]
        print(f"\n[Part B] Target layers: {target_layers}")
        try:
            generator, local_attn_size, sink_size = _load_model(args)
            part_b = run_part_b(
                blocks          = blocks,
                input_dir       = args.input_dir,
                output_dir      = args.output_dir,
                generator       = generator,
                local_attn_size = local_attn_size,
                sink_size       = sink_size,
                target_layers   = target_layers,
                context_noise   = args.context_noise,
                probe_every     = args.probe_every,
            )
        except Exception as exc:
            print(f"[Part B] Failed: {exc}")
            import traceback; traceback.print_exc()
    else:
        if not args.config_path:
            print("\n[Part B] Skipped (no --config_path). Pass --config_path to enable.")
        else:
            print("\n[Part B] Skipped (--skip_part_b).")

    _save_metrics(part_a, part_b, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
