#!/usr/bin/env python3
"""
Offline analysis of saved denoising-trajectory data from LatentCollector.

Schema discovered from probes/latent_collector.py
--------------------------------------------------
latents_X_Y.pt:
    {
        "latents": {
            step_idx (int): {
                "step_idx": int,
                "input_timestep": float,
                "input_before_forward": Tensor[B, F, C, H, W],
                "pred_x0":             Tensor[B, F, C, H, W],
            },
            ...
        }
        OR (single-step legacy format — only final step):
        {
            "step_idx": int,
            "input_timestep": float,
            "input_before_forward": Tensor,
            "pred_x0": Tensor,
        },
        "embed_path": str,
    }

kv_X.pt (ignored by this script):
    { layer_id: {"k_hist", "v_hist", "write_info"} }

NOTE: The current collector code saves only the FINAL denoising step per block
(single-step format). Multi-step probes require updating the collector to save
all steps as a dict keyed by step_idx. This script handles both formats.

Usage:
    python probes/analyze_latent_dynamics.py \
        --input_dir outputs/probes/latent_1 \
        --output_dir outputs/probes/latent_1/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-8


# ---------------------------------------------------------------------------
# Schema detection and block loading
# ---------------------------------------------------------------------------

def _parse_step_record(obj: dict) -> dict:
    """Validate that a step dict has all required keys."""
    required = {"step_idx", "input_timestep", "input_before_forward", "pred_x0"}
    missing = required - set(obj.keys())
    if missing:
        raise ValueError(f"Step record missing keys: {missing}")
    return obj


def load_latent_file(path: str) -> Tuple[List[dict], Optional[str]]:
    """
    Load one latents_X_Y.pt file.

    Returns:
        steps: list of step-dicts sorted by step_idx (ascending = early→final).
        embed_path: str or None.
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    latents = data["latents"]
    embed_path = data.get("embed_path", None)

    if not isinstance(latents, dict):
        raise ValueError(f"{path}: 'latents' is {type(latents)}, expected dict.")

    keys = list(latents.keys())

    # Multi-step format: keys are integers (step indices)
    if keys and isinstance(keys[0], int):
        steps = [_parse_step_record(latents[k]) for k in sorted(latents.keys())]
    # Single-step format: keys are strings like "step_idx", "pred_x0", ...
    elif "step_idx" in latents:
        steps = [_parse_step_record(latents)]
    else:
        raise ValueError(f"{path}: Cannot interpret latents keys: {keys[:8]}")

    return steps, embed_path


def discover_blocks(input_dir: str) -> List[dict]:
    """
    Find all latents_X_Y.pt files, load them, sort by block start index.
    Each returned record has:
        block_start, block_end, steps (list of step-dicts), embed_path, kv_path
    """
    pattern = re.compile(r"^latents_(\d+)_(\d+)\.pt$")
    records = []

    for fname in sorted(os.listdir(input_dir)):
        m = pattern.match(fname)
        if m is None:
            continue
        start, end = int(m.group(1)), int(m.group(2))
        path = os.path.join(input_dir, fname)

        try:
            steps, embed_path = load_latent_file(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load {path}: {exc}") from exc

        kv_candidate = os.path.join(input_dir, f"kv_{start}.pt")
        records.append({
            "block_start": start,
            "block_end": end,
            "steps": steps,
            "embed_path": embed_path,
            "kv_path": kv_candidate if os.path.exists(kv_candidate) else None,
        })

    if not records:
        raise RuntimeError(f"No latents_*.pt files found in: {input_dir}")

    records.sort(key=lambda r: r["block_start"])
    return records


def validate_and_summarize(records: List[dict]) -> dict:
    """
    Check consistency across blocks, print a summary, return a metadata dict.
    The final step is always the one with the largest step_idx.
    """
    step_counts = [len(r["steps"]) for r in records]
    if len(set(step_counts)) > 1:
        warnings.warn(f"Inconsistent step counts across blocks: {sorted(set(step_counts))}")

    n_steps = step_counts[0]
    multi_step = n_steps > 1

    # Denoising timesteps from first block, sorted by step_idx (early → final)
    timesteps = [float(r["steps"][si]["input_timestep"]) for si in range(n_steps)
                 for r in records[:1]]
    step_ids = [int(r["steps"][si]["step_idx"]) for si in range(n_steps)
                for r in records[:1]]

    ex = records[0]["steps"][-1]["pred_x0"]
    latent_shape = list(ex.shape)

    # Parse prompt_id from embed_path (format: prompt{id}_embeds.pt)
    prompt_ids: Dict[int, int] = {}
    for rec in records:
        ep = rec.get("embed_path") or ""
        m = re.search(r"prompt(\d+)_embeds", ep)
        if m:
            prompt_ids[rec["block_start"]] = int(m.group(1))

    info = {
        "n_blocks": len(records),
        "n_steps": n_steps,
        "multi_step": multi_step,
        "timesteps": timesteps,
        "step_ids": step_ids,
        "latent_shape": latent_shape,
        "prompt_ids": prompt_ids,
    }

    print(f"\n{'='*60}")
    print(f"Blocks found:        {info['n_blocks']}")
    print(f"Denoising steps:     {n_steps}  ({'multi-step' if multi_step else 'SINGLE-STEP ONLY'})")
    print(f"Timesteps (early→final): {[f'{t:.1f}' for t in timesteps]}")
    print(f"Latent shape:        {latent_shape}")
    print(f"Prompt IDs found:    {sorted(set(prompt_ids.values())) if prompt_ids else 'none'}")
    if not multi_step:
        print()
        print("  WARNING: Collector saved only the final denoising step per block.")
        print("  Multi-step probes (1-7) require all steps. Update LatentCollector")
        print("  to save self._block_buffer_dict (all steps) instead of only")
        print("  self._block_buffer_dict[current_denoise_timestep_index].")
    print(f"{'='*60}\n")

    return info


# ---------------------------------------------------------------------------
# Compression helpers (deterministic, same for all blocks/steps)
# ---------------------------------------------------------------------------

def spatial_pool_flat(x: torch.Tensor, target_h: int = 8, target_w: int = 14) -> np.ndarray:
    """
    Compress [B, F, C, H, W] → 1-D float32 numpy via adaptive average pooling.
    B is squeezed (assumed 1). Target spatial size: (target_h, target_w).
    Same parameters must be used for every call for geometry to be comparable.
    """
    x = x.float()
    B, F, C, H, W = x.shape
    flat = x.reshape(B * F * C, 1, H, W)
    pooled = torch.nn.functional.adaptive_avg_pool2d(flat, (target_h, target_w))
    return pooled.reshape(B, F, C, target_h, target_w).squeeze(0).numpy().flatten()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)) + EPS
    nb = float(np.linalg.norm(b)) + EPS
    return float(np.dot(a / na, b / nb))


def centered_linear_cka(H1: np.ndarray, H2: np.ndarray) -> float:
    """Linear CKA between two [N, D] matrices via centered Gram matrices."""
    def center_gram(K: np.ndarray) -> np.ndarray:
        n = K.shape[0]
        row_mean = K.mean(axis=1, keepdims=True)
        col_mean = K.mean(axis=0, keepdims=True)
        total_mean = K.mean()
        return K - row_mean - col_mean + total_mean

    K1 = H1 @ H1.T
    K2 = H2 @ H2.T
    cK1, cK2 = center_gram(K1), center_gram(K2)
    num = float(np.sum(cK1 * cK2))
    denom = float(np.sqrt(np.sum(cK1 * cK1) * np.sum(cK2 * cK2))) + EPS
    return num / denom


def topk_energy_fraction(E_flat: np.ndarray, frac: float) -> float:
    """Fraction of total energy in the top `frac` spatial locations."""
    n = max(1, int(np.ceil(frac * len(E_flat))))
    sorted_desc = np.sort(E_flat)[::-1]
    return float(sorted_desc[:n].sum() / (E_flat.sum() + EPS))


# ---------------------------------------------------------------------------
# Prompt-boundary helper for plots
# ---------------------------------------------------------------------------

def mark_prompt_boundaries(ax, records: List[dict], prompt_ids: Dict[int, int]) -> None:
    """Draw vertical lines at prompt_id change points (reference only)."""
    prev_pid = None
    for bi, rec in enumerate(records):
        pid = prompt_ids.get(rec["block_start"])
        if pid is not None and pid != prev_pid and prev_pid is not None:
            ax.axvline(x=bi, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
        prev_pid = pid


# ---------------------------------------------------------------------------
# Build compressed representations for all blocks × steps
# ---------------------------------------------------------------------------

def build_compressed(records: List[dict], n_steps: int) -> np.ndarray:
    """
    Returns H[n_blocks, n_steps, D] using spatial pooling on pred_x0.
    """
    rows = []
    for rec in records:
        row = [spatial_pool_flat(rec["steps"][si]["pred_x0"].float())
               for si in range(n_steps)]
        rows.append(row)
    return np.array(rows, dtype=np.float32)  # [N, T, D]


def build_corrections(records: List[dict], early_si: int) -> np.ndarray:
    """
    Correction d_b = x0_final - x0_early for each block, spatially pooled.
    Returns [N, D].
    """
    result = []
    for rec in records:
        final = rec["steps"][-1]["pred_x0"].float()
        early = rec["steps"][early_si]["pred_x0"].float()
        result.append(spatial_pool_flat(final - early))
    return np.array(result, dtype=np.float32)


# ---------------------------------------------------------------------------
# Probe 1: predicted-x0 convergence to final
# ---------------------------------------------------------------------------

def probe_x0_convergence(records, info, out_dir) -> dict:
    if not info["multi_step"]:
        print("[Probe 1] Skipped — single step only.")
        return {}

    n_blocks, n_steps = info["n_blocks"], info["n_steps"]
    ts_labels = [f"{t:.1f}" for t in info["timesteps"]]
    n_early = n_steps - 1  # all steps except final

    rel_l2 = np.zeros((n_blocks, n_early))
    cos_fin = np.zeros((n_blocks, n_early))

    for bi, rec in enumerate(records):
        final = rec["steps"][-1]["pred_x0"].float().flatten()
        norm_f = float(final.norm()) + EPS
        for si in range(n_early):
            pred = rec["steps"][si]["pred_x0"].float().flatten()
            rel_l2[bi, si] = float((pred - final).norm()) / norm_f
            cos_fin[bi, si] = cosine(pred.numpy(), final.numpy())

    early_labels = ts_labels[:n_early]
    xs = np.arange(n_early)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for bi in range(n_blocks):
        ax1.plot(xs, rel_l2[bi], color="steelblue", alpha=0.18, linewidth=0.6)
    med = np.median(rel_l2, axis=0)
    ax1.plot(xs, med, color="navy", linewidth=2, label="median")
    ax1.fill_between(xs, np.percentile(rel_l2, 10, axis=0),
                     np.percentile(rel_l2, 90, axis=0), alpha=0.22, color="navy", label="p10–p90")
    ax1.set_xticks(xs); ax1.set_xticklabels(early_labels)
    ax1.set_xlabel("Denoising step (timestep)"); ax1.set_ylabel("Relative L2 to final pred_x0")
    ax1.set_title("Clean-prediction convergence (L2)"); ax1.legend(); ax1.grid(True, alpha=0.3)

    for bi in range(n_blocks):
        ax2.plot(xs, cos_fin[bi], color="coral", alpha=0.18, linewidth=0.6)
    med_c = np.median(cos_fin, axis=0)
    ax2.plot(xs, med_c, color="darkred", linewidth=2, label="median")
    ax2.fill_between(xs, np.percentile(cos_fin, 10, axis=0),
                     np.percentile(cos_fin, 90, axis=0), alpha=0.22, color="darkred", label="p10–p90")
    ax2.set_xticks(xs); ax2.set_xticklabels(early_labels)
    ax2.set_xlabel("Denoising step (timestep)"); ax2.set_ylabel("Cosine with final pred_x0")
    ax2.set_title("Clean-prediction convergence (cosine)"); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle("Probe 1: Predicted-x0 convergence to final step")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "x0_convergence.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Heatmap
    fig2, ax = plt.subplots(figsize=(max(6, n_blocks // 3), 3))
    im = ax.imshow(rel_l2.T, aspect="auto", origin="upper", cmap="viridis")
    ax.set_yticks(np.arange(n_early)); ax.set_yticklabels(early_labels)
    ax.set_xlabel("AR block index"); ax.set_ylabel("Denoising timestep")
    ax.set_title("Relative L2 to final pred_x0 per block")
    plt.colorbar(im, ax=ax)
    mark_prompt_boundaries(ax, records, info["prompt_ids"])
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "x0_convergence_map.png"), dpi=120, bbox_inches="tight")
    plt.close(fig2)

    print("  Median relative L2 to final pred_x0:")
    summary = {}
    for si, ts in enumerate(early_labels):
        med_v = float(np.median(rel_l2[:, si]))
        p10_v = float(np.percentile(rel_l2[:, si], 10))
        p90_v = float(np.percentile(rel_l2[:, si], 90))
        print(f"    t={ts}: median={med_v:.4f}  p10={p10_v:.4f}  p90={p90_v:.4f}")
        summary[ts] = {"median_rel_l2": med_v, "p10": p10_v, "p90": p90_v,
                       "median_cosine": float(np.median(cos_fin[:, si]))}

    return {"x0_convergence": summary, "_rel_l2": rel_l2, "_cos_fin": cos_fin}


# ---------------------------------------------------------------------------
# Probe 2: per-step refinement contribution
# ---------------------------------------------------------------------------

def probe_step_refinement(records, info, out_dir) -> dict:
    if not info["multi_step"]:
        print("[Probe 2] Skipped — single step only.")
        return {}

    n_blocks, n_steps = info["n_blocks"], info["n_steps"]
    n_deltas = n_steps - 1
    ts = info["timesteps"]

    delta_norms = np.zeros((n_blocks, n_deltas))
    delta_fracs = np.zeros((n_blocks, n_deltas))

    for bi, rec in enumerate(records):
        norms = []
        for si in range(n_deltas):
            a = rec["steps"][si]["pred_x0"].float().flatten()
            b = rec["steps"][si + 1]["pred_x0"].float().flatten()
            norms.append(float((b - a).norm()))
        total = sum(norms) + EPS
        delta_norms[bi] = norms
        delta_fracs[bi] = [v / total for v in norms]

    delta_labels = [f"Δ{i}\n{ts[i]:.0f}→{ts[i+1]:.0f}" for i in range(n_deltas)]
    xs = np.arange(n_deltas)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for bi in range(n_blocks):
        ax1.plot(xs, delta_norms[bi], color="steelblue", alpha=0.18, linewidth=0.6)
    ax1.plot(xs, np.median(delta_norms, axis=0), color="navy", linewidth=2, label="median")
    ax1.fill_between(xs, np.percentile(delta_norms, 10, axis=0),
                     np.percentile(delta_norms, 90, axis=0), alpha=0.22, color="navy")
    ax1.set_xticks(xs); ax1.set_xticklabels(delta_labels)
    ax1.set_ylabel("||Δx0||"); ax1.set_title("Step delta norm"); ax1.legend(); ax1.grid(True, alpha=0.3)

    for bi in range(n_blocks):
        ax2.plot(xs, delta_fracs[bi], color="coral", alpha=0.18, linewidth=0.6)
    ax2.plot(xs, np.median(delta_fracs, axis=0), color="darkred", linewidth=2, label="median")
    ax2.fill_between(xs, np.percentile(delta_fracs, 10, axis=0),
                     np.percentile(delta_fracs, 90, axis=0), alpha=0.22, color="darkred")
    ax2.set_xticks(xs); ax2.set_xticklabels(delta_labels)
    ax2.set_ylabel("Fraction of total correction"); ax2.set_title("Relative step contribution")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle("Probe 2: Per-step refinement contribution")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "step_refinement_contribution.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("  Median step contribution fractions:")
    summary = {}
    for i in range(n_deltas):
        med_f = float(np.median(delta_fracs[:, i]))
        med_n = float(np.median(delta_norms[:, i]))
        print(f"    Δ{i} ({ts[i]:.0f}→{ts[i+1]:.0f}): frac={med_f:.3f}  norm={med_n:.3f}")
        summary[f"delta_{i}"] = {"median_frac": med_f, "median_norm": med_n}

    return {"step_refinement": summary, "_delta_norms": delta_norms, "_delta_fracs": delta_fracs}


# ---------------------------------------------------------------------------
# Probe 3: denoising path geometry (tortuosity, direction cosines)
# ---------------------------------------------------------------------------

def probe_trajectory_geometry(records, info, out_dir) -> dict:
    if not info["multi_step"]:
        print("[Probe 3] Skipped — single step only.")
        return {}

    n_blocks, n_steps = info["n_blocks"], info["n_steps"]
    tortuosities = np.zeros(n_blocks)
    n_dc_pairs = n_steps - 2  # number of consecutive delta pairs
    dir_cos = np.full((n_blocks, max(1, n_dc_pairs)), np.nan)

    for bi, rec in enumerate(records):
        preds = [rec["steps"][si]["pred_x0"].float().flatten().numpy()
                 for si in range(n_steps)]
        deltas = [preds[i + 1] - preds[i] for i in range(n_steps - 1)]
        seg_norms = [float(np.linalg.norm(d)) for d in deltas]

        L = sum(seg_norms)
        D = float(np.linalg.norm(preds[-1] - preds[0]))
        tortuosities[bi] = L / (D + EPS)

        for pi in range(n_dc_pairs):
            dir_cos[bi, pi] = cosine(deltas[pi], deltas[pi + 1])

    n_panels = 2 if n_dc_pairs > 0 else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    bx = np.arange(n_blocks)
    axes[0].plot(bx, tortuosities, color="steelblue", linewidth=1)
    axes[0].axhline(float(np.median(tortuosities)), color="navy", linewidth=1.5,
                    linestyle="--", label=f"median={np.median(tortuosities):.2f}")
    mark_prompt_boundaries(axes[0], records, info["prompt_ids"])
    axes[0].set_xlabel("AR block"); axes[0].set_ylabel("Tortuosity  L / D")
    axes[0].set_title("Denoising path tortuosity"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    if n_dc_pairs > 0:
        colors = plt.cm.tab10(np.linspace(0, 0.5, n_dc_pairs))
        for pi in range(n_dc_pairs):
            ts = info["timesteps"]
            lbl = f"cos(Δ{pi},Δ{pi+1})  {ts[pi]:.0f}→{ts[pi+2]:.0f}"
            axes[1].plot(bx, dir_cos[:, pi], color=colors[pi], linewidth=1, alpha=0.85, label=lbl)
        mark_prompt_boundaries(axes[1], records, info["prompt_ids"])
        axes[1].set_xlabel("AR block"); axes[1].set_ylabel("Direction cosine")
        axes[1].set_title("Consecutive correction-direction cosines"); axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

    fig.suptitle("Probe 3: Denoising trajectory geometry")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trajectory_geometry.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  Median tortuosity: {np.median(tortuosities):.3f}")
    if n_dc_pairs > 0:
        for pi in range(n_dc_pairs):
            print(f"  Median cos(Δ{pi},Δ{pi+1}): {np.nanmedian(dir_cos[:, pi]):.3f}")

    summary = {"median_tortuosity": float(np.median(tortuosities))}
    for pi in range(n_dc_pairs):
        summary[f"median_dir_cos_{pi}_{pi+1}"] = float(np.nanmedian(dir_cos[:, pi]))

    return {"trajectory_geometry": summary, "_tortuosities": tortuosities}


# ---------------------------------------------------------------------------
# Probe 4: correction Gram matrix (cross-block cosine similarity)
# ---------------------------------------------------------------------------

def probe_correction_gram(records, info, out_dir, early_si: int) -> dict:
    if not info["multi_step"]:
        print("[Probe 4] Skipped — single step only.")
        return {}

    early_ts = info["timesteps"][early_si]
    corr = build_corrections(records, early_si)  # [N, D]
    n = len(corr)

    norms = np.linalg.norm(corr, axis=1, keepdims=True) + EPS
    corr_normed = corr / norms
    G = (corr_normed @ corr_normed.T).astype(np.float32)

    fig, ax = plt.subplots(figsize=(max(5, n // 4), max(4, n // 4)))
    im = ax.imshow(G, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xlabel("AR block"); ax.set_ylabel("AR block")
    ax.set_title(f"Correction direction Gram  (d = x0_final − x0_t{early_ts:.0f})")
    plt.colorbar(im, ax=ax)

    prev_pid = None
    for bi, rec in enumerate(records):
        pid = info["prompt_ids"].get(rec["block_start"])
        if pid is not None and pid != prev_pid and prev_pid is not None:
            ax.axvline(x=bi - 0.5, color="white", linewidth=0.8, alpha=0.6)
            ax.axhline(y=bi - 0.5, color="white", linewidth=0.8, alpha=0.6)
        prev_pid = pid

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "correction_gram.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    triu_idx = np.triu_indices(n, k=1)
    off = G[triu_idx]
    print(f"  Correction Gram off-diagonal:  mean={off.mean():.3f}  std={off.std():.3f}")

    return {
        "correction_gram": {
            "early_timestep": float(early_ts),
            "mean_off_diag_cosine": float(off.mean()),
            "std_off_diag_cosine": float(off.std()),
        },
        "_corr_compressed": corr,
    }


# ---------------------------------------------------------------------------
# Probe 5: correction SVD spectrum
# ---------------------------------------------------------------------------

def probe_correction_spectrum(records, info, out_dir,
                               corr_compressed: Optional[np.ndarray] = None,
                               early_si: int = 1) -> dict:
    if not info["multi_step"]:
        print("[Probe 5] Skipped — single step only.")
        return {}

    if corr_compressed is None:
        corr_compressed = build_corrections(records, early_si)

    D = corr_compressed - corr_compressed.mean(axis=0, keepdims=True)
    print("D shape: ", D.shape)
    _, S, _ = np.linalg.svd(D, full_matrices=False)
    energy = S ** 2
    cum = np.cumsum(energy) / (energy.sum() + EPS)

    def k_for(thresh: float) -> int:
        idx = int(np.searchsorted(cum, thresh))
        return min(idx + 1, len(cum))

    k50, k80, k90, k95 = k_for(0.50), k_for(0.80), k_for(0.90), k_for(0.95)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum, color="steelblue", linewidth=1.5)
    for k, label, col in [(k50, "50%", "orange"), (k80, "80%", "red"),
                           (k90, "90%", "darkred"), (k95, "95%", "purple")]:
        ax.axvline(x=k, color=col, linestyle="--", alpha=0.7, label=f"k={k} ({label})")
    ax.set_xlabel("Number of singular components")
    ax.set_ylabel("Cumulative explained energy")
    ax.set_title("Probe 5: Correction direction SVD spectrum")
    ax.set_xlim(1, len(cum))
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "correction_spectrum.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  SVD: k50={k50}  k80={k80}  k90={k90}  k95={k95}")

    return {"correction_subspace": {"k50": k50, "k80": k80, "k90": k90, "k95": k95}}


# ---------------------------------------------------------------------------
# Probe 6: representation geometry preservation (CKA + KNN)
# ---------------------------------------------------------------------------

def _cosine_matrix(H: np.ndarray) -> np.ndarray:
    """[N, D] → [N, N] cosine similarity (diagonal = 1)."""
    norms = np.linalg.norm(H, axis=1, keepdims=True) + EPS
    N = H / norms
    return (N @ N.T).astype(np.float32)


def _knn_sets(cos_mat: np.ndarray, k: int) -> List[set]:
    """Return top-k nearest neighbor sets (excluding self)."""
    n = cos_mat.shape[0]
    result = []
    for i in range(n):
        row = cos_mat[i].copy()
        row[i] = -2.0  # exclude self
        nbrs = set(np.argsort(row)[::-1][:k].tolist())
        result.append(nbrs)
    return result


def probe_geometry_preservation(records, info, out_dir,
                                  H_all: Optional[np.ndarray] = None) -> dict:
    if not info["multi_step"]:
        print("[Probe 6] Skipped — single step only.")
        return {}

    n_blocks, n_steps = info["n_blocks"], info["n_steps"]
    ts_labels = [f"{t:.1f}" for t in info["timesteps"]]
    n_early = n_steps - 1

    if H_all is None:
        H_all = build_compressed(records, n_steps)  # [N, T, D]

    H_final = H_all[:, -1, :]  # [N, D]
    cos_final = _cosine_matrix(H_final)

    ks = [k for k in [3, 5, 10] if k < n_blocks]
    knn_final = {k: _knn_sets(cos_final, k) for k in ks}

    cka_vals = []
    knn_overlap = {k: [] for k in ks}

    for si in range(n_early):
        H_t = H_all[:, si, :]
        cka_vals.append(centered_linear_cka(H_t, H_final))

        cos_t = _cosine_matrix(H_t)
        for k in ks:
            knn_t = _knn_sets(cos_t, k)
            overlap = [len(knn_t[bi] & knn_final[k][bi]) / k for bi in range(n_blocks)]
            knn_overlap[k].append(float(np.median(overlap)))

    xs = np.arange(n_early)
    early_labels = ts_labels[:n_early]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(xs, cka_vals, "o-", color="steelblue", linewidth=2, markersize=6)
    ax1.set_xticks(xs); ax1.set_xticklabels(early_labels)
    ax1.set_xlabel("Denoising timestep"); ax1.set_ylabel("Linear CKA with final step")
    ax1.set_title("Block-level geometry preservation (CKA)")
    ax1.set_ylim(0, 1.05); ax1.grid(True, alpha=0.3)

    colors = ["steelblue", "coral", "forestgreen"]
    for ki, k in enumerate(ks):
        ax2.plot(xs, knn_overlap[k], "o-", color=colors[ki], linewidth=2,
                 markersize=6, label=f"K={k}")
    ax2.set_xticks(xs); ax2.set_xticklabels(early_labels)
    ax2.set_xlabel("Denoising timestep"); ax2.set_ylabel("Median KNN overlap with final step")
    ax2.set_title("Neighborhood preservation"); ax2.set_ylim(0, 1.05); ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Probe 6: Representation geometry preservation across denoising")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "geometry_preservation.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  CKA to final:")
    cka_summary = {}
    for si, ts in enumerate(early_labels):
        print(f"    t={ts}: CKA={cka_vals[si]:.3f}")
        cka_summary[ts] = float(cka_vals[si])

    knn_summary = {}
    for k in ks:
        knn_summary[f"K{k}"] = {ts: float(v) for ts, v in zip(early_labels, knn_overlap[k])}
        print(f"  KNN-{k} overlap: {dict(zip(early_labels, [f'{v:.3f}' for v in knn_overlap[k]]))}")

    return {"cka_to_final": cka_summary, "knn_preservation": knn_summary, "_H_all": H_all}


# ---------------------------------------------------------------------------
# Probe 7: spatial refinement concentration
# ---------------------------------------------------------------------------

def probe_spatial_refinement(records, info, out_dir) -> dict:
    if not info["multi_step"]:
        print("[Probe 7] Skipped — single step only.")
        return {}

    n_blocks, n_steps = info["n_blocks"], info["n_steps"]
    n_early = n_steps - 1
    ts_labels = [f"{t:.1f}" for t in info["timesteps"][:n_early]]

    top_fracs = {p: np.zeros((n_blocks, n_early)) for p in [0.10, 0.20, 0.30, 0.50]}
    total_energy = np.zeros(n_blocks)

    for bi, rec in enumerate(records):
        final = rec["steps"][-1]["pred_x0"].float()  # [B, F, C, H, W]
        # global energy (vs earliest step)
        early0 = rec["steps"][0]["pred_x0"].float()
        total_energy[bi] = float((early0 - final).norm())

        for si in range(n_early):
            pred = rec["steps"][si]["pred_x0"].float()
            diff = pred - final            # [B, F, C, H, W]
            # E[f, h, w] = sqrt(sum_c diff^2)
            E = diff.pow(2).sum(dim=2).sqrt()   # [B, F, H, W]
            E_flat = E.squeeze(0).flatten().numpy()  # [F*H*W]
            for p in [0.10, 0.20, 0.30, 0.50]:
                top_fracs[p][bi, si] = topk_energy_fraction(E_flat, p)

    xs = np.arange(n_early)
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = ["steelblue", "coral", "forestgreen", "purple"]
    for ci, p in enumerate([0.10, 0.20, 0.30, 0.50]):
        med = np.median(top_fracs[p], axis=0)
        p10 = np.percentile(top_fracs[p], 10, axis=0)
        p90 = np.percentile(top_fracs[p], 90, axis=0)
        ax.plot(xs, med, "o-", color=palette[ci], linewidth=2, label=f"top {int(p*100)}%")
        ax.fill_between(xs, p10, p90, alpha=0.15, color=palette[ci])
        ax.axhline(p, color=palette[ci], linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(ts_labels)
    ax.set_xlabel("Denoising timestep")
    ax.set_ylabel("Fraction of correction energy in top-N% spatial locs")
    ax.set_title("Probe 7: Spatial refinement concentration  (dotted = uniform baseline)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spatial_refinement_concentration.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Example correction-energy maps (latent space, no video decoding)
    si_ex = 0  # earliest step
    idx_candidates = [
        0,
        n_blocks // 2,
        n_blocks - 1,
        int(np.argmax(total_energy)),
        int(np.argmax(top_fracs[0.20][:, 0])),
    ]
    ex_indices = list(dict.fromkeys(i for i in idx_candidates if 0 <= i < n_blocks))[:5]
    n_ex = len(ex_indices)

    ex_label_map = {0: "earliest", n_blocks // 2: "middle", n_blocks - 1: "latest",
                    int(np.argmax(total_energy)): "max_corr",
                    int(np.argmax(top_fracs[0.20][:, 0])): "max_conc"}

    fig2, axes2 = plt.subplots(1, n_ex, figsize=(3.5 * n_ex, 3))
    if n_ex == 1:
        axes2 = [axes2]

    for pi, idx in enumerate(ex_indices):
        rec = records[idx]
        final = rec["steps"][-1]["pred_x0"].float()
        pred_t = rec["steps"][si_ex]["pred_x0"].float()
        diff = pred_t - final
        E = diff.pow(2).sum(dim=2).sqrt().squeeze(0)  # [F, H, W]
        E_mean = E.mean(dim=0).numpy()                 # [H, W]
        im = axes2[pi].imshow(E_mean, cmap="hot", aspect="auto")
        lbl = ex_label_map.get(idx, f"b{idx}")
        axes2[pi].set_title(f"b{records[idx]['block_start']} ({lbl})", fontsize=8)
        axes2[pi].axis("off")
        plt.colorbar(im, ax=axes2[pi])

    fig2.suptitle(f"Correction energy map  t={ts_labels[si_ex]} vs final  (avg over frames)", y=1.01)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "spatial_refinement_examples.png"), dpi=120, bbox_inches="tight")
    plt.close(fig2)

    med_top20_step0 = float(np.median(top_fracs[0.20][:, 0]))
    print(f"  Top-20% spatial concentration at t={ts_labels[0]}: median={med_top20_step0:.3f}  (uniform=0.20)")

    summary = {}
    for si, ts in enumerate(ts_labels):
        summary[ts] = {
            "top10_median": float(np.median(top_fracs[0.10][:, si])),
            "top20_median": float(np.median(top_fracs[0.20][:, si])),
            "top30_median": float(np.median(top_fracs[0.30][:, si])),
        }

    return {"spatial_concentration": summary, "_top_fracs_20": top_fracs[0.20]}


# ---------------------------------------------------------------------------
# CSV + JSON outputs
# ---------------------------------------------------------------------------

def save_block_metrics(records, info, results, out_dir) -> None:
    rows = []
    rel_l2 = results.get("_rel_l2")
    tortuosities = results.get("_tortuosities")
    fracs_20 = results.get("_top_fracs_20")
    delta_norms = results.get("_delta_norms")
    delta_fracs = results.get("_delta_fracs")

    for bi, rec in enumerate(records):
        row: dict = {
            "block_start": rec["block_start"],
            "block_end": rec["block_end"],
            "prompt_id": info["prompt_ids"].get(rec["block_start"], ""),
        }
        if rel_l2 is not None:
            for si in range(rel_l2.shape[1]):
                row[f"rel_l2_step{si}"] = float(rel_l2[bi, si])
        if tortuosities is not None:
            row["tortuosity"] = float(tortuosities[bi])
        if fracs_20 is not None:
            for si in range(fracs_20.shape[1]):
                row[f"top20_frac_step{si}"] = float(fracs_20[bi, si])
        if delta_norms is not None:
            for si in range(delta_norms.shape[1]):
                row[f"delta_norm_step{si}"] = float(delta_norms[bi, si])
        if delta_fracs is not None:
            for si in range(delta_fracs.shape[1]):
                row[f"delta_frac_step{si}"] = float(delta_fracs[bi, si])
        rows.append(row)

    if not rows:
        return
    path = os.path.join(out_dir, "metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def save_summary_json(info, results, out_dir) -> None:
    skip_keys = {k for k in results if k.startswith("_")}
    summary = {
        "num_blocks": info["n_blocks"],
        "num_denoising_steps": info["n_steps"],
        "multi_step_data": info["multi_step"],
        "denoising_timesteps": info["timesteps"],
        "latent_shape": info["latent_shape"],
        "prompt_ids_seen": sorted(set(info["prompt_ids"].values())) if info["prompt_ids"] else [],
    }
    for k, v in results.items():
        if k not in skip_keys:
            summary[k] = v

    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline analysis of saved denoising-trajectory data (no model loading)."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing latents_X_Y.pt files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write figures, metrics.json, metrics.csv")
    parser.add_argument("--early_timestep", type=float, default=833.33,
                        help="Target timestep for correction Gram/SVD (default: 833.33)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records = discover_blocks(args.input_dir)
    info = validate_and_summarize(records)

    # Choose early step index for correction-based probes (closest to early_timestep)
    ts = info["timesteps"]
    early_si = min(range(info["n_steps"] - 1),
                   key=lambda i: abs(ts[i] - args.early_timestep),
                   default=0)
    if info["multi_step"]:
        print(f"Correction probes use early step idx={early_si}  (t={ts[early_si]:.1f}, "
              f"closest to --early_timestep={args.early_timestep})\n")

    results: dict = {}

    print("--- Probe 1: x0 convergence to final ---")
    results.update(probe_x0_convergence(records, info, args.output_dir))

    print("\n--- Probe 2: Per-step refinement contribution ---")
    results.update(probe_step_refinement(records, info, args.output_dir))

    print("\n--- Probe 3: Denoising trajectory geometry ---")
    results.update(probe_trajectory_geometry(records, info, args.output_dir))

    print("\n--- Probe 4: Correction Gram matrix ---")
    results.update(probe_correction_gram(records, info, args.output_dir, early_si=early_si))

    print("\n--- Probe 5: Correction SVD spectrum ---")
    corr_cached = results.get("_corr_compressed")
    results.update(probe_correction_spectrum(records, info, args.output_dir,
                                              corr_compressed=corr_cached, early_si=early_si))

    print("\n--- Probe 6: Representation geometry preservation ---")
    results.update(probe_geometry_preservation(records, info, args.output_dir))

    print("\n--- Probe 7: Spatial refinement concentration ---")
    results.update(probe_spatial_refinement(records, info, args.output_dir))

    print("\n--- Saving outputs ---")
    save_block_metrics(records, info, results, args.output_dir)
    save_summary_json(info, results, args.output_dir)

    print(f"\nDone. All outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
