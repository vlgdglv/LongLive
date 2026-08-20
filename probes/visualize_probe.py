"""
Visualization for LongLive Jacobian probe results.

Reads the JSON file produced by JacobianProbe.finalize() and generates:
  Plot 1: gamma_c vs gamma_x by layer  (one figure per probe time)
  Plot 2: control_ratio heatmap around prompt switch
  Plot 3: historical memory observability (J_o) per cache slot
  Plot 4: memory persistence (J_m) per cache slot  (if available)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Load records
# ---------------------------------------------------------------------------

def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1: gamma_c vs gamma_x by layer
# ---------------------------------------------------------------------------

def plot1_sensitivity_by_layer(records: List[Dict], output_dir: str) -> None:
    """One subplot per unique (relative_to_switch, global_chunk) point."""
    # Group by time point
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        key = f"chunk{r['global_chunk']}_{r['relative_to_switch']}"
        groups[key].append(r)

    fig, axes = plt.subplots(
        1, len(groups), figsize=(5 * max(len(groups), 1), 4), squeeze=False
    )
    axes = axes[0]

    for ax, (key, grp) in zip(axes, sorted(groups.items())):
        layers = sorted(set(r["layer_id"] for r in grp))
        gc = [np.mean([r["gamma_c"] for r in grp if r["layer_id"] == l]) for l in layers]
        gx = [np.mean([r["gamma_x"] for r in grp if r["layer_id"] == l]) for l in layers]

        ax.plot(layers, gc, "o-", label="gamma_c (prompt)", color="tab:orange")
        ax.plot(layers, gx, "s-", label="gamma_x (visual)", color="tab:blue")
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("Transformer layer")
        ax.set_ylabel("Normalized sensitivity")
        ax.legend(fontsize=8)
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Probe A/B: Prompt vs Visual sensitivity of KV writing")
    fig.tight_layout()
    out = os.path.join(output_dir, "plot1_sensitivity_by_layer.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: control_ratio heatmap
# ---------------------------------------------------------------------------

def plot2_control_ratio_heatmap(records: List[Dict], output_dir: str) -> None:
    """Rows = relative_to_switch, Cols = layer."""
    all_rel = sorted(set(r["relative_to_switch"] for r in records),
                     key=lambda s: ["baseline", "pre", "switch", "post1", "post2"].index(s)
                     if s in ["baseline", "pre", "switch", "post1", "post2"] else 99)
    all_layers = sorted(set(r["layer_id"] for r in records))

    # Build matrix: average control_ratio over any repeated measurements
    mat = np.full((len(all_rel), len(all_layers)), np.nan)
    for ri, rel in enumerate(all_rel):
        for li, layer in enumerate(all_layers):
            vals = [r["control_ratio"] for r in records
                    if r["relative_to_switch"] == rel and r["layer_id"] == layer]
            if vals:
                mat[ri, li] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(max(len(all_layers), 3), max(len(all_rel), 2) + 1))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(range(len(all_layers)))
    ax.set_xticklabels(all_layers)
    ax.set_yticks(range(len(all_rel)))
    ax.set_yticklabels(all_rel)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Position relative to switch")
    ax.set_title("control_ratio = gamma_c / (gamma_c + gamma_x)  [0=visual, 1=prompt]")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    out = os.path.join(output_dir, "plot2_control_ratio_heatmap.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: historical memory observability (J_o)
# ---------------------------------------------------------------------------

def plot3_observability(records: List[Dict], output_dir: str) -> None:
    """Combined norm of d y_t / d m_t by historical cache slot."""
    # Pick one representative layer (first probe layer with data)
    has_data = [r for r in records if r["jo_combined_by_slot"]]
    if not has_data:
        print("No J_o data to plot.")
        return

    # Group by relative_to_switch (aggregate over layers by summing energy)
    rel_groups = ["pre", "switch", "post1", "post2", "baseline"]
    colors = {"pre": "tab:blue", "switch": "tab:red", "post1": "tab:green",
               "post2": "tab:purple", "baseline": "tab:gray"}

    fig, ax = plt.subplots(figsize=(10, 4))
    plotted_any = False
    for rel in rel_groups:
        grp = [r for r in has_data if r["relative_to_switch"] == rel]
        if not grp:
            continue
        # Average over layers and chunks for this rel position
        max_len = max(len(r["jo_combined_by_slot"]) for r in grp)
        arr = np.zeros(max_len)
        count = np.zeros(max_len, dtype=int)
        for r in grp:
            sl = r["jo_combined_by_slot"]
            arr[:len(sl)] += np.array(sl)
            count[:len(sl)] += 1
        count = np.maximum(count, 1)
        avg = arr / count
        xs = np.arange(len(avg))
        ax.plot(xs, avg, label=rel, color=colors.get(rel, None), alpha=0.85)
        plotted_any = True

    if plotted_any:
        ax.set_xlabel("Historical cache slot (0 = oldest)")
        ax.set_ylabel("||d y_t / d KV[slot]|| (normalized grad norm)")
        ax.set_title("Probe C: Historical memory observability (J_o)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(output_dir, "plot3_observability.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4: memory persistence (J_m)
# ---------------------------------------------------------------------------

def plot4_persistence(records: List[Dict], output_dir: str) -> None:
    """d KV_new / d historical KV by cache slot."""
    has_data = [r for r in records if r["jm_combined_by_slot"]]
    if not has_data:
        print("No J_m data to plot.")
        return

    rel_groups = ["pre", "switch", "post1", "post2", "baseline"]
    colors = {"pre": "tab:blue", "switch": "tab:red", "post1": "tab:green",
               "post2": "tab:purple", "baseline": "tab:gray"}

    fig, ax = plt.subplots(figsize=(10, 4))
    plotted_any = False
    for rel in rel_groups:
        grp = [r for r in has_data if r["relative_to_switch"] == rel]
        if not grp:
            continue
        max_len = max(len(r["jm_combined_by_slot"]) for r in grp)
        arr = np.zeros(max_len)
        count = np.zeros(max_len, dtype=int)
        for r in grp:
            sl = r["jm_combined_by_slot"]
            arr[:len(sl)] += np.array(sl)
            count[:len(sl)] += 1
        count = np.maximum(count, 1)
        avg = arr / count
        ax.plot(np.arange(len(avg)), avg, label=rel, color=colors.get(rel, None), alpha=0.85)
        plotted_any = True

    if plotted_any:
        ax.set_xlabel("Historical cache slot (0 = oldest)")
        ax.set_ylabel("||d KV_new / d KV[slot]|| (grad norm)")
        ax.set_title("Probe D: Memory persistence (J_m)  [exploratory proxy]")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(output_dir, "plot4_persistence.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Additional: entanglement proxy summary
# ---------------------------------------------------------------------------

def print_entanglement_summary(records: List[Dict]) -> None:
    print("\n--- Entanglement proxy (Pearson r between e_c and e_x per proj) ---")
    print("(NOTE: with M=2 projections this is extremely noisy; treat as qualitative only)")
    layers = sorted(set(r["layer_id"] for r in records))
    for layer in layers:
        vals = [r["entanglement_proxy"] for r in records
                if r["layer_id"] == layer and not np.isnan(r["entanglement_proxy"])]
        if vals:
            print(f"  Layer {layer:3d}: mean_ent={np.nanmean(vals):.3f}  n={len(vals)}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Visualize Jacobian probe results")
    parser.add_argument("--records_path", help="Path to probe_records.json")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save plots (defaults to same dir as records)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.records_path))
    os.makedirs(output_dir, exist_ok=True)

    records = load_records(args.records_path)
    print(f"Loaded {len(records)} probe records from {args.records_path}")

    # plot1_sensitivity_by_layer(records, output_dir)
    # plot2_control_ratio_heatmap(records, output_dir)
    plot3_observability(records, output_dir)
    # plot4_persistence(records, output_dir)
    # print_entanglement_summary(records)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
