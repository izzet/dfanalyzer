"""Eval 2: Bottleneck accuracy — hybrid vs full-trace analysis.

Compares bottleneck severity scores produced by dfdiagnoser when:
  (a) DFAnalyzer processes full normal traces (dft-normal)  [ground truth]
  (b) DFAnalyzer processes hybrid aggregated traces (dft-agg-selective)

Metrics:
  - Score agreement rate: % of (metric, entity) cells where severity matches
  - Rank correlation: Spearman rank correlation of severity scores
  - Per-metric accuracy: agreement breakdown by metric type
  - Score distance: mean absolute difference in severity levels

Usage:
  pytest benchmarks/eval_accuracy.py -v -s
"""

import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from dfdiagnoser.scoring import score_metrics

from conftest import (
    DFT_AGG_SELECTIVE_DIR,
    DFT_NORMAL_DIR,
    make_analyzer,
    requires_data,
    subset_trace_dir,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_score_columns(df):
    """Return list of score columns (those ending with _score)."""
    return sorted([c for c in df.columns if c.endswith("_score")])


def _get_dimension_columns(df):
    """Return list of dimension columns (those starting with d_)."""
    return sorted([c for c in df.columns if c.startswith("d_")])


def _run_analysis(dataset_dir, dask_client, tmp_path, label, n_files=None):
    """Run full analysis pipeline and return scored flat views.

    Uses n_files to control subset size (falls back to BENCH_N_FILES env var).
    """
    trace_dir = subset_trace_dir(dataset_dir, tmp_path, label, n_files=n_files)
    analyzer = make_analyzer(tmp_path / label, time_granularity=5, checkpoint=True)

    result = analyzer.analyze_trace(
        trace_path=trace_dir,
        view_types=["proc_name", "time_range"],
    )

    scored_views = {}
    for view_key, flat_view in result.flat_views.items():
        scored = score_metrics(flat_view, metric_boundaries={})
        scored_views[view_key] = scored

    return result, scored_views


def _align_scored_views(normal_scored, hybrid_scored):
    """Align two scored DataFrames on shared dimension columns.

    Returns (aligned_normal, aligned_hybrid) with matching rows.
    """
    dim_cols_n = _get_dimension_columns(normal_scored)
    dim_cols_h = _get_dimension_columns(hybrid_scored)
    dim_cols = sorted(set(dim_cols_n) & set(dim_cols_h))

    if not dim_cols:
        # No shared dimensions — merge on index
        merged = normal_scored.merge(
            hybrid_scored,
            left_index=True,
            right_index=True,
            how="inner",
            suffixes=("_normal", "_hybrid"),
        )
        return merged

    # Inner join on dimension columns
    merged = normal_scored.merge(
        hybrid_scored,
        on=dim_cols,
        how="inner",
        suffixes=("_normal", "_hybrid"),
    )
    return merged


def _compute_agreement(merged, score_cols_normal, score_cols_hybrid):
    """Compute per-metric and overall agreement statistics."""
    results = []

    for col_n, col_h in zip(score_cols_normal, score_cols_hybrid):
        # Extract the base metric name
        metric_name = col_n.replace("_score_normal", "").replace("_score", "")

        s_n = merged[col_n].dropna()
        s_h = merged[col_h].dropna()
        common_idx = s_n.index.intersection(s_h.index)

        if len(common_idx) == 0:
            continue

        vals_n = s_n.loc[common_idx].values.astype(float)
        vals_h = s_h.loc[common_idx].values.astype(float)

        exact_match = np.sum(vals_n == vals_h)
        within_1 = np.sum(np.abs(vals_n - vals_h) <= 1)
        total = len(common_idx)
        mae = np.mean(np.abs(vals_n - vals_h))

        # Spearman rank correlation (handle constant arrays)
        if np.std(vals_n) > 0 and np.std(vals_h) > 0:
            rho, p_val = scipy_stats.spearmanr(vals_n, vals_h)
        else:
            rho, p_val = np.nan, np.nan

        # Bottleneck detection: severity > 0
        n_detected = int(np.sum(vals_n > 0))
        h_detected = int(np.sum(vals_h > 0))
        both_detected = int(np.sum((vals_n > 0) & (vals_h > 0)))

        results.append({
            "metric": metric_name,
            "total_cells": total,
            "exact_agreement": exact_match,
            "exact_agreement_pct": exact_match / total * 100,
            "within_1_agreement": within_1,
            "within_1_pct": within_1 / total * 100,
            "mae": mae,
            "spearman_rho": rho,
            "spearman_p": p_val,
            "normal_bottlenecks": n_detected,
            "hybrid_bottlenecks": h_detected,
            "shared_bottlenecks": both_detected,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main accuracy evaluation
# ---------------------------------------------------------------------------

BENCH_SCALES = [int(x) for x in os.environ.get("BENCH_SCALES", "4,8,16").split(",")]


@requires_data
def test_bottleneck_accuracy_normal_vs_hybrid(dask_client, tmp_path):
    """Compare bottleneck severity rankings: dft-normal vs dft-agg-selective.

    Runs at multiple scales (controlled by BENCH_SCALES env var, default "4,8,16")
    and produces a combined summary table.
    """

    print("\n\n=== Eval 2: Bottleneck Accuracy ===\n")

    output_dir = tmp_path / "eval_accuracy_results"
    output_dir.mkdir(exist_ok=True)
    all_summary_rows = []

    for n_files in BENCH_SCALES:
        print(f"\n{'='*60}")
        print(f"  Scale: n_files={n_files}")
        print(f"{'='*60}")

        label_n = f"normal_n{n_files}"
        label_h = f"hybrid_n{n_files}"

        print(f"  Analyzing dft-normal (ground truth, {n_files} files)...")
        normal_result, normal_scored = _run_analysis(
            DFT_NORMAL_DIR, dask_client, tmp_path, label_n, n_files=n_files
        )

        print(f"  Analyzing dft-agg-selective (hybrid, {n_files} files)...")
        hybrid_result, hybrid_scored = _run_analysis(
            DFT_AGG_SELECTIVE_DIR, dask_client, tmp_path, label_h, n_files=n_files
        )

        # Print raw stats comparison
        print(f"\n  --- Event Counts ---")
        print(f"  Normal:  {int(normal_result.raw_stats.trace_event_count):>10,} trace + "
              f"{int(normal_result.raw_stats.profile_event_count):>10,} profile = "
              f"{int(normal_result.raw_stats.total_event_count):>10,} total")
        print(f"  Hybrid:  {int(hybrid_result.raw_stats.trace_event_count):>10,} trace + "
              f"{int(hybrid_result.raw_stats.profile_event_count):>10,} profile = "
              f"{int(hybrid_result.raw_stats.total_event_count):>10,} total")

        # Compare flat views
        shared_keys = set(normal_scored.keys()) & set(hybrid_scored.keys())
        all_agreements = []

        for view_key in sorted(shared_keys):
            n_df = normal_scored[view_key]
            h_df = hybrid_scored[view_key]

            merged = _align_scored_views(n_df, h_df)

            all_cols = merged.columns.tolist()
            score_cols_n = sorted([c for c in all_cols if c.endswith("_score_normal")])
            score_cols_h = sorted([c for c in all_cols if c.endswith("_score_hybrid")])

            if not score_cols_n or not score_cols_h:
                continue

            agreement = _compute_agreement(merged, score_cols_n, score_cols_h)
            if agreement.empty:
                continue

            agreement["view_key"] = str(view_key)
            all_agreements.append(agreement)

            view_label = ",".join(view_key)
            overall_exact = agreement["exact_agreement"].sum() / max(agreement["total_cells"].sum(), 1) * 100
            overall_within1 = agreement["within_1_agreement"].sum() / max(agreement["total_cells"].sum(), 1) * 100
            overall_mae = agreement["mae"].mean()

            print(f"\n  --- View: {view_label} ---")
            print(f"  Rows aligned: {len(merged)}")
            print(f"  Exact agreement:   {overall_exact:.1f}%")
            print(f"  Within-1 agreement: {overall_within1:.1f}%")
            print(f"  MAE:               {overall_mae:.3f}")

        if not all_agreements:
            continue

        full_df = pd.concat(all_agreements, ignore_index=True)

        # Per-metric summary
        metric_summary = full_df.groupby("metric").agg(
            total_cells=("total_cells", "sum"),
            exact_pct=("exact_agreement_pct", "mean"),
            within_1_pct=("within_1_pct", "mean"),
            mae=("mae", "mean"),
            spearman_rho=("spearman_rho", "mean"),
        ).sort_values("exact_pct", ascending=True)

        print(f"\n  === Per-Metric Summary ===")
        print(metric_summary.to_string())

        # Save per-scale details
        full_df.to_csv(output_dir / f"detailed_agreement_n{n_files}.csv", index=False)
        metric_summary.to_csv(output_dir / f"metric_summary_n{n_files}.csv")

        # Grand totals
        grand_exact = full_df["exact_agreement"].sum() / max(full_df["total_cells"].sum(), 1) * 100
        grand_within1 = full_df["within_1_agreement"].sum() / max(full_df["total_cells"].sum(), 1) * 100
        grand_mae = full_df["mae"].mean()

        total_normal = int(full_df["normal_bottlenecks"].sum())
        total_hybrid = int(full_df["hybrid_bottlenecks"].sum())
        total_shared = int(full_df["shared_bottlenecks"].sum())
        recall = total_shared / max(total_normal, 1) * 100
        precision = total_shared / max(total_hybrid, 1) * 100

        print(f"\n  GRAND TOTAL exact: {grand_exact:.1f}%  within-1: {grand_within1:.1f}%  MAE: {grand_mae:.3f}")
        print(f"  Bottlenecks — normal: {total_normal:,}  hybrid: {total_hybrid:,}  "
              f"shared: {total_shared:,}  recall: {recall:.1f}%  precision: {precision:.1f}%")

        all_summary_rows.append({
            "n_files": n_files,
            "normal_events": int(normal_result.raw_stats.total_event_count),
            "hybrid_events": int(hybrid_result.raw_stats.total_event_count),
            "total_cells": int(full_df["total_cells"].sum()),
            "normal_bottlenecks": total_normal,
            "hybrid_bottlenecks": total_hybrid,
            "shared_bottlenecks": total_shared,
            "recall_pct": round(recall, 1),
            "precision_pct": round(precision, 1),
            "exact_agreement_pct": round(grand_exact, 1),
            "within_1_agreement_pct": round(grand_within1, 1),
            "mae": round(grand_mae, 3),
        })

    # Combined summary across all scales
    if all_summary_rows:
        summary_df = pd.DataFrame(all_summary_rows)
        summary_df.to_csv(output_dir / "bottleneck_summary.csv", index=False)

        print(f"\n\n{'='*60}")
        print(f"  COMBINED SUMMARY (all scales)")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))
        print(f"\n  Summary saved to: {output_dir / 'bottleneck_summary.csv'}")


# ---------------------------------------------------------------------------
# Score distribution comparison
# ---------------------------------------------------------------------------

@requires_data
def test_score_distribution_comparison(dask_client, tmp_path):
    """Compare the distribution of severity scores between normal and hybrid."""

    print("\n\n=== Score Distribution Comparison ===\n")

    n_files = int(os.environ.get("BENCH_N_FILES", "4"))
    _, normal_scored = _run_analysis(DFT_NORMAL_DIR, dask_client, tmp_path, "normal_dist", n_files=n_files)
    _, hybrid_scored = _run_analysis(DFT_AGG_SELECTIVE_DIR, dask_client, tmp_path, "hybrid_dist", n_files=n_files)

    shared_keys = set(normal_scored.keys()) & set(hybrid_scored.keys())

    for view_key in sorted(shared_keys):
        n_df = normal_scored[view_key]
        h_df = hybrid_scored[view_key]

        n_score_cols = _get_score_columns(n_df)
        h_score_cols = _get_score_columns(h_df)
        common_scores = sorted(set(n_score_cols) & set(h_score_cols))

        if not common_scores:
            continue

        view_label = ",".join(view_key)
        print(f"  --- View: {view_label} ---")

        for col in common_scores:
            n_dist = n_df[col].dropna().value_counts().sort_index()
            h_dist = h_df[col].dropna().value_counts().sort_index()

            # Align on all possible scores 0-5
            all_scores = range(6)
            n_counts = [n_dist.get(s, 0) for s in all_scores]
            h_counts = [h_dist.get(s, 0) for s in all_scores]

            metric_name = col.replace("_score", "")
            print(f"    {metric_name:40s} | Normal: {n_counts} | Hybrid: {h_counts}")


# ---------------------------------------------------------------------------
# Layer-level accuracy
# ---------------------------------------------------------------------------

@requires_data
def test_layer_level_accuracy(dask_client, tmp_path):
    """Compare HLM count/time/size totals per layer between normal and hybrid."""

    print("\n\n=== Layer-Level HLM Accuracy ===\n")

    n_files = int(os.environ.get("BENCH_N_FILES", "4"))
    normal_result, _ = _run_analysis(DFT_NORMAL_DIR, dask_client, tmp_path, "normal_layer", n_files=n_files)
    hybrid_result, _ = _run_analysis(DFT_AGG_SELECTIVE_DIR, dask_client, tmp_path, "hybrid_layer", n_files=n_files)

    shared_layers = set(normal_result.layers) & set(hybrid_result.layers)
    print(f"  Shared layers: {sorted(shared_layers)}")

    rows = []
    for layer in sorted(shared_layers):
        try:
            n_hlm = normal_result.get_hlm(layer).compute()
            h_hlm = hybrid_result.get_hlm(layer).compute()
        except KeyError:
            continue

        for metric in ["count", "time", "size"]:
            if metric not in n_hlm.columns or metric not in h_hlm.columns:
                continue
            n_total = n_hlm[metric].sum()
            h_total = h_hlm[metric].sum()
            if n_total == 0:
                rel_error = 0.0 if h_total == 0 else float("inf")
            else:
                rel_error = abs(h_total - n_total) / abs(n_total)

            rows.append({
                "layer": layer,
                "metric": metric,
                "normal_total": n_total,
                "hybrid_total": h_total,
                "abs_diff": abs(h_total - n_total),
                "rel_error": rel_error,
            })

            print(f"  {layer:25s} {metric:6s} | "
                  f"Normal: {n_total:>15,.1f} | Hybrid: {h_total:>15,.1f} | "
                  f"RelErr: {rel_error:.4f}")

    if rows:
        summary = pd.DataFrame(rows)
        output_dir = tmp_path / "eval_accuracy_results"
        output_dir.mkdir(exist_ok=True)
        summary.to_csv(output_dir / "layer_accuracy.csv", index=False)
        print(f"\n  Results saved to: {output_dir / 'layer_accuracy.csv'}")
