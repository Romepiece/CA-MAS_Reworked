from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches as mpatches
from matplotlib.lines import Line2D

from fire_core import OUTPUTS_DIR


REPORTS_DIR = OUTPUTS_DIR / "reports"
LEGACY_STUDY_BBOX = (-121.5, -120.5, 39.5, 40.5)


def ensure_report_dirs(base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else REPORTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    return root


def hybrid_metrics_frame(hybrid_run: dict[str, object]) -> pd.DataFrame:
    metrics = hybrid_run["hybrid_result"]["metrics"]
    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame(asdict(metric) for metric in metrics)


def ca_metrics_frame(ca_run: dict[str, object]) -> pd.DataFrame:
    metrics = ca_run["metrics"]
    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame(asdict(metric) for metric in metrics)


def build_comparison_metrics_frame(hybrid_run: dict[str, object], ca_run: dict[str, object] | None = None) -> pd.DataFrame:
    hybrid_df = hybrid_metrics_frame(hybrid_run)
    if hybrid_df.empty:
        return hybrid_df
    if ca_run is None:
        return hybrid_df

    ca_df = ca_metrics_frame(ca_run)
    if ca_df.empty:
        return hybrid_df

    ca_df = ca_df.rename(
        columns={
            "iou": "iou_ca",
            "precision": "precision_ca",
            "recall": "recall_ca",
            "f1": "f1_ca",
            "dice": "dice_ca",
        }
    )
    merged = hybrid_df.drop(columns=["iou_ca", "precision_ca", "recall_ca", "f1_ca"], errors="ignore").merge(
        ca_df[["date", "iou_ca", "precision_ca", "recall_ca", "f1_ca", "dice_ca"]],
        on="date",
        how="inner",
    )
    merged["iou_improvement"] = merged["iou_hybrid"] - merged["iou_ca"]
    merged["precision_improvement"] = merged["precision_hybrid"] - merged["precision_ca"]
    merged["recall_improvement"] = merged["recall_hybrid"] - merged["recall_ca"]
    merged["f1_improvement"] = merged["f1_hybrid"] - merged["f1_ca"]
    if "dice" not in merged.columns and "dice_ca" in merged.columns:
        merged["dice"] = merged["dice_ca"]
    return merged


def build_daily_comparison_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=["date", "model", "iou", "precision", "recall", "f1", "dice", "frontier_iou", "frontier_f1", "iou_improvement", "frontier_iou_improvement", "frontier_f1_improvement"])

    ca_daily = pd.DataFrame(
        {
            "date": metrics_df["date"],
            "model": "CA (Baseline)",
            "iou": metrics_df["iou_ca"],
            "precision": metrics_df["precision_ca"],
            "recall": metrics_df["recall_ca"],
            "f1": metrics_df["f1_ca"],
            "dice": metrics_df["dice_ca"],
            "frontier_iou": metrics_df.get("frontier_iou_ca", pd.Series(np.nan, index=metrics_df.index)),
            "frontier_f1": metrics_df.get("frontier_f1_ca", pd.Series(np.nan, index=metrics_df.index)),
            "iou_improvement": 0.0,
            "frontier_iou_improvement": 0.0,
            "frontier_f1_improvement": 0.0,
        }
    )
    hybrid_daily = pd.DataFrame(
        {
            "date": metrics_df["date"],
            "model": "CA+MAS Drones",
            "iou": metrics_df["iou_hybrid"],
            "precision": metrics_df["precision_hybrid"],
            "recall": metrics_df["recall_hybrid"],
            "f1": metrics_df["f1_hybrid"],
            "dice": metrics_df["dice"],
            "frontier_iou": metrics_df.get("frontier_iou_hybrid", pd.Series(np.nan, index=metrics_df.index)),
            "frontier_f1": metrics_df.get("frontier_f1_hybrid", pd.Series(np.nan, index=metrics_df.index)),
            "iou_improvement": metrics_df["iou_improvement"],
            "frontier_iou_improvement": metrics_df.get("frontier_iou_improvement", pd.Series(np.nan, index=metrics_df.index)),
            "frontier_f1_improvement": metrics_df.get("frontier_f1_improvement", pd.Series(np.nan, index=metrics_df.index)),
        }
    )
    return pd.concat([ca_daily, hybrid_daily], ignore_index=True)


def build_summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=["model", "mean_iou", "mean_precision", "mean_recall", "mean_f1", "mean_dice", "mean_frontier_iou", "mean_frontier_f1", "mean_growth_iou", "mean_growth_f1"])

    ca_row = {
        "model": "CA (Baseline)",
        "mean_iou": float(metrics_df["iou_ca"].mean()),
        "mean_precision": float(metrics_df["precision_ca"].mean()),
        "mean_recall": float(metrics_df["recall_ca"].mean()),
        "mean_f1": float(metrics_df["f1_ca"].mean()),
        "mean_dice": float(metrics_df["dice_ca"].mean()),
        "mean_frontier_iou": float(metrics_df["frontier_iou_ca"].mean()) if "frontier_iou_ca" in metrics_df else np.nan,
        "mean_frontier_f1": float(metrics_df["frontier_f1_ca"].mean()) if "frontier_f1_ca" in metrics_df else np.nan,
        "mean_growth_iou": float(metrics_df["growth_iou_ca"].mean()) if "growth_iou_ca" in metrics_df else np.nan,
        "mean_growth_f1": float(metrics_df["growth_f1_ca"].mean()) if "growth_f1_ca" in metrics_df else np.nan,
    }
    hybrid_row = {
        "model": "CA+MAS Drones",
        "mean_iou": float(metrics_df["iou_hybrid"].mean()),
        "mean_precision": float(metrics_df["precision_hybrid"].mean()),
        "mean_recall": float(metrics_df["recall_hybrid"].mean()),
        "mean_f1": float(metrics_df["f1_hybrid"].mean()),
        "mean_dice": float(metrics_df["dice"].mean()),
        "mean_frontier_iou": float(metrics_df["frontier_iou_hybrid"].mean()) if "frontier_iou_hybrid" in metrics_df else np.nan,
        "mean_frontier_f1": float(metrics_df["frontier_f1_hybrid"].mean()) if "frontier_f1_hybrid" in metrics_df else np.nan,
        "mean_growth_iou": float(metrics_df["growth_iou_hybrid"].mean()) if "growth_iou_hybrid" in metrics_df else np.nan,
        "mean_growth_f1": float(metrics_df["growth_f1_hybrid"].mean()) if "growth_f1_hybrid" in metrics_df else np.nan,
    }
    improvement_row = {
        "model": "Improvement (CA+MAS - CA)",
        "mean_iou": hybrid_row["mean_iou"] - ca_row["mean_iou"],
        "mean_precision": hybrid_row["mean_precision"] - ca_row["mean_precision"],
        "mean_recall": hybrid_row["mean_recall"] - ca_row["mean_recall"],
        "mean_f1": hybrid_row["mean_f1"] - ca_row["mean_f1"],
        "mean_dice": hybrid_row["mean_dice"] - ca_row["mean_dice"],
        "mean_frontier_iou": hybrid_row["mean_frontier_iou"] - ca_row["mean_frontier_iou"],
        "mean_frontier_f1": hybrid_row["mean_frontier_f1"] - ca_row["mean_frontier_f1"],
        "mean_growth_iou": hybrid_row["mean_growth_iou"] - ca_row["mean_growth_iou"],
        "mean_growth_f1": hybrid_row["mean_growth_f1"] - ca_row["mean_growth_f1"],
    }
    return pd.DataFrame([ca_row, hybrid_row, improvement_row])


def build_observability_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "targeting_mode",
                "cluster_count",
                "assigned_cluster_count",
                "selected_cluster_mean_pairwise_distance",
                "selected_cluster_min_pairwise_distance",
                "selected_macro_sector_count",
                "selected_connected_component_count",
                "drone_spatial_dispersion",
                "mean_cluster_size_pixels",
                "waypoints_total",
                "mean_waypoints_per_drone",
                "observed_cluster_pixels",
                "observed_cluster_coverage_ratio",
                "growth_waypoint_ratio",
                "frontier_waypoint_ratio",
                "unobserved_waypoint_ratio",
                "observed_growth_pixels",
                "growth_coverage_ratio",
                "true_growth_pixels",
                "raw_predicted_growth_pixels",
                "filtered_growth_target_pixels",
                "frontier_candidate_pixels",
                "selected_target_pixels",
                "assigned_target_pixels",
                "predicted_growth_ca_pixels",
                "predicted_growth_hybrid_pixels",
                "growth_iou_ca",
                "growth_iou_hybrid",
                "growth_iou_improvement",
                "growth_f1_ca",
                "growth_f1_hybrid",
                "growth_f1_improvement",
                "frontier_pixels",
                "observed_frontier_pixels",
                "frontier_coverage_ratio",
                "observed_footprint_pixels",
                "observed_burned_pixels",
                "observed_clear_pixels",
                "corrected_to_burned_pixels",
                "corrected_to_clear_pixels",
                "corrected_total_pixels",
            ]
        )

    columns = [
        "date",
        "targeting_mode",
        "cluster_count",
        "assigned_cluster_count",
        "selected_cluster_mean_pairwise_distance",
        "selected_cluster_min_pairwise_distance",
        "selected_macro_sector_count",
        "selected_connected_component_count",
        "drone_spatial_dispersion",
        "mean_cluster_size_pixels",
        "waypoints_total",
        "mean_waypoints_per_drone",
        "observed_cluster_pixels",
        "observed_cluster_coverage_ratio",
        "growth_waypoint_ratio",
        "frontier_waypoint_ratio",
        "unobserved_waypoint_ratio",
        "observed_growth_pixels",
        "growth_coverage_ratio",
        "true_growth_pixels",
        "raw_predicted_growth_pixels",
        "filtered_growth_target_pixels",
        "frontier_candidate_pixels",
        "selected_target_pixels",
        "assigned_target_pixels",
        "predicted_growth_ca_pixels",
        "predicted_growth_hybrid_pixels",
        "growth_iou_ca",
        "growth_iou_hybrid",
        "growth_iou_improvement",
        "growth_f1_ca",
        "growth_f1_hybrid",
        "growth_f1_improvement",
        "frontier_pixels",
        "observed_frontier_pixels",
        "frontier_coverage_ratio",
        "observed_footprint_pixels",
        "observed_burned_pixels",
        "observed_clear_pixels",
        "corrected_to_burned_pixels",
        "corrected_to_clear_pixels",
        "corrected_total_pixels",
    ]
    return metrics_df[columns].copy()


def build_assignment_table(hybrid_run: dict[str, object]) -> pd.DataFrame:
    logs = hybrid_run["hybrid_result"]["assign_logs"]
    if not logs:
        return pd.DataFrame(columns=["date", "total_distance", "mean_distance", "num_pairs"])
    return pd.DataFrame(logs)


def _table_output_paths(tables_dir: Path) -> dict[str, Path]:
    return {
        "daily": tables_dir / "all_models_daily.csv",
        "summary": tables_dir / "all_models_summary.csv",
        "observability": tables_dir / "hybrid_observability_daily.csv",
        "assignments": tables_dir / "hungarian_assignment_summary.csv",
        "wide_metrics": tables_dir / "hybrid_metrics_wide.csv",
    }


def save_tables(hybrid_run: dict[str, object], output_dir: str | Path | None = None, ca_run: dict[str, object] | None = None) -> dict[str, str]:
    report_dir = ensure_report_dirs(output_dir)
    tables_dir = report_dir / "tables"
    metrics_df = build_comparison_metrics_frame(hybrid_run, ca_run=ca_run)
    daily_df = build_daily_comparison_table(metrics_df)
    summary_df = build_summary_table(metrics_df)
    observability_df = build_observability_table(metrics_df)
    assignment_df = build_assignment_table(hybrid_run)
    output_paths = _table_output_paths(tables_dir)

    daily_df.to_csv(output_paths["daily"], index=False)
    summary_df.to_csv(output_paths["summary"], index=False)
    observability_df.to_csv(output_paths["observability"], index=False)
    assignment_df.to_csv(output_paths["assignments"], index=False)
    metrics_df.to_csv(output_paths["wide_metrics"], index=False)

    return {name: str(path) for name, path in output_paths.items()}


def _finalize_axis(ax, ylabel: str, title: str | None = None) -> None:
    ax.set_xlabel("Date", fontsize=11, fontname="Times New Roman")
    ax.set_ylabel(ylabel, fontsize=11, fontname="Times New Roman")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", fontname="Times New Roman")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim([0, 1])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=9, loc="best", frameon=True, fancybox=True)


def _grid_to_bbox_coordinates(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    min_lon, max_lon, min_lat, max_lat = LEGACY_STUDY_BBOX
    height, width = shape
    lon = min_lon + (cols / max(width - 1, 1)) * (max_lon - min_lon)
    lat = min_lat + (rows / max(height - 1, 1)) * (max_lat - min_lat)
    return lon, lat


def plot_temporal_metrics(metrics_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    axes = axes.flatten()
    dates = metrics_df["date"].tolist()

    plots = [
        ("iou_ca", "iou_hybrid", "IoU", "(a) Temporal IoU Comparison"),
        ("precision_ca", "precision_hybrid", "Precision", "(b) Temporal Precision Comparison"),
        ("recall_ca", "recall_hybrid", "Recall", "(c) Temporal Recall Comparison"),
        ("f1_ca", "f1_hybrid", "F1 Score", "(d) Temporal F1 Comparison"),
    ]
    for ax, (ca_col, hybrid_col, ylabel, title) in zip(axes, plots, strict=True):
        ax.plot(dates, metrics_df[ca_col], "o-", color="#E74C3C", linewidth=2.5, markersize=7, label="CA (Baseline)", alpha=0.8)
        ax.plot(dates, metrics_df[hybrid_col], "s-", color="#27AE60", linewidth=2.5, markersize=7, label="CA+MAS Drones", alpha=0.8)
        ax.fill_between(dates, metrics_df[ca_col], metrics_df[hybrid_col], where=(metrics_df[hybrid_col] >= metrics_df[ca_col]), color="green", alpha=0.2)
        _finalize_axis(ax, ylabel, title)

    plt.tight_layout()
    output_path = figures_dir / "figure_temporal_metrics.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_iou_distribution(metrics_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    box = ax.boxplot(
        [metrics_df["iou_ca"], metrics_df["iou_hybrid"]],
        tick_labels=["CA\n(Baseline)", "CA+MAS\nDrones"],
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanline=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
        meanprops=dict(color="green", linewidth=2, linestyle="--"),
    )
    for patch, color in zip(box["boxes"], ["#E74C3C", "#27AE60"], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Daily IoU", fontsize=11, fontname="Times New Roman")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim([0, 1])
    plt.tight_layout()
    output_path = figures_dir / "figure_iou_distribution.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_daily_iou_improvement(metrics_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    improvement_daily = metrics_df["iou_improvement"]
    colors = ["#27AE60" if value >= 0 else "#E74C3C" for value in improvement_daily]
    ax.bar(range(len(improvement_daily)), improvement_daily, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    mean_improvement = float(improvement_daily.mean())
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(mean_improvement, color="blue", linewidth=1.5, linestyle="--", label=f"Mean: {mean_improvement:.4f}")
    ax.set_xlabel("Date", fontsize=11, fontname="Times New Roman")
    ax.set_ylabel("IoU Improvement (CA+MAS - CA)", fontsize=11, fontname="Times New Roman")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xticks(range(len(improvement_daily)))
    ax.set_xticklabels(metrics_df["date"].tolist(), rotation=45, ha="right")
    plt.tight_layout()
    output_path = figures_dir / "figure_daily_iou_improvement.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_mean_metrics(metrics_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ca_values = [metrics_df["iou_ca"].mean(), metrics_df["precision_ca"].mean(), metrics_df["recall_ca"].mean(), metrics_df["f1_ca"].mean()]
    hybrid_values = [metrics_df["iou_hybrid"].mean(), metrics_df["precision_hybrid"].mean(), metrics_df["recall_hybrid"].mean(), metrics_df["f1_hybrid"].mean()]
    x = np.arange(4)
    width = 0.35
    bars1 = ax.bar(x - width / 2, ca_values, width, label="CA (Baseline)", color="#E74C3C", alpha=0.8, edgecolor="black", linewidth=1.5)
    bars2 = ax.bar(x + width / 2, hybrid_values, width, label="CA+MAS Drones", color="#27AE60", alpha=0.8, edgecolor="black", linewidth=1.5)
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["IoU", "Precision", "Recall", "F1"], fontname="Times New Roman")
    ax.set_ylabel("Mean Score", fontsize=11, fontname="Times New Roman")
    ax.set_ylim([0, 1.05])
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    output_path = figures_dir / "figure_mean_metrics.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _draw_drone(ax, x: float, y: float, color: str, size: float = 8) -> None:
    body = mpatches.Rectangle((x - size / 4, y - size / 4), size / 2, size / 2, facecolor=color, edgecolor="black", linewidth=1.5, zorder=15)
    ax.add_patch(body)
    propeller_offset = size / 2.5
    propeller_size = size / 4
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        propeller = mpatches.Circle((x + dx * propeller_offset, y + dy * propeller_offset), propeller_size, facecolor=color, edgecolor="black", linewidth=1, alpha=0.8, zorder=14)
        ax.add_patch(propeller)


def _select_representative_trajectory_indices(total_results: int, max_panels: int = 3) -> list[int]:
    if total_results <= 0:
        return []

    panel_count = min(max_panels, total_results)
    if panel_count == total_results:
        return list(range(total_results))

    # Prefer a representative middle-to-late window rather than the earliest days,
    # which tend to be visually less informative in the short verified experiment.
    start_idx = min(max(1, (total_results - panel_count) // 2), total_results - panel_count)
    return list(range(start_idx, start_idx + panel_count))


def plot_drone_trajectories(hybrid_run: dict[str, object], figures_dir: Path, num_days_to_show: int = 6) -> list[Path]:
    results = hybrid_run["hybrid_result"]["results"]
    if not results:
        return []
    static_arrays = hybrid_run["dataset"]["static_arrays"]
    burned_series = hybrid_run["dataset"]["burned_series"]
    dem = static_arrays["dem"]
    dem_norm = ((dem - dem.min()) / (dem.max() - dem.min() + 1e-9)).astype(np.float32)
    dem_rgb = (plt.cm.terrain(dem_norm) * 255).astype(np.uint8)
    extent = [LEGACY_STUDY_BBOX[0], LEGACY_STUDY_BBOX[1], LEGACY_STUDY_BBOX[2], LEGACY_STUDY_BBOX[3]]
    selected_indices = _select_representative_trajectory_indices(len(results), max_panels=min(3, num_days_to_show))
    if not selected_indices:
        return []

    fig = plt.figure(figsize=(14.6, 6.9), dpi=170)
    grid = fig.add_gridspec(2, len(selected_indices), height_ratios=[15, 2.2], hspace=0.18, wspace=0.12)
    panel_axes = [fig.add_subplot(grid[0, i]) for i in range(len(selected_indices))]
    legend_ax = fig.add_subplot(grid[1, :])

    for ax, idx in zip(panel_axes, selected_indices, strict=True):
        result = results[idx]
        burned_true = burned_series[idx + 1]
        ax.imshow(dem_rgb, origin="lower", extent=extent)
        fire_rgba = np.zeros((*burned_true.shape, 4), dtype=np.uint8)
        fire_mask = burned_true > 0
        fire_rgba[fire_mask] = [255, 77, 0, 153]
        ax.imshow(fire_rgba, origin="lower", extent=extent)

        for drone_state in result["drone_states"]:
            path = drone_state["daily_path"]
            if not path:
                continue
            path_arr = np.asarray(path, dtype=float)
            lon_path, lat_path = _grid_to_bbox_coordinates(path_arr[:, 0], path_arr[:, 1], burned_true.shape)
            ax.plot(lon_path, lat_path, color=drone_state["color"], linewidth=3.1, alpha=0.95, linestyle="--", marker="o", markersize=4.8, markerfacecolor=drone_state["color"], markeredgecolor="black", markeredgewidth=0.6, zorder=10)
            _draw_drone(ax, lon_path[-1], lat_path[-1], drone_state["color"], size=0.018)

        ax.set_xlabel("Longitude", fontsize=11, fontname="Times New Roman")
        ax.set_ylabel("Latitude", fontsize=11, fontname="Times New Roman")
        ax.set_title(result["date"], fontsize=12, fontweight="bold", fontname="Times New Roman")
        ax.grid(True, alpha=0.2, linestyle=":")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")

    dem_sm = plt.cm.ScalarMappable(cmap="terrain", norm=plt.Normalize(vmin=float(dem.min()), vmax=float(dem.max())))
    dem_sm.set_array([])
    legend_ax.axis("off")
    colorbar_ax = legend_ax.inset_axes([0.1, 0.08, 0.8, 0.26])
    colorbar = fig.colorbar(dem_sm, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label("Elevation (relative units)", fontsize=10, fontname="Times New Roman")

    legend_handles = [
        mpatches.Patch(facecolor=(1.0, 0.3, 0.0, 0.6), edgecolor="black", label="Observed burned cells"),
    ]
    for drone_state in results[selected_indices[-1]]["drone_states"]:
        legend_handles.append(Line2D([0], [0], color=drone_state["color"], linestyle="--", marker="o", markersize=6, linewidth=3.0, label=f"{drone_state['id']} trajectory"))
    legend_handles.append(Line2D([0], [0], color="black", marker="s", linestyle="None", markersize=7, label="Current drone position"))
    legend_ax.legend(handles=legend_handles, loc="center", bbox_to_anchor=(0.5, 0.72), ncol=min(5, len(legend_handles)), frameon=True, fontsize=9)
    fig.suptitle("Representative simulated drone trajectories during the mid-fire period", fontsize=13, fontweight="bold", fontname="Times New Roman")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.08, wspace=0.12, hspace=0.16)
    output_path = figures_dir / "figure_drone_trajectory_composite.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return [output_path]


def plot_spatial_comparison(hybrid_run: dict[str, object], figures_dir: Path, ca_run: dict[str, object] | None = None) -> Path | None:
    if ca_run is None:
        return None

    hybrid_results = hybrid_run["hybrid_result"]["results"]
    if not hybrid_results:
        return None

    static_arrays = hybrid_run["dataset"]["static_arrays"]
    burned_dates = hybrid_run["dataset"]["burned_dates"]
    burned_series = hybrid_run["dataset"]["burned_series"]
    dem = static_arrays["dem"]
    dem_norm = ((dem - dem.min()) / (dem.max() - dem.min() + 1e-9)).astype(np.float32)
    dem_gray = plt.cm.gray(dem_norm)
    extent = [LEGACY_STUDY_BBOX[0], LEGACY_STUDY_BBOX[1], LEGACY_STUDY_BBOX[2], LEGACY_STUDY_BBOX[3]]

    final_result = hybrid_results[-1]
    final_metrics = final_result.get("metrics")
    target_date = final_result["date"]
    target_index = burned_dates.index(target_date)
    observed = burned_series[target_index]
    hybrid_map = final_result["corrected_fire"]

    ca_dates = burned_dates[burned_dates.index(ca_run["start_date"]) + 1 : burned_dates.index(ca_run["end_date"]) + 1]
    if target_date not in ca_dates:
        return None
    ca_index = ca_dates.index(target_date)
    ca_map = ca_run["predictions"][ca_index + 1]
    iou_ca = float(getattr(final_metrics, "iou_ca", np.nan))
    iou_hybrid = float(getattr(final_metrics, "iou_hybrid", np.nan))
    f1_ca = float(getattr(final_metrics, "f1_ca", np.nan))
    f1_hybrid = float(getattr(final_metrics, "f1_hybrid", np.nan))
    delta_iou = iou_hybrid - iou_ca if np.isfinite(iou_ca) and np.isfinite(iou_hybrid) else np.nan
    delta_f1 = f1_hybrid - f1_ca if np.isfinite(f1_ca) and np.isfinite(f1_hybrid) else np.nan

    panels = [
        ("(a) Observed burned area", observed),
        ("(b) CA baseline forecast", ca_map),
        ("(c) CA+MAS hybrid forecast", hybrid_map),
    ]

    fig = plt.figure(figsize=(15.0, 6.3), dpi=150)
    grid = fig.add_gridspec(2, 3, height_ratios=[14, 2.4], hspace=0.16, wspace=0.12)
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    legend_ax = fig.add_subplot(grid[1, :])

    panel_border_colors = ["#2f2f2f", "#b85450", "#4c9f50"]
    panel_metric_text = [
        None,
        f"Final day\nIoU = {iou_ca:.3f}\nF1 = {f1_ca:.3f}",
        f"Final day\nIoU = {iou_hybrid:.3f}\nF1 = {f1_hybrid:.3f}\nΔIoU = {delta_iou:+.3f}\nΔF1 = {delta_f1:+.3f}",
    ]

    for ax, (title, fire_map), border_color, metric_text in zip(axes, panels, panel_border_colors, panel_metric_text, strict=True):
        ax.imshow(dem_gray, origin="lower", extent=extent)
        overlay = np.zeros((*fire_map.shape, 4), dtype=np.uint8)
        overlay[np.asarray(fire_map) > 0] = [255, 120, 0, 190]
        ax.imshow(overlay, origin="lower", extent=extent)
        ax.set_title(title, fontsize=11, fontweight="bold", fontname="Times New Roman")
        ax.set_xlabel("Longitude", fontsize=10, fontname="Times New Roman")
        ax.set_ylabel("Latitude", fontsize=10, fontname="Times New Roman")
        ax.grid(True, alpha=0.15, linestyle=":")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
            spine.set_edgecolor(border_color)
        if metric_text is not None:
            ax.text(
                0.03,
                0.05,
                metric_text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9.5,
                fontname="Times New Roman",
                color="black",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor=border_color, linewidth=1.0),
            )

    fig.suptitle(f"Spatial comparison for {target_date}", fontsize=12, fontweight="bold", fontname="Times New Roman")
    legend_handles = [mpatches.Patch(facecolor=(1.0, 0.47, 0.0, 0.75), edgecolor="black", label="Burned cells")]
    legend_ax.axis("off")
    legend_ax.legend(handles=legend_handles, loc="center", bbox_to_anchor=(0.2, 0.62), ncol=1, frameon=True, fontsize=10)
    legend_ax.text(0.54, 0.62, f"Final-day metrics for {target_date}: ΔIoU = {delta_iou:+.3f}, ΔF1 = {delta_f1:+.3f}", ha="center", va="center", fontsize=10, fontname="Times New Roman", color="#2f2f2f", transform=legend_ax.transAxes)
    legend_ax.text(0.87, 0.62, "Red frame = baseline CA\nGreen frame = CA+MAS hybrid", ha="center", va="center", fontsize=10, fontname="Times New Roman", color="#2f2f2f", transform=legend_ax.transAxes)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.9, bottom=0.08, wspace=0.12, hspace=0.14)
    output_path = figures_dir / "figure_spatial_comparison.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_hybrid_report(hybrid_run: dict[str, object], output_dir: str | Path | None = None, num_days_to_show: int = 6, ca_run: dict[str, object] | None = None) -> dict[str, object]:
    report_dir = ensure_report_dirs(output_dir)
    figures_dir = report_dir / "figures"
    metrics_df = build_comparison_metrics_frame(hybrid_run, ca_run=ca_run)

    table_paths = save_tables(hybrid_run, report_dir, ca_run=ca_run)
    if metrics_df.empty:
        return {"tables": table_paths, "figures": []}

    figure_paths = [
        str(plot_temporal_metrics(metrics_df, figures_dir)),
        str(plot_mean_metrics(metrics_df, figures_dir)),
        str(plot_iou_distribution(metrics_df, figures_dir)),
        str(plot_daily_iou_improvement(metrics_df, figures_dir)),
    ]
    spatial_path = plot_spatial_comparison(hybrid_run, figures_dir, ca_run=ca_run)
    if spatial_path is not None:
        figure_paths.append(str(spatial_path))
    figure_paths.extend(str(path) for path in plot_drone_trajectories(hybrid_run, figures_dir, num_days_to_show=num_days_to_show))
    return {"tables": table_paths, "figures": figure_paths}