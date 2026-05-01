#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "value",
    "repeat",
    "seed",
    "best_epoch",
    "best_train_metric",
    "best_val_metric",
    "best_test_metric",
    "best_train_loss",
    "best_val_loss",
    "best_test_loss",
    "search_best_test_metric",
    "search_best_val_loss",
    "time_window",
]

SUMMARY_METRICS = [
    "best_epoch",
    "best_train_metric",
    "best_val_metric",
    "best_test_metric",
    "best_train_loss",
    "best_val_loss",
    "best_test_loss",
    "search_best_test_metric",
    "search_best_val_loss",
]

PARAMETER_LABELS = {
    "time_window": "Snapshot Time Window",
    "lr": "Learning Rate",
    "weight_decay": "Weight Decay",
    "hidden_channels": "Hidden Channels",
    "layers": "Diffusion Layers",
    "dropout": "Dropout",
    "d": "Stalk Width",
    "temporal_d_model": "Temporal Hidden Size",
    "closure_hops": "Closure Hops",
    "temporal_bptt_steps": "BPTT Steps",
    "left_weights": "Left Weights",
    "right_weights": "Right Weights",
    "second_linear": "Second Linear Layer",
    "edge_weights": "Edge Weights",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize saved raw results from temporal_mamba_sensitivity_ablation.ipynb."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("notebooks/artifacts/temporal-mamba-trade-studies"),
        help="Directory containing sensitivity_raw.csv and ablation_raw.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save figures and derived summaries. Defaults to <input-dir>/visualizations.",
    )
    parser.add_argument(
        "--dataset-label",
        type=str,
        default="tgbn-trade",
        help="Dataset label shown in figure titles.",
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default="Temporal Mamba Sheaf",
        help="Model label shown in figure titles.",
    )
    parser.add_argument(
        "--metric-column",
        type=str,
        default="best_test_metric",
        choices=["best_train_metric", "best_val_metric", "best_test_metric"],
        help="Raw metric column to summarize and visualize.",
    )
    return parser.parse_args()


def load_raw_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def summarize_runs(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["study", "parameter", "value", "value_label", "experiment_name"]
    summary = raw_df.groupby(group_cols, dropna=False)[SUMMARY_METRICS].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


def pretty_label(name: str) -> str:
    return PARAMETER_LABELS.get(name, name.replace("_", " ").title())


def metric_axis_label(metric_column: str, metric_name: str) -> str:
    split = metric_column.replace("best_", "").replace("_metric", "").replace("_", " ").title()
    return f"Mean {split} {metric_name.upper()}"


def round_tick_formatter(decimals: int = 3):
    return FuncFormatter(lambda value, pos: f"{value:.{decimals}f}")


def value_sort_key(parameter: str, value_label: str, numeric_value):
    if parameter in {"left_weights", "right_weights", "second_linear", "edge_weights"}:
        order = {"False": 0, "True": 1}
        return order.get(str(value_label), 99)
    if pd.notna(numeric_value):
        return float(numeric_value)
    return str(value_label)


def formatted_value_label(parameter: str, value_label: str, numeric_value):
    if parameter in {"lr", "weight_decay"} and pd.notna(numeric_value):
        return f"{float(numeric_value):.2e}"
    if parameter in {"dropout"} and pd.notna(numeric_value):
        return f"{float(numeric_value):.2f}"
    if parameter == "time_window":
        if pd.isna(numeric_value):
            return "Exact"
        return str(int(numeric_value))
    if pd.notna(numeric_value) and float(numeric_value).is_integer():
        return str(int(numeric_value))
    return str(value_label)


def style_axis(ax, y_label: str):
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(round_tick_formatter(3))
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_sensitivity_parameter(summary_df, parameter, metric_column, metric_name, dataset_label, model_label, output_path: Path):
    metric_mean_col = f"{metric_column}_mean"
    metric_std_col = f"{metric_column}_std"

    param_df = summary_df[summary_df["parameter"] == parameter].copy()
    if param_df.empty:
        return None

    param_df["sort_key"] = [
        value_sort_key(parameter, value_label, value)
        for value_label, value in zip(param_df["value_label"], param_df["value"])
    ]
    param_df = param_df.sort_values(by=["sort_key", "value_label"], kind="stable").reset_index(drop=True)

    x = np.arange(len(param_df))
    y = param_df[metric_mean_col].to_numpy(dtype=float)
    yerr = param_df[metric_std_col].fillna(0.0).to_numpy(dtype=float)
    labels = [
        formatted_value_label(parameter, value_label, value)
        for value_label, value in zip(param_df["value_label"], param_df["value"])
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        color="#1f5aa6",
        ecolor="#7aa6d8",
        elinewidth=1.4,
        linewidth=2.0,
        capsize=4,
        markersize=6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel(pretty_label(parameter))
    style_axis(ax, metric_axis_label(metric_column, metric_name))
    ax.set_title(f"Sensitivity to {pretty_label(parameter)}")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_ablation_bar_plot(summary_df, metric_column, metric_name, dataset_label, model_label, output_path: Path):
    metric_mean_col = f"{metric_column}_mean"
    metric_std_col = f"{metric_column}_std"
    ablation_df = summary_df.copy()
    if ablation_df.empty:
        return None

    flags = list(dict.fromkeys(ablation_df["parameter"].tolist()))
    x = np.arange(len(flags))
    width = 0.34

    false_means = []
    false_stds = []
    true_means = []
    true_stds = []
    for flag in flags:
        flag_df = ablation_df[ablation_df["parameter"] == flag].set_index("value_label")
        false_means.append(flag_df.loc["False", metric_mean_col] if "False" in flag_df.index else np.nan)
        false_stds.append(flag_df.loc["False", metric_std_col] if "False" in flag_df.index else 0.0)
        true_means.append(flag_df.loc["True", metric_mean_col] if "True" in flag_df.index else np.nan)
        true_stds.append(flag_df.loc["True", metric_std_col] if "True" in flag_df.index else 0.0)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(
        x - width / 2,
        false_means,
        width,
        yerr=false_stds,
        color="#cfd8dc",
        edgecolor="#78909c",
        capsize=4,
        label="Disabled",
    )
    ax.bar(
        x + width / 2,
        true_means,
        width,
        yerr=true_stds,
        color="#2e7d32",
        edgecolor="#1b5e20",
        capsize=4,
        label="Enabled",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_label(flag) for flag in flags], rotation=0)
    style_axis(ax, metric_axis_label(metric_column, metric_name))
    ax.set_title("Ablation Study")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_ablation_pivot_table(summary_df, metric_column, output_csv: Path, output_png: Path):
    metric_mean_col = f"{metric_column}_mean"
    metric_std_col = f"{metric_column}_std"

    pivot_mean = summary_df.pivot(index="parameter", columns="value_label", values=metric_mean_col)
    pivot_std = summary_df.pivot(index="parameter", columns="value_label", values=metric_std_col)
    pivot_mean.columns = [str(column) for column in pivot_mean.columns]
    pivot_std.columns = [str(column) for column in pivot_std.columns]
    ordered_columns = [column for column in ["False", "True"] if column in pivot_mean.columns]
    if not ordered_columns:
        ordered_columns = list(pivot_mean.columns)
    pivot_mean = pivot_mean.reindex(columns=ordered_columns)
    pivot_std = pivot_std.reindex(columns=ordered_columns)
    pivot_mean.index = [pretty_label(index) for index in pivot_mean.index]
    pivot_mean.to_csv(output_csv)

    display_df = pivot_mean.copy()
    for column in display_df.columns:
        display_df[column] = [
            f"{mean:.4f} +/- {std:.4f}" if pd.notna(mean) else ""
            for mean, std in zip(pivot_mean[column], pivot_std[column].fillna(0.0))
        ]

    fig_height = max(2.2, 0.6 * (len(display_df.index) + 1))
    fig, ax = plt.subplots(figsize=(6.4, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        rowLabels=display_df.index,
        colLabels=[f"{column}" for column in display_df.columns],
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    ax.set_title("Ablation Mean Test Metric by Flag Value", pad=14)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_csv, output_png


def write_summary_csv(summary_df: pd.DataFrame, output_path: Path):
    pretty_df = summary_df.copy()
    pretty_df["parameter"] = pretty_df["parameter"].map(pretty_label)
    pretty_df.to_csv(output_path, index=False)
    return output_path


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or (input_dir / "visualizations")
    sensitivity_raw_path = input_dir / "sensitivity_raw.csv"
    ablation_raw_path = input_dir / "ablation_raw.csv"

    if not sensitivity_raw_path.exists() and not ablation_raw_path.exists():
        raise FileNotFoundError(
            f"Could not find sensitivity_raw.csv or ablation_raw.csv under {input_dir}."
        )

    generated = []

    if sensitivity_raw_path.exists():
        sensitivity_raw_df = load_raw_results(sensitivity_raw_path)
        sensitivity_summary_df = summarize_runs(sensitivity_raw_df)
        metric_name = sensitivity_raw_df["metric_name"].dropna().iloc[0] if "metric_name" in sensitivity_raw_df else "metric"
        sensitivity_output_dir = output_dir / "sensitivity"
        sensitivity_output_dir.mkdir(parents=True, exist_ok=True)
        generated.append(write_summary_csv(sensitivity_summary_df, sensitivity_output_dir / "sensitivity_summary.csv"))

        for parameter in sensitivity_summary_df["parameter"].drop_duplicates().tolist():
            path = plot_sensitivity_parameter(
                sensitivity_summary_df,
                parameter=parameter,
                metric_column=args.metric_column,
                metric_name=metric_name,
                dataset_label=args.dataset_label,
                model_label=args.model_label,
                output_path=sensitivity_output_dir / f"{parameter}.png",
            )
            if path is not None:
                generated.append(path)

    if ablation_raw_path.exists():
        ablation_raw_df = load_raw_results(ablation_raw_path)
        ablation_summary_df = summarize_runs(ablation_raw_df)
        metric_name = ablation_raw_df["metric_name"].dropna().iloc[0] if "metric_name" in ablation_raw_df else "metric"
        ablation_output_dir = output_dir / "ablation"
        ablation_output_dir.mkdir(parents=True, exist_ok=True)
        generated.append(write_summary_csv(ablation_summary_df, ablation_output_dir / "ablation_summary.csv"))
        path = save_ablation_bar_plot(
            ablation_summary_df,
            metric_column=args.metric_column,
            metric_name=metric_name,
            dataset_label=args.dataset_label,
            model_label=args.model_label,
            output_path=ablation_output_dir / "ablation_bar_plot.png",
        )
        if path is not None:
            generated.append(path)
        pivot_csv, pivot_png = save_ablation_pivot_table(
            ablation_summary_df,
            metric_column=args.metric_column,
            output_csv=ablation_output_dir / "ablation_pivot.csv",
            output_png=ablation_output_dir / "ablation_pivot_table.png",
        )
        generated.extend([pivot_csv, pivot_png])

    print("Generated visualization artifacts:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
