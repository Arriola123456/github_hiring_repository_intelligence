"""
Visualization utilities for exploratory analysis and model results.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import ensure_dirs, setup_logging, FIGURES_DIR, PROCESSED_DIR, LABELED_DIR

logger = setup_logging("visualization")

sns.set_theme(style="whitegrid", palette="viridis")


def plot_label_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = df["maturity_label"].value_counts()
    counts.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(counts)))
    ax.set_title("Distribution of Maturity Labels")
    ax.set_xlabel("Maturity Level")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "label_distribution.png", dpi=150)
    plt.close()
    logger.info("Saved label distribution plot")


def plot_signal_distributions(df: pd.DataFrame):
    signals = ["stars", "forks", "contributors_count", "commit_count",
               "pr_count", "release_count", "readme_length", "repo_age_days"]
    signals = [s for s in signals if s in df.columns]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, signal in enumerate(signals):
        if i >= len(axes):
            break
        ax = axes[i]
        for label in df["maturity_label"].unique():
            subset = df[df["maturity_label"] == label][signal]
            ax.hist(subset, bins=20, alpha=0.5, label=label)
        ax.set_title(signal)
        ax.set_xlabel(signal)
        ax.legend(fontsize=7)

    plt.suptitle("Signal Distributions by Maturity Level", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "signal_distributions.png", dpi=150)
    plt.close()
    logger.info("Saved signal distributions plot")


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["label_id", "label_confidence"]
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    if len(numeric_cols) < 3:
        logger.warning("Not enough numeric columns for correlation heatmap")
        return

    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()
    logger.info("Saved correlation heatmap")


def plot_stars_vs_contributors(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 8))
    for label in df["maturity_label"].unique():
        subset = df[df["maturity_label"] == label]
        ax.scatter(subset["stars"], subset["contributors_count"],
                   alpha=0.6, label=label, s=40)
    ax.set_xlabel("Stars")
    ax.set_ylabel("Contributors")
    ax.set_title("Stars vs Contributors by Maturity Level")
    ax.legend()
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "stars_vs_contributors.png", dpi=150)
    plt.close()
    logger.info("Saved stars vs contributors scatter plot")


def plot_boxplots_by_label(df: pd.DataFrame):
    signals = ["stars", "commit_count", "contributors_count", "pr_count"]
    signals = [s for s in signals if s in df.columns]

    fig, axes = plt.subplots(1, len(signals), figsize=(5 * len(signals), 6))
    if len(signals) == 1:
        axes = [axes]

    for ax, signal in zip(axes, signals):
        sns.boxplot(data=df, x="maturity_label", y=signal, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title(f"{signal} by Maturity Level")

    plt.suptitle("Signal Boxplots by Maturity Level", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "boxplots_by_label.png", dpi=150)
    plt.close()
    logger.info("Saved boxplots by label")


def generate_all_plots():
    ensure_dirs()

    labeled_path = LABELED_DIR / "labeled_repositories.csv"
    df = pd.read_csv(labeled_path)
    logger.info(f"Loaded {len(df)} labeled repositories for visualization")

    plot_label_distribution(df)
    plot_signal_distributions(df)
    plot_correlation_heatmap(df)
    plot_stars_vs_contributors(df)
    plot_boxplots_by_label(df)

    logger.info("All exploratory plots generated!")


def main():
    generate_all_plots()


if __name__ == "__main__":
    main()
