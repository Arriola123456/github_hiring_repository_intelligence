"""
Stage 2a: Data Preprocessing
Clean and normalize repository signals for downstream use.
"""

import pandas as pd
import numpy as np

from src.utils import ensure_dirs, setup_logging, PROCESSED_DIR

logger = setup_logging("preprocessing")


def load_raw_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "repositories.csv"
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} repositories from {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["description"] = df["description"].fillna("")
    df["language"] = df["language"].fillna("Unknown")
    df["license"] = df["license"].fillna("None")
    df["topics"] = df["topics"].fillna("[]")

    numeric_cols = [
        "stars", "forks", "watchers", "contributors_count", "commit_count",
        "commits_per_month", "open_issues", "closed_issues", "pr_count",
        "release_count", "readme_length", "dependency_count",
        "repo_age_days", "days_since_last_push", "description_length", "size_kb",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["readme_has_badges", "readme_has_images", "has_ci_cd"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    df["total_issues"] = df["open_issues"] + df["closed_issues"]
    df["issue_close_rate"] = np.where(
        df["total_issues"] > 0,
        df["closed_issues"] / df["total_issues"],
        0,
    )
    df["stars_per_month"] = np.where(
        df["repo_age_days"] > 30,
        df["stars"] / (df["repo_age_days"] / 30),
        df["stars"],
    )
    df["activity_score"] = (
        df["commits_per_month"] * 0.3
        + df["pr_count"] * 0.2
        + df["release_count"] * 0.15
        + df["contributors_count"] * 0.2
        + df["issue_close_rate"] * 0.15
    )

    return df


def main():
    ensure_dirs()
    df = load_raw_data()
    df = clean_data(df)
    output_path = PROCESSED_DIR / "repositories_clean.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data ({len(df)} rows) to {output_path}")
    return df


if __name__ == "__main__":
    main()
