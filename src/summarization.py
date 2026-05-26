"""
Stage 2b: Repository Summarization
Convert repository signals into textual representations for LLM labeling and BERT input.
"""

import ast
import pandas as pd

from src.utils import ensure_dirs, setup_logging, PROCESSED_DIR

logger = setup_logging("summarization")


def generate_summary(row: pd.Series) -> str:
    topics = row.get("topics", "[]")
    if isinstance(topics, str):
        try:
            topics = ast.literal_eval(topics)
        except (ValueError, SyntaxError):
            topics = []
    topics_str = ", ".join(topics) if topics else "none"

    parts = [
        f"Repository '{row['name']}' by {row['owner']}.",
        f"Language: {row.get('language', 'Unknown')}. License: {row.get('license', 'None')}.",
        f"Description: {row.get('description', 'No description')}.",
        f"Topics: {topics_str}.",
        f"Stars: {int(row.get('stars', 0))}, Forks: {int(row.get('forks', 0))}, Watchers: {int(row.get('watchers', 0))}.",
        f"Contributors: {int(row.get('contributors_count', 0))}.",
        f"Total commits: {int(row.get('commit_count', 0))}, Commits per month: {row.get('commits_per_month', 0):.1f}.",
        f"Open issues: {int(row.get('open_issues', 0))}, Closed issues: {int(row.get('closed_issues', 0))}.",
        f"Pull requests: {int(row.get('pr_count', 0))}. Releases: {int(row.get('release_count', 0))}.",
        f"README length: {int(row.get('readme_length', 0))} characters.",
        f"Has CI/CD: {row.get('has_ci_cd', False)}. Has README badges: {row.get('readme_has_badges', False)}.",
        f"Dependencies: {int(row.get('dependency_count', 0))}.",
        f"Repository age: {int(row.get('repo_age_days', 0))} days. Days since last push: {int(row.get('days_since_last_push', 0))}.",
        f"Repository size: {int(row.get('size_kb', 0))} KB.",
    ]
    return " ".join(parts)


def main():
    ensure_dirs()
    input_path = PROCESSED_DIR / "repositories_clean.csv"
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} repositories")

    df["summary"] = df.apply(generate_summary, axis=1)

    output_path = PROCESSED_DIR / "repo_summaries.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved summaries to {output_path}")
    logger.info(f"Average summary length: {df['summary'].str.len().mean():.0f} characters")

    return df


if __name__ == "__main__":
    main()
