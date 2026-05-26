"""
Stage 3: Weak Labeling with Open-Source LLMs
Uses HuggingFace Inference API with free open-source models
to classify repositories by engineering maturity level.
"""

import os
import time
import json
import pandas as pd
from huggingface_hub import InferenceClient
from tqdm import tqdm

from src.utils import (
    ensure_dirs, setup_logging, PROCESSED_DIR, LABELED_DIR, MATURITY_LABELS
)

logger = setup_logging("llm_labeling")

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

SYSTEM_PROMPT = """You are an expert software engineering recruiter and technical assessor.
Your task is to classify GitHub repositories by the engineering maturity level they represent.

Categories:
- intern: Very simple projects (hello world, basic calculators, tutorials, homework assignments). Minimal commits, no CI/CD, few contributors, small codebase.
- junior: Functional but straightforward projects (simple web apps, basic CRUD, simple bots, scripts). Some structure but limited complexity.
- senior: Well-engineered projects with good practices (ML pipelines, data platforms, complex APIs). Multiple contributors, CI/CD, releases, good documentation.
- lead_architect: Framework-level or infrastructure projects. High complexity, many contributors, extensive documentation, robust CI/CD, SDK/library design patterns.
- template: Boilerplate, starter, cookiecutter, or scaffold projects. Designed to be copied, not used directly. Often have "template" or "boilerplate" in name/description.
- low_value: Abandoned, empty, or very low activity repositories. Very old with no recent updates, minimal content, or forked without modification.

Respond with ONLY a JSON object: {"label": "<category>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}
"""


def classify_repo(client: InferenceClient, summary: str) -> dict:
    prompt = f"""Classify this GitHub repository into one of these categories: {', '.join(MATURITY_LABELS)}

Repository information:
{summary}

Respond with ONLY a JSON object: {{"label": "<category>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

    try:
        response = client.text_generation(
            prompt=f"[INST] {SYSTEM_PROMPT}\n\n{prompt} [/INST]",
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
        )

        response_text = response.strip()

        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end > start:
            result = json.loads(response_text[start:end])
            if result.get("label") in MATURITY_LABELS:
                return result

        return {
            "label": "low_value",
            "confidence": 0.3,
            "reasoning": f"Could not parse LLM response: {response_text[:100]}",
        }

    except Exception as e:
        logger.warning(f"LLM classification error: {e}")
        return {
            "label": "low_value",
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)[:100]}",
        }


def fallback_rule_based(row: pd.Series) -> dict:
    """Rule-based fallback when LLM API is unavailable."""
    name = str(row.get("name", "")).lower()
    desc = str(row.get("description", "")).lower()
    combined = f"{name} {desc}"

    if any(kw in combined for kw in ["template", "boilerplate", "starter", "scaffold", "cookiecutter"]):
        return {"label": "template", "confidence": 0.8, "reasoning": "Name/description indicates template project"}

    stars = row.get("stars", 0)
    contributors = row.get("contributors_count", 0)
    commits = row.get("commit_count", 0)
    has_ci = row.get("has_ci_cd", False)
    pr_count = row.get("pr_count", 0)
    releases = row.get("release_count", 0)
    age_days = row.get("repo_age_days", 0)
    days_since_push = row.get("days_since_last_push", 0)

    if days_since_push > 365 and commits < 10 and stars < 5:
        return {"label": "low_value", "confidence": 0.7, "reasoning": "Abandoned: old, few commits, no stars"}

    if commits < 20 and contributors <= 1 and stars < 10 and not has_ci:
        return {"label": "intern", "confidence": 0.7, "reasoning": "Simple project: few commits, single contributor, no CI"}

    if stars > 500 and contributors > 20 and has_ci and releases > 5 and pr_count > 50:
        return {"label": "lead_architect", "confidence": 0.7, "reasoning": "High activity: many stars, contributors, releases, PRs"}

    if stars > 50 and contributors > 5 and (has_ci or releases > 2) and pr_count > 10:
        return {"label": "senior", "confidence": 0.6, "reasoning": "Moderate complexity: multiple contributors, some CI/releases"}

    return {"label": "junior", "confidence": 0.5, "reasoning": "Default: moderate activity, doesn't fit other categories clearly"}


def main():
    ensure_dirs()

    input_path = PROCESSED_DIR / "repo_summaries.csv"
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} repositories for labeling")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    use_llm = hf_token is not None

    if use_llm:
        logger.info(f"Using LLM model: {MODEL_ID}")
        client = InferenceClient(model=MODEL_ID, token=hf_token)
    else:
        logger.warning(
            "No HF_TOKEN found. Using rule-based fallback labeling. "
            "Set HF_TOKEN for LLM-based labeling: export HF_TOKEN=hf_..."
        )

    labels = []
    confidences = []
    reasonings = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
        if use_llm:
            result = classify_repo(client, row.get("summary", ""))
            time.sleep(1)
        else:
            result = fallback_rule_based(row)

        labels.append(result["label"])
        confidences.append(result["confidence"])
        reasonings.append(result["reasoning"])

    df["maturity_label"] = labels
    df["label_confidence"] = confidences
    df["label_reasoning"] = reasonings

    output_path = LABELED_DIR / "labeled_repositories.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved labeled data to {output_path}")
    logger.info(f"Label distribution:\n{df['maturity_label'].value_counts()}")
    logger.info(f"Average confidence: {df['label_confidence'].mean():.2f}")
    logger.info(f"Labeling method: {'LLM' if use_llm else 'Rule-based fallback'}")

    return df


if __name__ == "__main__":
    main()
