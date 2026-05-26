"""
Stage 6: Evaluation and Error Analysis
Computes metrics, generates confusion matrix, and analyzes classification errors.
"""

import json
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import (
    ensure_dirs, setup_logging, save_json,
    SPLITS_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR, TABLES_DIR,
    MATURITY_LABELS, LABEL_TO_ID, ID_TO_LABEL,
)

logger = setup_logging("evaluation")


def load_model():
    model_path = str(MODELS_DIR / "bert_maturity_classifier")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, texts: list[str], batch_size: int = 16) -> np.ndarray:
    all_preds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, truncation=True, padding=True, max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)

    return np.array(all_preds)


def evaluate_on_test():
    ensure_dirs()

    model, tokenizer = load_model()
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    logger.info(f"Loaded test set: {len(test_df)} samples")

    texts = test_df["summary"].tolist()
    true_labels = test_df["label_id"].values

    pred_ids = predict(model, tokenizer, texts)
    pred_labels = [ID_TO_LABEL[p] for p in pred_ids]
    true_label_names = [ID_TO_LABEL[t] for t in true_labels]

    acc = accuracy_score(true_labels, pred_ids)
    precision = precision_score(true_labels, pred_ids, average="macro", zero_division=0)
    recall = recall_score(true_labels, pred_ids, average="macro", zero_division=0)
    f1_macro = f1_score(true_labels, pred_ids, average="macro", zero_division=0)
    f1_weighted = f1_score(true_labels, pred_ids, average="weighted", zero_division=0)

    metrics = {
        "accuracy": round(acc, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "test_size": len(test_df),
    }

    save_json(metrics, METRICS_DIR / "test_metrics.json")
    logger.info(f"Test metrics: {metrics}")

    report = classification_report(
        true_labels, pred_ids,
        target_names=MATURITY_LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(TABLES_DIR / "classification_report.csv")
    logger.info(f"Classification report:\n{classification_report(true_labels, pred_ids, target_names=MATURITY_LABELS, zero_division=0)}")

    cm = confusion_matrix(true_labels, pred_ids)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=MATURITY_LABELS, yticklabels=MATURITY_LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Repository Maturity Classification")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    logger.info("Saved confusion matrix plot")

    test_df["predicted_label"] = pred_labels
    test_df["true_label_name"] = true_label_names
    test_df["correct"] = test_df["predicted_label"] == test_df["true_label_name"]

    errors = test_df[~test_df["correct"]].copy()
    error_analysis = []
    for _, row in errors.iterrows():
        error_analysis.append({
            "repo": row.get("full_name", row.get("name", "unknown")),
            "true_label": row["true_label_name"],
            "predicted_label": row["predicted_label"],
            "reasoning": row.get("label_reasoning", ""),
        })
    save_json(error_analysis, METRICS_DIR / "error_analysis.json")
    logger.info(f"Errors: {len(errors)} / {len(test_df)} ({len(errors)/len(test_df)*100:.1f}%)")

    per_class_acc = {}
    for label in MATURITY_LABELS:
        mask = test_df["true_label_name"] == label
        if mask.sum() > 0:
            per_class_acc[label] = round(test_df.loc[mask, "correct"].mean(), 4)
    save_json(per_class_acc, METRICS_DIR / "per_class_accuracy.json")

    fig, ax = plt.subplots(figsize=(10, 6))
    labels_sorted = sorted(per_class_acc.keys())
    values = [per_class_acc[l] for l in labels_sorted]
    bars = ax.bar(labels_sorted, values, color=sns.color_palette("viridis", len(labels_sorted)))
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_accuracy.png", dpi=150)
    plt.close()

    return metrics


def main():
    metrics = evaluate_on_test()
    logger.info("Evaluation complete!")
    return metrics


if __name__ == "__main__":
    main()
