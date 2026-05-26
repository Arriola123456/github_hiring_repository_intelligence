"""
Stage 4-5: Data Splitting and BERT Fine-Tuning
Splits labeled data and fine-tunes a DistilBERT model for maturity classification.
"""

import json
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

from src.utils import (
    ensure_dirs, setup_logging, LABELED_DIR, SPLITS_DIR, MODELS_DIR,
    MATURITY_LABELS, LABEL_TO_ID, ID_TO_LABEL
)

logger = setup_logging("train")

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = len(MATURITY_LABELS)
MAX_LENGTH = 512
EPOCHS = 4
BATCH_SIZE = 8
LEARNING_RATE = 2e-5


class RepoDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def split_data():
    ensure_dirs()
    input_path = LABELED_DIR / "labeled_repositories.csv"
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} labeled repositories")

    df["label_id"] = df["maturity_label"].map(LABEL_TO_ID)
    df = df.dropna(subset=["label_id", "summary"])
    df["label_id"] = df["label_id"].astype(int)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label_id"],
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"],
    )

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Train label distribution:\n{train_df['maturity_label'].value_counts()}")

    return train_df, val_df, test_df


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame):
    ensure_dirs()
    logger.info(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS,
    )

    model.config.id2label = ID_TO_LABEL
    model.config.label2id = LABEL_TO_ID

    train_texts = train_df["summary"].tolist()
    train_labels = train_df["label_id"].tolist()
    val_texts = val_df["summary"].tolist()
    val_labels = val_df["label_id"].tolist()

    train_dataset = RepoDataset(train_texts, train_labels, tokenizer)
    val_dataset = RepoDataset(val_texts, val_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=str(MODELS_DIR / "logs"),
        logging_steps=10,
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info(f"Training complete. Metrics: {train_result.metrics}")

    model_path = str(MODELS_DIR / "bert_maturity_classifier")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"Model saved to {model_path}")

    eval_result = trainer.evaluate()
    logger.info(f"Validation metrics: {eval_result}")

    return trainer, model, tokenizer


def main():
    train_df, val_df, test_df = split_data()
    trainer, model, tokenizer = train_model(train_df, val_df)

    results = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "model_name": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }

    from src.utils import METRICS_DIR, save_json
    save_json(results, METRICS_DIR / "training_config.json")

    return trainer, model, tokenizer


if __name__ == "__main__":
    main()
