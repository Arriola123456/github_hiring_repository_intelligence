import os
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELED_DIR = DATA_DIR / "labeled"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
METRICS_DIR = OUTPUT_DIR / "metrics"

ALL_DIRS = [RAW_DIR, PROCESSED_DIR, LABELED_DIR, SPLITS_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR, METRICS_DIR]

MATURITY_LABELS = ["intern", "junior", "senior", "lead_architect", "template", "low_value"]

LABEL_TO_ID = {label: i for i, label in enumerate(MATURITY_LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(MATURITY_LABELS)}


def ensure_dirs():
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    return logging.getLogger(name)


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
