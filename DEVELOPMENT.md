# GitHub Hiring Repository Intelligence

## Project Overview
Track A: Classify GitHub repositories by engineering maturity level (intern, junior, senior, lead/architect, template/boilerplate, low-value) using LLM weak labeling + BERT fine-tuning.

## Tech Stack
- Python 3.10+
- HuggingFace Transformers (DistilBERT or DeBERTa-v3-small)
- pandas, scikit-learn, matplotlib, seaborn
- Streamlit (4-tab dashboard)
- GitHub REST API (PyGithub)
- OpenAI/Claude API for weak labeling

## Project Structure
```
├── app.py                    # Streamlit app (4 tabs)
├── CLAUDE.md
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── github_collector.py   # Stage 1: GitHub API data collection
│   ├── preprocessing.py      # Stage 2: Clean and structure data
│   ├── summarization.py      # Stage 2: Convert repos to text representations
│   ├── llm_labeling.py       # Stage 3: Weak labeling with LLMs
│   ├── train.py              # Stage 4-5: Split data + fine-tune BERT
│   ├── evaluation.py         # Stage 6: Metrics and error analysis
│   ├── visualization.py      # Charts and plots
│   └── utils.py              # Shared utilities
├── data/
│   ├── raw/                  # Raw API responses
│   ├── processed/            # Cleaned DataFrames
│   ├── labeled/              # LLM-labeled data
│   └── splits/               # train/val/test CSVs
├── models/trained_models/    # Saved BERT model
├── output/
│   ├── figures/              # Plots
│   ├── tables/               # Result tables
│   └── metrics/              # JSON metrics
└── video/link.txt            # Video link
```

## Stage Plan

### Stage 1 - GitHub Data Collection
**File:** `src/github_collector.py`
- Use PyGithub to collect 200-400 repositories
- Search strategy: mix of query terms targeting different maturity levels
  - Beginner: "todo app", "calculator", "hello world", "tutorial", "learning python"
  - Junior: "REST API", "web scraper", "CRUD app", "flask app"
  - Senior: "microservices", "distributed system", "ML pipeline", "data platform"
  - Lead/Architect: "framework", "SDK", "infrastructure", "orchestration"
  - Template: "template", "boilerplate", "starter", "scaffold"
- Extract minimum 8 signals per repo:
  1. contributors_count
  2. commit_frequency (commits per month)
  3. stars, forks, watchers
  4. open_issues_count, closed_issues_count
  5. pull_request_count (open + closed)
  6. release_count
  7. readme_length, readme_has_badges, readme_has_images
  8. has_ci_cd (check for .github/workflows, .travis.yml, etc.)
  9. dependency_count
  10. topics
  11. repo_age_days
  12. days_since_last_push
  13. language, license
  14. description_length
- Save raw data to `data/raw/repositories.json`
- Save processed DataFrame to `data/processed/repositories.csv`

### Stage 2 - Repository Representation
**Files:** `src/preprocessing.py`, `src/summarization.py`
- Clean and normalize all signals
- Create textual summaries for each repo (hybrid approach):
  - Structured natural language paragraph combining all signals
  - Example: "This repository has 15 contributors, 200 commits over 2 years, 450 stars..."
- Save summaries to `data/processed/repo_summaries.csv`

### Stage 3 - Weak Labeling with LLMs
**File:** `src/llm_labeling.py`
- Use HuggingFace Inference API with Mistral-7B-Instruct (free, open-source)
- Fallback: rule-based classification if HF_TOKEN not available
- Prompt design: provide category definitions + repo summary → get label
- Categories: intern, junior, senior, lead_architect, template, low_value
- Include confidence scores
- Save to `data/labeled/labeled_repositories.csv`

### Stage 4 - Train/Val/Test Split
**File:** `src/train.py` (first part)
- Stratified split: 70% train, 15% val, 15% test
- Save splits to `data/splits/`

### Stage 5 - Fine-Tune BERT
**File:** `src/train.py` (second part)
- Use `distilbert-base-uncased` or `microsoft/deberta-v3-small`
- Tokenize repo summaries
- Train for 3-5 epochs with early stopping on validation loss
- Save model to `models/trained_models/`

### Stage 6 - Evaluation & Streamlit App
**Files:** `src/evaluation.py`, `src/visualization.py`, `app.py`
- Compute accuracy, precision, recall, F1 (macro + per-class)
- Confusion matrix
- Error analysis: which categories get confused
- Build Streamlit app with 4 tabs:
  - Tab 1: Problem & Methodology
  - Tab 2: Exploratory Analysis
  - Tab 3: Model Results
  - Tab 4: Interactive Exploration
- Write comprehensive README.md
- Save metrics to `output/metrics/`

## Commands
- Install dependencies: `pip install -r requirements.txt`
- Run data collection: `python -m src.github_collector`
- Run preprocessing: `python -m src.preprocessing`
- Run summarization: `python -m src.summarization`
- Run LLM labeling: `python -m src.llm_labeling`
- Run training: `python -m src.train`
- Run evaluation: `python -m src.evaluation`
- Run Streamlit app: `streamlit run app.py`

## Environment Variables Needed
- `GITHUB_TOKEN`: GitHub personal access token for API access
- `HF_TOKEN`: HuggingFace token for Inference API (free at huggingface.co/settings/tokens)
  - If not set, llm_labeling.py uses a rule-based fallback classifier

## Git Workflow
- Use feature branches: `feature/github-scraping`, `feature/llm-labeling`, `feature/bert-training`, `feature/streamlit-dashboard`
- Descriptive commits in English
- Merge through the working branch

## Current Progress Tracker
- [ ] Stage 1: GitHub Data Collection
- [ ] Stage 2: Repository Representation
- [ ] Stage 3: Weak Labeling with LLMs
- [ ] Stage 4: Train/Val/Test Split
- [ ] Stage 5: Fine-Tune BERT
- [ ] Stage 6: Evaluation & Error Analysis
- [ ] Streamlit App (4 tabs)
- [ ] README.md
- [ ] Video link
