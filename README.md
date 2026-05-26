# GitHub Hiring Repository Intelligence

**Track A: Hiring-Oriented Repository Maturity Classification**

Classify GitHub repositories by engineering maturity level using LLM weak labeling and BERT fine-tuning. The system estimates repository complexity — not developer skill — using GitHub signals like commit frequency, contributors, CI/CD presence, and documentation quality.

## Maturity Categories

| Level | Description |
|-------|-------------|
| **Intern** | Simple projects: hello world, calculators, tutorials, homework |
| **Junior** | Functional apps: CRUD, web scrapers, bots, basic APIs |
| **Senior** | Well-engineered: ML pipelines, complex APIs, data platforms |
| **Lead/Architect** | Framework-level: SDKs, infrastructure, orchestration tools |
| **Template** | Boilerplate, starters, scaffold projects |
| **Low Value** | Abandoned, empty, or minimal activity repositories |

## Pipeline

1. **Data Collection**: GitHub REST API → 200-400 repos with 14+ signals
2. **Representation**: Repository signals → structured textual summaries
3. **Weak Labeling**: Open-source LLM (Mistral-7B via HuggingFace) generates training labels
4. **Data Split**: Stratified 70% train / 15% validation / 15% test
5. **BERT Fine-Tuning**: DistilBERT for lightweight maturity classification
6. **Evaluation**: Accuracy, precision, recall, F1 + confusion matrix + error analysis

## Signals Extracted

Stars, forks, watchers, contributors, commit count, commit frequency, open/closed issues, pull requests, releases, README length/badges/images, CI/CD presence, dependency count, topics, repo age, days since last push, language, license, repository size.

## Tech Stack

- **Data**: PyGithub, pandas, numpy
- **LLM Labeling**: HuggingFace Inference API (Mistral-7B-Instruct) — free, open-source
- **ML**: HuggingFace Transformers (DistilBERT), scikit-learn, PyTorch
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit (4 tabs)

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GITHUB_TOKEN=ghp_your_token_here
export HF_TOKEN=hf_your_token_here  # Optional: uses rule-based fallback if not set

# Run pipeline stages
python -m src.github_collector     # Stage 1: Collect repos
python -m src.preprocessing        # Stage 2a: Clean data
python -m src.summarization        # Stage 2b: Generate summaries
python -m src.llm_labeling         # Stage 3: Weak labeling
python -m src.train                # Stage 4-5: Split + train BERT
python -m src.evaluation           # Stage 6: Evaluate model
python -m src.visualization        # Generate plots

# Launch Streamlit dashboard
streamlit run app.py
```

## Streamlit Dashboard

| Tab | Content |
|-----|---------|
| **Problem & Methodology** | Objective, signals, prompts, dataset construction, limitations |
| **Exploratory Analysis** | Distributions, signal comparisons, interactive visualizations |
| **Model Results** | Metrics, confusion matrix, per-class performance, error analysis |
| **Interactive Exploration** | Search/filter repos, view predictions and metadata |

## Project Structure

```
├── app.py                    # Streamlit dashboard (4 tabs)
├── CLAUDE.md                 # Development plan and progress tracker
├── README.md
├── requirements.txt
├── src/
│   ├── github_collector.py   # Stage 1: GitHub API data collection
│   ├── preprocessing.py      # Stage 2a: Data cleaning
│   ├── summarization.py      # Stage 2b: Text representation
│   ├── llm_labeling.py       # Stage 3: LLM/rule-based weak labeling
│   ├── train.py              # Stage 4-5: Split + BERT fine-tuning
│   ├── evaluation.py         # Stage 6: Metrics and error analysis
│   ├── visualization.py      # Exploratory plots
│   └── utils.py              # Shared constants and helpers
├── data/                     # Raw, processed, labeled, splits
├── models/trained_models/    # Saved BERT model
├── output/                   # Figures, tables, metrics
└── video/link.txt            # Explanatory video link
```

## Limitations

- Weak labels from LLM may contain noise and systematic biases
- GitHub signals are proxies — they don't capture actual code quality
- Repository maturity does not equal developer skill level
- Search queries introduce selection bias toward popular Python projects

## Business Applications

- **Recruiters**: Screen candidate portfolios by repository complexity
- **Startups**: Identify experienced engineers from open-source contributions
- **Interview pipelines**: Pre-filter candidates based on project sophistication
- **Engineering managers**: Assess team technical depth and growth potential
