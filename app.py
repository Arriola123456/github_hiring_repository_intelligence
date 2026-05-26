"""
Streamlit Dashboard: GitHub Repository Intelligence
4-tab application for exploring repository maturity classification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="GitHub Repository Intelligence",
    page_icon="🔍",
    layout="wide",
)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
MODELS_DIR = Path("models/trained_models")

MATURITY_LABELS = ["intern", "junior", "senior", "lead_architect", "template", "low_value"]

LABEL_COLORS = {
    "intern": "#FF6B6B",
    "junior": "#FFA726",
    "senior": "#66BB6A",
    "lead_architect": "#42A5F5",
    "template": "#AB47BC",
    "low_value": "#BDBDBD",
}

LABEL_DESCRIPTIONS = {
    "intern": "Very simple projects: hello world, calculators, tutorials, homework. Minimal commits, no CI/CD.",
    "junior": "Functional but straightforward: simple web apps, CRUD, bots, scripts. Some structure.",
    "senior": "Well-engineered: ML pipelines, data platforms, complex APIs. Multiple contributors, CI/CD.",
    "lead_architect": "Framework-level or infrastructure. High complexity, many contributors, SDK patterns.",
    "template": "Boilerplate, starter, scaffold projects. Designed to be copied, not used directly.",
    "low_value": "Abandoned, empty, or very low activity repositories.",
}


@st.cache_data
def load_data():
    data = {}
    labeled_path = DATA_DIR / "labeled" / "labeled_repositories.csv"
    if labeled_path.exists():
        data["labeled"] = pd.read_csv(labeled_path)

    for split in ["train", "val", "test"]:
        path = DATA_DIR / "splits" / f"{split}.csv"
        if path.exists():
            data[split] = pd.read_csv(path)

    metrics_path = OUTPUT_DIR / "metrics" / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data["metrics"] = json.load(f)

    report_path = OUTPUT_DIR / "tables" / "classification_report.csv"
    if report_path.exists():
        data["report"] = pd.read_csv(report_path, index_col=0)

    errors_path = OUTPUT_DIR / "metrics" / "error_analysis.json"
    if errors_path.exists():
        with open(errors_path) as f:
            data["errors"] = json.load(f)

    per_class_path = OUTPUT_DIR / "metrics" / "per_class_accuracy.json"
    if per_class_path.exists():
        with open(per_class_path) as f:
            data["per_class"] = json.load(f)

    return data


def tab_methodology():
    st.header("Problem & Methodology")

    st.subheader("Objective")
    st.write("""
    Classify GitHub repositories by **engineering maturity level** to estimate
    the complexity and sophistication of projects. This is useful for:
    - **Recruiters**: Quickly assess candidate portfolio quality
    - **Startups**: Identify experienced engineers from their open-source work
    - **Engineering managers**: Evaluate team technical depth
    """)

    st.subheader("Track: Hiring-Oriented Repository Intelligence")
    st.write("We classify repositories into 6 maturity levels:")

    cols = st.columns(3)
    for i, (label, desc) in enumerate(LABEL_DESCRIPTIONS.items()):
        with cols[i % 3]:
            st.markdown(f"**{label.replace('_', ' ').title()}**")
            st.caption(desc)

    st.subheader("Data Collection")
    st.write("""
    - **Source**: GitHub REST API via PyGithub
    - **Strategy**: Diverse search queries targeting different maturity levels
    - **Signals extracted** (14+): stars, forks, contributors, commit frequency,
      PR count, releases, README characteristics, CI/CD presence, dependencies,
      topics, repo age, last activity, language, license
    """)

    st.subheader("Pipeline")
    st.write("""
    1. **GitHub Data Collection** - Extract repository signals via API
    2. **Repository Representation** - Convert signals to textual summaries
    3. **Weak Labeling with LLM** - Use open-source LLM (Mistral-7B via HuggingFace) to generate training labels
    4. **Train/Val/Test Split** - Stratified 70/15/15 split
    5. **Fine-Tune BERT** - DistilBERT for lightweight classification
    6. **Evaluation** - Accuracy, precision, recall, F1 + error analysis
    """)

    st.subheader("Limitations")
    st.write("""
    - Weak labels from LLM may contain noise and biases
    - GitHub signals don't capture code quality directly
    - Repository maturity ≠ developer skill level
    - Search queries introduce selection bias
    """)


def tab_exploratory(data):
    st.header("Exploratory Data Analysis")

    if "labeled" not in data:
        st.warning("No labeled data found. Run the pipeline first.")
        return

    df = data["labeled"]
    st.metric("Total Repositories", len(df))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Label Distribution")
        counts = df["maturity_label"].value_counts().reset_index()
        counts.columns = ["label", "count"]
        fig = px.bar(counts, x="label", y="count",
                     color="label", color_discrete_map=LABEL_COLORS,
                     title="Repository Count by Maturity Level")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Language Distribution")
        lang_counts = df["language"].value_counts().head(10).reset_index()
        lang_counts.columns = ["language", "count"]
        fig = px.pie(lang_counts, names="language", values="count",
                     title="Top 10 Programming Languages")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Signal Comparisons by Maturity Level")
    signal = st.selectbox("Select signal to compare",
                          ["stars", "forks", "contributors_count", "commit_count",
                           "pr_count", "release_count", "readme_length",
                           "repo_age_days", "dependency_count"])

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x="maturity_label", y=signal,
                     color="maturity_label", color_discrete_map=LABEL_COLORS,
                     title=f"{signal} by Maturity Level")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.violin(df, x="maturity_label", y=signal,
                        color="maturity_label", color_discrete_map=LABEL_COLORS,
                        title=f"{signal} Distribution (Violin)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Stars vs Contributors")
    fig = px.scatter(df, x="stars", y="contributors_count",
                     color="maturity_label", color_discrete_map=LABEL_COLORS,
                     hover_data=["full_name", "description"],
                     title="Stars vs Contributors by Maturity Level",
                     log_x=True, log_y=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary Statistics")
    numeric_cols = ["stars", "forks", "contributors_count", "commit_count",
                    "pr_count", "release_count", "readme_length"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    summary = df.groupby("maturity_label")[numeric_cols].describe().round(1)
    st.dataframe(summary, use_container_width=True)


def tab_results(data):
    st.header("Model Results")

    if "metrics" not in data:
        st.warning("No evaluation metrics found. Run the evaluation pipeline first.")
        return

    metrics = data["metrics"]

    st.subheader("Overall Test Metrics")
    cols = st.columns(5)
    metric_items = [
        ("Accuracy", metrics.get("accuracy", 0)),
        ("Precision (Macro)", metrics.get("precision_macro", 0)),
        ("Recall (Macro)", metrics.get("recall_macro", 0)),
        ("F1 (Macro)", metrics.get("f1_macro", 0)),
        ("F1 (Weighted)", metrics.get("f1_weighted", 0)),
    ]
    for col, (name, value) in zip(cols, metric_items):
        col.metric(name, f"{value:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Per-Class Accuracy")
        if "per_class" in data:
            pc = data["per_class"]
            fig = px.bar(x=list(pc.keys()), y=list(pc.values()),
                         color=list(pc.keys()), color_discrete_map=LABEL_COLORS,
                         title="Accuracy per Maturity Level",
                         labels={"x": "Label", "y": "Accuracy"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Classification Report")
        if "report" in data:
            st.dataframe(data["report"].round(3), use_container_width=True)

    st.subheader("Confusion Matrix")
    cm_path = Path("output/figures/confusion_matrix.png")
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)
    else:
        st.info("Confusion matrix image not generated yet.")

    st.subheader("Error Analysis")
    if "errors" in data and data["errors"]:
        errors_df = pd.DataFrame(data["errors"])
        st.write(f"Total misclassifications: **{len(errors_df)}** out of {metrics.get('test_size', '?')} test samples")

        confusion_pairs = errors_df.groupby(["true_label", "predicted_label"]).size().reset_index(name="count")
        confusion_pairs = confusion_pairs.sort_values("count", ascending=False)
        st.write("Most common confusion pairs:")
        st.dataframe(confusion_pairs.head(10), use_container_width=True)


def tab_interactive(data):
    st.header("Interactive Exploration")

    if "labeled" not in data:
        st.warning("No labeled data found. Run the pipeline first.")
        return

    df = data["labeled"]

    st.subheader("Search & Filter Repositories")

    col1, col2, col3 = st.columns(3)
    with col1:
        search = st.text_input("Search by name or description")
    with col2:
        selected_labels = st.multiselect("Filter by maturity level",
                                         MATURITY_LABELS, default=MATURITY_LABELS)
    with col3:
        min_stars = st.number_input("Minimum stars", min_value=0, value=0)

    filtered = df[df["maturity_label"].isin(selected_labels)]
    if search:
        mask = (
            filtered["name"].str.contains(search, case=False, na=False)
            | filtered["description"].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]
    if min_stars > 0:
        filtered = filtered[filtered["stars"] >= min_stars]

    st.write(f"Showing **{len(filtered)}** of {len(df)} repositories")

    display_cols = ["full_name", "maturity_label", "stars", "forks",
                    "contributors_count", "commit_count", "language",
                    "label_confidence", "description"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].sort_values("stars", ascending=False),
        use_container_width=True,
        height=400,
    )

    st.subheader("Repository Detail View")
    if not filtered.empty:
        repo_names = filtered["full_name"].tolist()
        selected_repo = st.selectbox("Select a repository", repo_names)

        if selected_repo:
            repo = filtered[filtered["full_name"] == selected_repo].iloc[0]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Maturity", repo["maturity_label"])
            col2.metric("Stars", int(repo.get("stars", 0)))
            col3.metric("Contributors", int(repo.get("contributors_count", 0)))
            col4.metric("Commits", int(repo.get("commit_count", 0)))

            with st.expander("Full Details"):
                for col in repo.index:
                    if col not in ["readme_text", "summary"] and pd.notna(repo[col]):
                        st.write(f"**{col}**: {repo[col]}")

            if "label_reasoning" in repo.index:
                st.write(f"**Classification reasoning**: {repo.get('label_reasoning', 'N/A')}")


def main():
    st.title("GitHub Repository Intelligence")
    st.caption("Hiring-Oriented Repository Maturity Classification")

    data = load_data()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Problem & Methodology",
        "Exploratory Analysis",
        "Model Results",
        "Interactive Exploration",
    ])

    with tab1:
        tab_methodology()
    with tab2:
        tab_exploratory(data)
    with tab3:
        tab_results(data)
    with tab4:
        tab_interactive(data)


if __name__ == "__main__":
    main()
