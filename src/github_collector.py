"""
Stage 1: GitHub Data Collection
Collects repository data from GitHub API using diverse search queries
to capture different maturity levels.
"""

import os
import time
import pandas as pd
from github import Github, GithubException
from tqdm import tqdm

from src.utils import (
    ensure_dirs, setup_logging, save_json, RAW_DIR, PROCESSED_DIR
)

logger = setup_logging("github_collector")

SEARCH_QUERIES = {
    "intern_beginner": [
        "todo app python",
        "calculator python",
        "hello world python",
        "student project python",
        "learning python project",
        "beginner python project",
        "simple python game",
        "python homework",
    ],
    "junior": [
        "REST API flask",
        "web scraper python",
        "CRUD application python",
        "django blog",
        "fastapi project",
        "python bot telegram",
        "python cli tool",
        "python automation script",
    ],
    "senior": [
        "machine learning pipeline python",
        "data platform python",
        "microservices python",
        "python ETL pipeline",
        "deep learning framework",
        "NLP pipeline python",
        "recommendation system python",
        "python distributed computing",
    ],
    "lead_architect": [
        "python framework",
        "python SDK",
        "python infrastructure",
        "python orchestration platform",
        "python developer tools",
        "python compiler",
        "python database engine",
        "python cloud platform",
    ],
    "template": [
        "python template",
        "python boilerplate",
        "python starter",
        "python cookiecutter",
        "python scaffold project",
        "python project template",
    ],
}

REPOS_PER_QUERY = 8


def get_github_client() -> Github:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN not set. Export it before running: "
            "export GITHUB_TOKEN=ghp_..."
        )
    return Github(token, per_page=30)


def extract_repo_signals(repo) -> dict | None:
    try:
        contributors = repo.get_contributors()
        try:
            contributor_count = contributors.totalCount
        except GithubException:
            contributor_count = 0

        commits = repo.get_commits()
        commit_count = commits.totalCount

        age_days = (pd.Timestamp.utcnow() - pd.Timestamp(repo.created_at, tz="UTC")).days
        days_since_push = (pd.Timestamp.utcnow() - pd.Timestamp(repo.pushed_at, tz="UTC")).days
        commits_per_month = (commit_count / max(age_days / 30, 1)) if age_days > 0 else 0

        pulls = repo.get_pulls(state="all")
        pr_count = pulls.totalCount

        releases = repo.get_releases()
        release_count = releases.totalCount

        issues_open = repo.open_issues_count
        issues_closed = 0
        try:
            closed_issues = repo.get_issues(state="closed")
            issues_closed = closed_issues.totalCount
        except GithubException:
            pass

        readme_length = 0
        readme_has_badges = False
        readme_has_images = False
        readme_text = ""
        try:
            readme = repo.get_readme()
            readme_text = readme.decoded_content.decode("utf-8", errors="replace")
            readme_length = len(readme_text)
            readme_has_badges = "[![" in readme_text or "badge" in readme_text.lower()
            readme_has_images = "![" in readme_text or "<img" in readme_text.lower()
        except GithubException:
            pass

        has_ci = False
        try:
            contents = repo.get_contents("")
            filenames = [c.name for c in contents]
            if ".github" in filenames:
                try:
                    workflows = repo.get_contents(".github/workflows")
                    has_ci = len(workflows) > 0
                except GithubException:
                    pass
            if not has_ci:
                ci_files = [".travis.yml", "Jenkinsfile", ".circleci", "azure-pipelines.yml", "Makefile"]
                has_ci = any(f in filenames for f in ci_files)
        except GithubException:
            pass

        dep_count = 0
        try:
            req = repo.get_contents("requirements.txt")
            dep_count = len(req.decoded_content.decode("utf-8", errors="replace").strip().split("\n"))
        except GithubException:
            try:
                setup = repo.get_contents("setup.py")
                dep_count = setup.decoded_content.decode("utf-8", errors="replace").count("install_requires")
            except GithubException:
                pass

        return {
            "full_name": repo.full_name,
            "name": repo.name,
            "owner": repo.owner.login,
            "description": repo.description or "",
            "language": repo.language or "Unknown",
            "license": repo.license.spdx_id if repo.license else "None",
            "topics": repo.get_topics(),
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "watchers": repo.watchers_count,
            "contributors_count": contributor_count,
            "commit_count": commit_count,
            "commits_per_month": round(commits_per_month, 2),
            "open_issues": issues_open,
            "closed_issues": issues_closed,
            "pr_count": pr_count,
            "release_count": release_count,
            "readme_length": readme_length,
            "readme_has_badges": readme_has_badges,
            "readme_has_images": readme_has_images,
            "has_ci_cd": has_ci,
            "dependency_count": dep_count,
            "repo_age_days": age_days,
            "days_since_last_push": days_since_push,
            "description_length": len(repo.description or ""),
            "created_at": str(repo.created_at),
            "updated_at": str(repo.updated_at),
            "pushed_at": str(repo.pushed_at),
            "default_branch": repo.default_branch,
            "size_kb": repo.size,
            "readme_text": readme_text[:2000],
        }
    except GithubException as e:
        logger.warning(f"Error extracting signals from {repo.full_name}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error for {repo.full_name}: {e}")
        return None


def collect_repositories() -> list[dict]:
    g = get_github_client()
    all_repos = []
    seen = set()

    for category, queries in SEARCH_QUERIES.items():
        logger.info(f"Collecting repos for category hint: {category}")
        for query in queries:
            logger.info(f"  Searching: '{query}'")
            try:
                results = g.search_repositories(
                    query=query,
                    sort="stars",
                    order="desc",
                    language="python",
                )
                count = 0
                for repo in results:
                    if repo.full_name in seen:
                        continue
                    if count >= REPOS_PER_QUERY:
                        break

                    logger.info(f"    Extracting: {repo.full_name}")
                    signals = extract_repo_signals(repo)
                    if signals:
                        signals["search_category"] = category
                        signals["search_query"] = query
                        all_repos.append(signals)
                        seen.add(repo.full_name)
                        count += 1

                    time.sleep(0.5)

            except GithubException as e:
                if e.status == 403:
                    logger.warning("Rate limited. Waiting 60 seconds...")
                    time.sleep(60)
                else:
                    logger.warning(f"Search error for '{query}': {e}")
            except Exception as e:
                logger.warning(f"Unexpected search error for '{query}': {e}")

            time.sleep(1)

    logger.info(f"Total repositories collected: {len(all_repos)}")
    return all_repos


def save_results(repos: list[dict]):
    ensure_dirs()

    save_json(repos, RAW_DIR / "repositories.json")
    logger.info(f"Saved raw data to {RAW_DIR / 'repositories.json'}")

    df = pd.DataFrame(repos)
    df.to_csv(PROCESSED_DIR / "repositories.csv", index=False)
    logger.info(f"Saved processed CSV to {PROCESSED_DIR / 'repositories.csv'}")

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Search categories distribution:\n{df['search_category'].value_counts()}")

    return df


def main():
    logger.info("Starting GitHub repository collection...")
    repos = collect_repositories()
    if repos:
        df = save_results(repos)
        logger.info(f"Collection complete! {len(repos)} repositories saved.")
        return df
    else:
        logger.error("No repositories collected. Check your GITHUB_TOKEN.")
        return None


if __name__ == "__main__":
    main()
