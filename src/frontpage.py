import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import srsly
from jinja2 import Environment, FileSystemLoader
from napkinxc.models import PLT
from tqdm import tqdm
import numpy as np

from tagger.osb import bigram_indices, wrap_indices
from tagger.plt_classifier import VECTOR_SIZE, make_classifier


def load_tags() -> Dict[str, float]:
    """Load tags and their weights from JSON file."""
    with open("data/tags.json") as f:
        return json.load(f)


def format_paper(paper: Dict) -> str:
    authors = ", ".join(paper["authors"])
    return f"Title: {paper['title']}\nAuthors: {authors}\nAbstract: {paper['abstract']}"


def get_tag_scores(
    paper: Dict, idx_to_tag: Dict[str, int], classifier: PLT
) -> Dict[str, float]:
    """Get tag confidence scores for a paper."""

    paper_text = format_paper(paper)
    scores = {}
    indices = wrap_indices(bigram_indices(paper_text), VECTOR_SIZE)
    probas = classifier.predict_proba([indices], threshold=0.01)

    for proba in probas[0]:
        tag_idx, prob = proba
        tag = idx_to_tag.get(tag_idx, None)

        if tag is None:
            continue

        score = 100.0 * prob
        scores[tag] = score

    return scores


def process_paper_scores(
    paper: Dict, tags: Dict[str, float], idx_to_tag: Dict[str, int], classifier: PLT
) -> Dict:
    """Process a single paper to get its category scores."""

    paper["tag_scores"] = get_tag_scores(paper, idx_to_tag, classifier)
    paper["date"] = paper["created"]

    interestingness = 0.0
    for tag, score in paper["tag_scores"].items():
        interestingness += score * tags[tag]
    interestingness /= max(1, len(paper["tag_scores"]))

    paper["interestingness_score"] = round(interestingness, 4)

    return paper


def get_recent_papers(papers: List[Dict], days: int = 3) -> List[Dict]:
    """Get papers from the last N days."""

    recent_papers = []
    cutoff_date = datetime.now() - timedelta(days=days)

    for paper in papers:
        paper_date = datetime.strptime(paper["created"], "%Y-%m-%d")
        if paper_date >= cutoff_date:
            recent_papers.append(paper)

    return recent_papers


def get_tag_color(tag: str) -> str:
    """Generate a random but readable color for a tag based on its hash."""
    hash_val = int(hashlib.md5(tag.encode()).hexdigest()[:6], 16)
    r = (hash_val >> 16) & 0xFF
    g = (hash_val >> 8) & 0xFF
    b = hash_val & 0xFF
    if r + g + b < 200:
        r = min(255, r + 100)
        g = min(255, g + 100)
        b = min(255, b + 100)
    return f"#{r:02x}{g:02x}{b:02x}"


def select_papers(papers: List[Dict], target_count: int = 300) -> List[Dict]:
    """Select papers according to a mixed strategy:
    - Top 25 papers by interestingness score
    - Bottom 25 papers by interestingness score
    - 25 papers closest to zero interestingness score
    - Remaining papers selected with probability proportional to absolute interestingness score
    """
    if len(papers) <= target_count:
        return papers

    # Sort papers by interestingness score
    papers.sort(key=lambda p: -p["interestingness_score"])
    
    # Select fixed points
    top_papers = papers[:25]
    bottom_papers = papers[-25:]
    
    # Find papers closest to zero
    zero_papers = sorted(papers[25:-25], key=lambda p: abs(p["interestingness_score"]))[:25]
    
    # Get remaining papers
    remaining_papers = [p for p in papers[25:-25] if p not in zero_papers]
    
    # Calculate selection probabilities based on absolute interestingness score
    total_score = sum(abs(p["interestingness_score"]) for p in remaining_papers)
    if total_score > 0:
        probs = [abs(p["interestingness_score"]) / total_score for p in remaining_papers]
    else:
        probs = [1.0 / len(remaining_papers) for _ in remaining_papers]
    
    # Select remaining papers
    np.random.seed(42)  # For reproducibility
    selected_indices = np.random.choice(
        len(remaining_papers),
        size=target_count - 75,  # 75 = 25 + 25 + 25
        replace=False,
        p=probs
    )
    selected_remaining = [remaining_papers[i] for i in selected_indices]
    
    # Combine all selected papers
    final_papers = top_papers + selected_remaining + zero_papers + bottom_papers
    
    # Sort by interestingness score for final display
    final_papers.sort(key=lambda p: -p["interestingness_score"])
    
    return final_papers


def generate_html(
    papers: List[Dict],
    tags: Dict[str, float],
    classifier: PLT,
) -> str:
    """Generate HTML for the frontpage."""

    papers = get_recent_papers(papers)

    tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}

    processed_papers = [
        process_paper_scores(paper, tags, idx_to_tag, classifier)
        for paper in tqdm(papers, desc="Calculating scores")
    ]

    # Select papers according to our strategy
    processed_papers = select_papers(processed_papers)

    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("frontpage.html")

    return template.render(
        papers=processed_papers,
        get_tag_color=get_tag_color,
        last_updated=datetime.now().strftime("%Y-%m-%d"),
    )


def main():
    # Load classifier and tags
    classifier = make_classifier()
    tags = load_tags()

    # Read all JSONL files
    papers = []
    jsonl_files = list(Path("data/downloads").glob("*.jsonl"))
    for jsonl_file in tqdm(jsonl_files, desc="Reading JSONL files"):
        papers.extend(srsly.read_jsonl(jsonl_file))

    html_content = generate_html(
        papers=papers,
        tags=tags,
        classifier=classifier,
    )

    output_file = "index.html"
    with open(output_file, "w") as f:
        f.write(html_content)


if __name__ == "__main__":
    main()
