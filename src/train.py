import json
from pathlib import Path
from typing import Dict, List, Tuple

from tagger.osb import bigram_indices, wrap_indices
from tagger.plt_classifier import VECTOR_SIZE, make_classifier


def format_paper(paper: Dict) -> str:
    """Format paper information into a string."""
    authors = ", ".join(paper["authors"])
    return f"Title: {paper['title']}\nAuthors: {authors}\nAbstract: {paper['abstract']}"


def load_training_data(tags: Dict[str, float]) -> List[Tuple[List[int], str, bool]]:
    """Load training data from JSONL files and create bigram indices."""

    tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}

    paper_features = {}
    paper_targets = {}

    for tag in tags:
        jsonl_file = Path("data/train") / f"{tag}.jsonl"
        if not jsonl_file.exists():
            continue

        with open(jsonl_file, "r") as f:
            for line in f:
                paper = json.loads(line)
                paper_url = paper["url"]

                if paper_url in paper_features:
                    paper_targets[paper_url].append(tag_to_idx[tag])
                else:
                    paper_text = format_paper(paper)
                    indices = bigram_indices(paper_text)
                    indices = wrap_indices(indices, VECTOR_SIZE)
                    paper_features[paper_url] = indices
                    paper_targets[paper_url] = [tag_to_idx[tag]]

    X_train, y_train = [], []

    for paper_url, features in paper_features.items():
        X_train.append(features)
        y_train.append(paper_targets[paper_url])

    return X_train, y_train


def main():
    with open("data/tags.json", "r") as f:
        tags = json.load(f)

    X_train, y_train = load_training_data(tags)

    # HACK: Drive classification probabilities up to useful levels.
    # TODO: Remove once I have enough training data.
    X_train = X_train * 3
    y_train = y_train * 3

    plt = make_classifier()
    plt.fit(X_train, y_train)


if __name__ == "__main__":
    main()
