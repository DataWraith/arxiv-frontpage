# arxiv-frontpage

A tool that creates a personalized frontpage of arXiv computer science papers ranked by your interests using GitHub Actions.

## Demo

My frontpage for today can be viewed here:

<https://datawraith.github.io/arxiv-frontpage/>

## What does this do?

Inspired by <https://github.com/koaning/arxiv-frontpage>, this project fetches new computer science papers from [arXiv](https://arxiv.org) and uses a  classifier to infer tags from the paper metadata. Tags are displayed below each paper abstract once the classifier's confidence reaches a threshold.

Each tag is associated with an "interestingness" multiplier, and the final frontpage ranks papers by multiplying the confidence that a given tag is present with its interestingness modifier. The resulting score is then summed over all tags, giving you a personalized ranking of fresh papers.

The GitHub Actions automatically pull new data and regenerate the site once on every weekday -- you may need to allow GitHub Actions to commit new changes in the repository settings if you fork the project.

## How does it work?

1. **Tag Configuration**: Tags are defined in `data/tags.json` and mapped to their interestingness multiplier.
2. **Training Data**: Each tag must have an associated `.jsonl` file in the `data/train` directory.
3. **Paper Collection**: The system fetches recent papers from arXiv's CS categories via RSS feed.
4. **Classification**: A Probabilistic Label Tree classifier (via [napkinXC](https://napkinxc.readthedocs.io) determines the relevance of each tag for each paper.
5. **Ranking**: Papers are scored and the frontpage is generated.

The generated frontpage includes a copy button that displays the JSON data you need to put into the training files to improve future classifications.

You can also run the project locally using [uv](https://github.com/astral-sh/uv) -- see the `Justfile` for the available commands.
