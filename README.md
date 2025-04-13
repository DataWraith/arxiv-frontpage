# arxiv-frontpage

Inspired by <https://github.com/koaning/arxiv-frontpage>, this project fetches
new computer science papers from [Arxiv](https://arxiv.org) and uses a 
classifier to infer tags from the paper metadata. 

Each tag is associated with an "interestingness" multiplier, and the final
frontpage ranks papers by multiplying the confidence that a given tag is present
with its interestingness modifier. The resulting score is then summed over all
tags.

You can create your own version of this project by defining tags in
`data/tags.json` and creating a `.jsonl`-file of the same name in `data/train`.
The generated website includes a copy button that exposes the JSON lines you
need to put into the training files.

The `Justfile` contains tasks for downloading new papers, training the
classifiers and generating the frontpage.