default: download frontpage

download:
    uv run src/download.py

validate:
    uv run src/validate.py

train: validate
    uv run src/train.py

frontpage: train
    uv run src/frontpage.py

