#!/usr/bin/env python3

import json
import os
import re
import sys
from pathlib import Path


def load_tags():
    with open("data/tags.json", "r") as f:
        return json.load(f)


def get_train_files():
    train_dir = Path("data/train")
    return [f for f in train_dir.glob("*.jsonl")]


def validate_files_exist(tags):
    missing_files = []
    for tag in tags:
        expected_file = Path(f"data/train/{tag}.jsonl")
        if not expected_file.exists():
            missing_files.append(expected_file)
    return missing_files


def validate_no_extra_files(tags, train_files):
    extra_files = []
    tag_set = set(tags)
    for file in train_files:
        tag = file.stem
        if tag not in tag_set:
            extra_files.append(file)
    return extra_files


def process_jsonl_file(file_path):
    print(f"Processing {file_path}...")
    lines = []
    seen_urls = set()
    duplicates = []

    # Define allowed keys
    allowed_keys = {"title", "authors", "abstract", "created", "url"}

    with open(file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                filtered_data = {k: v for k, v in data.items() if k in allowed_keys}
                url = re.sub(r"v\d+$", "", filtered_data["url"])
                if url in seen_urls:
                    duplicates.append((url, filtered_data.get("title", "No title")))
                    continue
                seen_urls.add(url)
                filtered_data["url"] = url
                lines.append(filtered_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing line in {file_path}: {e}")
                return False

    # Sort by URL
    lines.sort(key=lambda x: x["url"])

    # Write back to file
    with open(file_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    # Print duplicates if any
    for url, title in duplicates:
        print(f"Removed duplicate in {file_path.name}: {title} ({url})")

    return True


def main():
    try:
        tags = load_tags()
    except Exception as e:
        print(f"Error loading tags.json: {e}")
        sys.exit(1)

    train_files = get_train_files()

    missing_files = validate_files_exist(tags)
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        sys.exit(1)

    extra_files = validate_no_extra_files(tags, train_files)
    if extra_files:
        print("Extra files found:")
        for file in extra_files:
            print(f"  - {file}")
        sys.exit(1)

    success = True
    for file in train_files:
        if not process_jsonl_file(file):
            success = False

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
