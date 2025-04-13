import datetime as dt
import re
from pathlib import Path

import feedparser
import srsly
from pydantic import BaseModel
from retry import retry
from rich.console import Console
from tqdm import tqdm


class ArxivPaper(BaseModel):
    created: str
    title: str
    abstract: str
    authors: list[str]
    url: str


console = Console()


def parse_date(date_str: str) -> str:
    """Convert RSS pubDate to YYYY-MM-DD format"""
    dt_obj = dt.datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
    return dt_obj.strftime("%Y-%m-%d")


def parse_authors(authors_str: str) -> list[str]:
    """Parse authors string into list of author names"""
    return [name.strip() for name in authors_str.split(",")]


def extract_abstract(description: str) -> str:
    """Extract abstract from description field"""
    match = re.search(r"Abstract: (.*)", description)
    return match.group(1) if match else ""


@retry(tries=5, delay=5, backoff=2)
def retrieve_articles() -> list[ArxivPaper]:
    console.log("Starting arxiv RSS feed parsing...")

    feed = feedparser.parse("https://rss.arxiv.org/rss/cs")

    if not feed.entries:
        console.log("No entries found in feed")
        return None

    articles = []
    for entry in tqdm(feed.entries, desc="Parsing feed entries"):
        article = ArxivPaper(
            created=parse_date(entry.published),
            title=entry.title,
            abstract=extract_abstract(entry.summary),
            url=entry.link,
            authors=parse_authors(entry.get("author", "")),
        )

        articles.append(dict(article))

    console.log(f"Found {len(articles)} new articles.")

    if not articles:
        return None

    articles_dict = {ex["title"]: ex for ex in articles}
    most_recent = list(sorted(Path("data/downloads/").glob("*.jsonl")))

    old_articles_dict = (
        {}
        if len(most_recent) == 0
        else {ex["title"]: ex for ex in srsly.read_jsonl(most_recent[-1])}
    )

    new_articles = [
        ex
        for title, ex in articles_dict.items()
        if title not in old_articles_dict.keys()
    ]

    old_articles = [
        ex for title, ex in articles_dict.items() if title in old_articles_dict.keys()
    ]

    if old_articles:
        console.log(
            f"Found {len(old_articles)} old articles in current batch. Skipping."
        )

    if new_articles:
        console.log(
            f"Found {len(new_articles)} new articles in current batch to write."
        )

        filename = str(dt.datetime.now())[:10] + ".jsonl"
        srsly.write_jsonl(Path("data") / "downloads" / filename, new_articles)
        console.log(f"Wrote {len(new_articles)} articles into {filename}.")

    return True


if __name__ == "__main__":
    retrieve_articles()
