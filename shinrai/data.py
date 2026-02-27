"""Data acquisition and PyTorch dataset.

Responsibilities:
  - Fetch text from a URL
  - Crawl a domain up to a given depth
  - Load a local .txt or .pdf file
  - Build a character vocabulary
  - Provide a ``CharDataset`` (sliding-window next-char prediction)
"""

from __future__ import annotations

import os
from collections import deque
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
import torch
from bs4 import BeautifulSoup
from torch.utils.data import Dataset

from shinrai.logging import log

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None  # type: ignore[assignment]


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SEED_URLS: list[str] = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://en.wikipedia.org/wiki/Computer_vision",
    "https://en.wikipedia.org/wiki/Neural_network",
    "https://en.wikipedia.org/wiki/Philosophy",
    "https://en.wikipedia.org/wiki/Mathematics",
]

_HTTP_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0 Safari/537.36"
    )
}

_MIN_TEXT_LENGTH = 50  # chars; pages shorter than this are skipped


# ──────────────────────────────────────────────────────────────────────────────
# Text acquisition
# ──────────────────────────────────────────────────────────────────────────────

def _extract_paragraphs(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(p.get_text() for p in soup.find_all("p")).lower()


def fetch_text(url: str, timeout: int = 10) -> str:
    """Return lowercased paragraph text from *url*, or empty string on failure."""
    try:
        r = requests.get(url, headers=_HTTP_HEADERS, timeout=timeout)
        r.raise_for_status()
        return _extract_paragraphs(r.text)
    except Exception as exc:
        log.warn(f"fetch_text({url}): {exc}")
        return ""


def fetch_seed_articles() -> str:
    """Fetch all curated seed articles and concatenate their text."""
    texts: list[str] = []
    for url in SEED_URLS:
        text = fetch_text(url)
        if len(text) > _MIN_TEXT_LENGTH:
            texts.append(text)
            log.info(f"  ✓ {url}  ({len(text):,} chars)")
        else:
            log.warn(f"  ✗ {url}  (skipped — too short)")
    return "\n".join(texts)


def crawl_text(
    start_url: str,
    max_pages: int = 20,
    max_depth: int = 1,
    timeout: int = 10,
) -> str:
    """BFS crawl within the same domain.  Returns concatenated paragraph text."""
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(start_url, 0)])
    start_netloc = urlparse(start_url).netloc
    session = requests.Session()
    texts: list[str] = []

    while queue and len(visited) < max_pages:
        url, depth = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            r = session.get(url, headers=_HTTP_HEADERS, timeout=timeout)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            text = " ".join(p.get_text() for p in soup.find_all("p")).lower()
            if len(text.strip()) > _MIN_TEXT_LENGTH:
                texts.append(text)
                log.info(f"  crawled: {url}  ({len(text):,} chars)")

            if depth < max_depth:
                for a in soup.find_all("a", href=True):
                    href = urljoin(url, a["href"])
                    parsed = urlparse(href)
                    if (
                        parsed.scheme in ("http", "https")
                        and parsed.netloc == start_netloc
                        and href not in visited
                    ):
                        queue.append((href, depth + 1))
        except Exception as exc:
            log.warn(f"  crawl error at {url}: {exc}")

    return "\n".join(texts)


def load_local_file(path: str) -> str:
    """Load a .txt or .pdf file and return lowercased text."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.lower().endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError(
                "PyPDF2 is not installed.  Run: pip install PyPDF2"
            )
        reader = PdfReader(path)
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(pages).lower()

    with open(path, "r", encoding="utf-8") as f:
        return f.read().lower()


# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    """Bidirectional character ↔ index mapping."""

    def __init__(self, chars: list[str]) -> None:
        self.chars = chars
        self.char2idx: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.idx2char: dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        unk = self.char2idx.get(" ", 0)
        return [self.char2idx.get(ch, unk) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.idx2char.get(i, "?") for i in ids)

    @classmethod
    def from_text(cls, text: str) -> "Vocabulary":
        return cls(sorted(set(text)))

    @classmethod
    def from_chars(cls, chars: list[str]) -> "Vocabulary":
        return cls(chars)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class CharDataset(Dataset):
    """Sliding-window character dataset.

    Each item is ``(input_ids, target_ids)`` of length ``seq_length``.
    Target is input shifted one position to the right, so the model predicts
    every character in the window, not just the last.
    """

    def __init__(self, encoded: list[int], seq_length: int) -> None:
        self.seq_length = seq_length
        self.data = encoded

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y


# ──────────────────────────────────────────────────────────────────────────────
# High-level helper
# ──────────────────────────────────────────────────────────────────────────────

_FALLBACK_TEXT = "the quick brown fox jumps over the lazy dog. " * 600


def acquire_text(
    text_file: Optional[str] = None,
    use_seed_articles: bool = False,
    crawl: bool = False,
    url: str = SEED_URLS[0],
    crawl_depth: int = 1,
    max_pages: int = 20,
    min_length: int = 200,
) -> str:
    """Route to the correct acquisition strategy and return raw text."""
    text = ""

    if text_file:
        log.info(f"Loading local file: {text_file}")
        try:
            text = load_local_file(text_file)
        except Exception as exc:
            log.error(f"Could not read file: {exc}")

    elif use_seed_articles:
        log.info("Fetching curated seed articles…")
        text = fetch_seed_articles()

    elif crawl:
        log.info(f"Crawling {url}  (depth={crawl_depth}, max={max_pages} pages)…")
        text = crawl_text(url, max_pages=max_pages, max_depth=crawl_depth)

    else:
        log.info(f"Fetching {url}…")
        text = fetch_text(url)

    if len(text) < min_length:
        log.warn("Acquired text is too short; using built-in fallback text.")
        text = _FALLBACK_TEXT

    return text
