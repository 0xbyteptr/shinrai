import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import pickle
import argparse

START_URLS = ["https://en.wikipedia.org/"]
CRAWL_LIMIT = 20
HEADERS = {
    "User-Agent": "ShinraiBot/1.0 (https://shinrai.wtf; contact: me@byteptr.xyz)"
}

def is_valid_wiki_url(url):
    parsed = urlparse(url)
    if not parsed.path.startswith("/wiki/"):
        return False
    if ":" in parsed.path:
        return False
    if "#" in parsed.path:
        return False
    return True

def scrape_text(url):
    print(f"[DEBUG] Requesting {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Znajdź tylko główny content artykułu
        content_div = soup.find("div", {"id": "mw-content-text"})
        if content_div:
            paragraphs = content_div.find_all("p")
            text = " ".join([p.get_text() for p in paragraphs])
            print(f"[DEBUG] Got {len(text.split())} words from {url}")
            return text
        else:
            print(f"[DEBUG] No content div found in {url}")
            return ""
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed {url}: {e}")
        return ""

def crawl_urls(start_urls, limit):
    queue = start_urls.copy()
    visited = set()
    texts = []
    while queue and len(texts) < limit:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        text = scrape_text(url)
        if text:
            texts.append(text)
        # znajdź linki w tym artykule
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            links_added = 0
            content_div = soup.find("div", {"id": "mw-content-text"})
            if content_div:
                for a in content_div.find_all("a", href=True):
                    full_url = urljoin(url, a["href"])
                    if full_url.startswith("https://en.wikipedia.org") and full_url not in visited and is_valid_wiki_url(full_url):
                        queue.append(full_url)
                        links_added += 1
            print(f"[DEBUG] Added {links_added} links from {url}")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Link fetch failed {url}: {e}")
        time.sleep(1)
    return texts[:limit]



def load_hf_dataset(spec):
    """Load a dataset from HuggingFace `datasets` library.

    ``spec`` may be a dataset name (e.g. "Xennon-BD/Alpaca-uncensored") or a
    local directory/JSON file that ``load_dataset`` understands.  We iterate
    through every split and concatenate any text-like fields we recognise.
    Empty records are silently skipped, which covers entries such as::

        {"instruction": "", "input": "", "output": ""}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets library not installed; run `pip install datasets`")

    ds = load_dataset(spec)
    texts = []
    for split, dataset in ds.items():
        for item in dataset:
            if isinstance(item, str):
                texts.append(item)
                continue
            # prefer the common keys, fall back to any string fields
            pieces = []
            for key in ("instruction", "input", "output", "text"):
                v = item.get(key)
                if isinstance(v, str) and v.strip():
                    pieces.append(v.strip())
            # if we still have nothing, try to grab any str field
            if not pieces:
                for v in item.values():
                    if isinstance(v, str) and v.strip():
                        pieces.append(v.strip())
                        break
            text = "\n".join(pieces).strip()
            if text:
                texts.append(text)
    return texts


def load_json_file(path):
    """Read a local JSON file containing a list of examples.

    The file may be either a proper JSON list or newline‑delimited JSON.  Each
    element is handled the same way as in :func:`load_hf_dataset`.
    """
    import json
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
        else:
            # assume one JSON object per line
            data = [json.loads(l) for l in f if l.strip()]
    for item in data:
        if isinstance(item, str):
            texts.append(item)
            continue
        pieces = []
        for key in ("instruction", "input", "output", "text"):
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                pieces.append(v.strip())
        if not pieces:
            for v in item.values():
                if isinstance(v, str) and v.strip():
                    pieces.append(v.strip())
                    break
        text = "\n".join(pieces).strip()
        if text:
            texts.append(text)
    return texts


def main(start_urls=None, limit=None, out_file="texts.pkl", hf=None, json_file=None):
    """Crawl a set of Wikipedia URLs **or** import a dataset and dump texts.

    ``hf`` is a HuggingFace dataset spec (see :func:`load_hf_dataset`).
    ``json_file`` points to a local JSON/NDJSON file with the same structure.

    If either ``hf`` or ``json_file`` is given the crawler is skipped; the
    corresponding loader will produce the list of documents instead.

    Args:
        start_urls: iterable of initial URLs to visit
        limit: maximum number of pages to scrape (only used with crawling)
        out_file: where to save the list of page texts (pickle)
        hf: HuggingFace dataset name or path
        json_file: path to a local JSON file
    """
    if hf or json_file:
        if hf and json_file:
            raise ValueError("Cannot specify both --hf and --json")
        if hf:
            texts = load_hf_dataset(hf)
            print(f"[INFO] Loaded {len(texts)} examples from HF dataset '{hf}'")
        else:
            texts = load_json_file(json_file)
            print(f"[INFO] Loaded {len(texts)} examples from {json_file}")
    else:
        start_urls = start_urls or START_URLS
        limit = limit or CRAWL_LIMIT
        texts = crawl_urls(start_urls, limit)
        print(f"[INFO] Crawled {len(texts)} pages")

    with open(out_file, "wb") as f:
        pickle.dump(texts, f)
    print(f"[DONE] Wrote {len(texts)} documents → {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape Wikipedia and build a corpus or import a dataset.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--start", nargs="+", default=START_URLS,
                       help="Initial URLs to crawl (default: %(default)s)")
    group.add_argument("--hf", help="HuggingFace dataset name or path to load")
    group.add_argument("--json", help="Path to a local JSON/NDJSON dataset file")

    parser.add_argument("--limit", type=int, default=CRAWL_LIMIT,
                        help="Maximum number of pages to scrape (ignored when --hf/--json given)")
    parser.add_argument("--out", default="texts.pkl",
                        help="Output pickle for scraped texts")
    args = parser.parse_args()
    main(start_urls=args.start, limit=args.limit, out_file=args.out,
         hf=args.hf, json_file=args.json)
