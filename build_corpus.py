import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import pickle

START_URLS = ["https://en.wikipedia.org/wiki/Python_(programming_language)"]
CRAWL_LIMIT = 100
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

texts = crawl_urls(START_URLS, CRAWL_LIMIT)
with open("texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print(f"[DONE] Scraped {len(texts)} pages")