"""
app/ingestion/wiki_api_fetcher.py
==================================
Fetch Wikipedia articles by title using the MediaWiki Action API.

Why this beats downloading 20 GB of Parquet shards:
  • Fetches ONLY the articles you need — no wasted bandwidth
  • Batches 50 titles per HTTP request → 2 000 titles = ~40 requests total
  • Rate limiting is a non-issue at that volume
  • Fully resumable — skips already-cached titles on every run
  • Retries with exponential backoff on any connection failure

Empty-extract fallback:
  Some articles return an empty extract from the extracts API — this happens
  for redirects that weren't fully resolved, short stubs, or non-English
  content. For those, we fall back to fetching raw wikitext via prop=revisions
  and parse out the plain text ourselves, which always returns content.

API limits (with a proper User-Agent header):
  • Action API  : up to 200 req/s  (we stay well under at ~1 req/s)
  • Batch size  : 50 titles/request (hard limit enforced by MediaWiki)
  • No key needed for read-only access
"""

import json
import re
import time
import requests

from pathlib import Path
from tqdm import tqdm

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MEDIAWIKI_API_URL  = "https://en.wikipedia.org/w/api.php"
BATCH_SIZE         = 50       # MediaWiki hard limit per request
DELAY_BETWEEN_REQS = 0.5      # seconds between batches (polite, well under rate limit)
MAX_RETRIES        = 6        # attempts per batch before giving up
BACKOFF_BASE       = 2        # exponential backoff: 2s, 4s, 8s, 16s, 32s, 64s

# Identify your bot — Wikipedia blocks requests without a descriptive User-Agent
USER_AGENT = (
    "WikiRAGFetcher/1.0 "
    "(RAG evaluation research; your-email@example.com)"   # ← update your email
)


# ─────────────────────────────────────────────────────────────────────────────
# Wikitext cleaner (used only in the empty-extract fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_wikitext(raw: str) -> str:
    """
    Strip the most common wikitext markup and return readable plain text.
    Not a full parser — covers 95 %+ of real article content.
    """
    # Remove templates  {{...}}  (can be nested)
    while re.search(r'\{\{[^{}]*\}\}', raw):
        raw = re.sub(r'\{\{[^{}]*\}\}', '', raw)
    # Remove [[File:...]] / [[Image:...]] blocks
    raw = re.sub(r'\[\[(?:File|Image):[^\]]*\]\]', '', raw, flags=re.IGNORECASE)
    # Convert [[link|display]] → display,  [[link]] → link
    raw = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', raw)
    # Remove <ref>...</ref> and self-closing <ref ... />
    raw = re.sub(r'<ref[^>]*/>', '', raw)
    raw = re.sub(r'<ref[^>]*>.*?</ref>', '', raw, flags=re.DOTALL)
    # Remove remaining HTML tags
    raw = re.sub(r'<[^>]+>', '', raw)
    # Remove bold/italic markup
    raw = re.sub(r"'{2,3}", '', raw)
    # Remove section headers === ... ===
    raw = re.sub(r'={2,6}[^=]+=+', '', raw)
    # Collapse whitespace
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    return raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: fetch raw wikitext for a single title
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_wikitext_fallback(title: str, session: requests.Session) -> str | None:
    """
    Fetch the latest wikitext revision for a single article and return
    cleaned plain text.  Used when the extracts API returns an empty string.
    """
    params = {
        "action":    "query",
        "titles":    title,
        "prop":      "revisions",
        "rvprop":    "content",
        "rvslots":   "main",
        "redirects": "1",
        "format":    "json",
        "formatversion": "2",
    }
    try:
        resp = session.get(MEDIAWIKI_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return None
        slots = pages[0].get("revisions", [{}])[0].get("slots", {})
        raw   = slots.get("main", {}).get("content", "")
        return _clean_wikitext(raw) or None
    except Exception as e:
        logger.warning(f"  Wikitext fallback failed for '{title}': {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Single-batch fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_batch(titles: list[str], session: requests.Session) -> list[dict]:
    """
    Fetch up to 50 Wikipedia articles in one API call.

    Uses action=query with prop=extracts (plain text) + info (URL).
    Handles redirects automatically (e.g. "ML" → "Machine learning").
    Retries with exponential backoff on any network or server error.

    For any article whose extract is empty, automatically retries via
    the wikitext fallback (_fetch_wikitext_fallback) before giving up.

    Returns a list of article dicts:
        {"title": str, "text": str, "url": str}
    Truly missing pages are skipped with a warning.
    """
    params = {
        "action":          "query",
        "titles":          "|".join(titles),
        "prop":            "extracts|info",
        "explaintext":     "1",          # MUST be "1" not True (requests serialises True as "True" which MediaWiki ignores)
        "exsectionformat": "plain",
        "inprop":          "url",
        "redirects":       "1",          # same — "1" not True
        "format":          "json",
        "formatversion":   "2",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(MEDIAWIKI_API_URL, params=params, timeout=30)

            # Rate limited — back off and retry
            if resp.status_code == 429:
                wait        = BACKOFF_BASE ** attempt
                retry_after = int(resp.headers.get("Retry-After", wait))
                logger.warning(f"Rate limited — waiting {retry_after}s (attempt {attempt})")
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            data  = resp.json()
            pages = data.get("query", {}).get("pages", [])

            results = []
            for page in pages:
                if page.get("missing"):
                    logger.warning(f"  Not found in Wikipedia: '{page.get('title')}'")
                    continue

                title = page["title"]
                url   = page.get("fullurl", "")
                text  = page.get("extract", "").strip()

                # ── Empty extract fallback ────────────────────────────────────
                if not text:
                    logger.debug(
                        f"  Empty extract for '{title}' — trying wikitext fallback."
                    )
                    text = _fetch_wikitext_fallback(title, session) or ""
                    if text:
                        logger.info(f"  Wikitext fallback succeeded for '{title}'.")
                    else:
                        logger.warning(
                            f"  No content found for '{title}' via either method — skipping."
                        )
                        continue

                results.append({"title": title, "text": text, "url": url})
            return results

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            wait = BACKOFF_BASE ** attempt
            logger.warning(
                f"Connection error (attempt {attempt}/{MAX_RETRIES}): {e} "
                f"— retrying in {wait}s"
            )
            time.sleep(wait)

        except Exception as e:
            logger.error(f"Unexpected error fetching batch: {e}")
            break

    logger.error(f"Giving up on batch after {MAX_RETRIES} attempts: {titles[:3]}...")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_title(title: str) -> str:
    import re
    return re.sub(r'[^\w\s-]', '_', title).strip().replace(' ', '_')


def _cache_path(title: str, cache_dir: Path) -> Path:
    return cache_dir / f"{_safe_title(title)}.json"


def _write_to_cache(article: dict, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(article["title"], cache_dir).write_text(
        json.dumps(article, ensure_ascii=False), encoding="utf-8"
    )


def _already_cached(titles: list[str], cache_dir: Path) -> set[str]:
    """Return the subset of titles that already exist in the cache."""
    return {t for t in titles if _cache_path(t, cache_dir).exists()}


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_wikipedia_articles(titles: list[str], cache_dir: Path) -> None:
    """
    Fetch Wikipedia articles for all `titles` and persist each one as a
    JSON file under `cache_dir`.

    Features:
      • Skips titles already present in cache (fully idempotent / resumable)
      • Batches 50 titles per API request  → ~40 requests for 2 000 titles
      • Exponential backoff on failures
      • Reports missing titles at the end

    Args:
        titles:    List of Wikipedia article titles to fetch.
        cache_dir: Directory where per-article JSON files are written.
    """
    cached   = _already_cached(titles, cache_dir)
    needed   = [t for t in titles if t not in cached]

    logger.info(
        f"Wikipedia API fetch — "
        f"{len(cached)} already cached, {len(needed)} to fetch."
    )

    if not needed:
        logger.info("All articles already in cache — nothing to do.")
        return

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Split into batches of 50
    batches   = [needed[i : i + BATCH_SIZE] for i in range(0, len(needed), BATCH_SIZE)]
    not_found = []
    fetched   = 0

    for batch in tqdm(batches, desc="Fetching Wikipedia batches", unit="batch"):
        articles = _fetch_batch(batch, session)

        # Write every article we got back
        for article in articles:
            _write_to_cache(article, cache_dir)
            fetched += 1

        # Any title in the batch that didn't come back is missing
        returned_titles = {a["title"] for a in articles}
        for title in batch:
            if title not in returned_titles:
                not_found.append(title)

        time.sleep(DELAY_BETWEEN_REQS)

    # ── Final report ─────────────────────────────────────────────────────────
    logger.info(f"Done — fetched {fetched}/{len(needed)} articles.")

    if not_found:
        missing_file = cache_dir / "missing_titles.txt"
        missing_file.write_text("\n".join(sorted(not_found)), encoding="utf-8")
        logger.warning(
            f"{len(not_found)} titles not found in Wikipedia → {missing_file}"
        )