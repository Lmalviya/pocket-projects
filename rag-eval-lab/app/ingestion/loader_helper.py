"""
app/ingestion/loader.py
========================

Two-phase data layer for Wikipedia + HotpotQA.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — build_cache()          (call once from main, slow, idempotent)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Step 1 — Stream HotpotQA distractor split.
            Collect up to (max_easy + max_medium + max_hard) QA items.
            Persist as data/hotpot_cache/{easy,medium,hard}.json.

  Step 2 — Extract every unique Wikipedia title referenced by those items.

  Step 3 — Fetch ONLY those Wikipedia articles via the MediaWiki Action API:
              • 50 titles per request  → ~40 requests for 2 000 titles
              • Exponential backoff on connection failures / rate limits
              • Skips already-cached titles  → fully resumable
              • No Parquet shards, no 20 GB download

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — DataLoader.load_stage()  (call anytime, fast, cache-only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Reads strictly from local cache — no network calls.
  Slices the golden dataset for the requested stage, loads the
  corresponding Wikipedia Documents from disk.
  Returns (documents, golden_dataset) ready for the RAG pipeline.
"""

import json
import os
import re

from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from llama_index.core import Document

from app.utils.wiki_api_fetcher import fetch_wikipedia_articles
from app.tracing.langfuse import LangfuseTracer, get_safe_tracer
from app.utils.logger import get_logger
from app.config.settings import get_settings

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stage configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Stage:
    easy: int
    medium: int
    hard: int

    @property
    def total(self) -> int:
        return self.easy + self.medium + self.hard


@dataclass(frozen=True)
class StageConfig:
    """
    QA item counts per difficulty level for each evaluation stage.
    Stage 3 values define the maximum collected during build_cache().
    """
    stage_1: Stage = Stage(easy=50,  medium=37,  hard=38)
    stage_2: Stage = Stage(easy=150, medium=175, hard=175)
    stage_3: Stage = Stage(easy=300, medium=450, hard=450)   # ← cache ceiling


STAGE_CONFIG = StageConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

WIKIPEDIA_CACHE_DIR = Path("data/wikipedia_cache")
HOTPOTQA_CACHE_DIR  = Path("data/hotpot_cache")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — title / cache utils
# ─────────────────────────────────────────────────────────────────────────────

def _safe_title(title: str) -> str:
    """Convert a Wikipedia title into a safe filename."""
    return re.sub(r'[^\w\s-]', '_', title).strip().replace(' ', '_')


def _cache_path(title: str) -> Path:
    return WIKIPEDIA_CACHE_DIR / f"{_safe_title(title)}.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — HotpotQA
# ─────────────────────────────────────────────────────────────────────────────

def _stream_and_cache_hotpotqa(max_easy: int, max_medium: int, max_hard: int) -> None:
    """
    Stream HotpotQA distractor split from HuggingFace and persist
    easy / medium / hard JSON files to HOTPOTQA_CACHE_DIR.
    Skipped automatically if the cache files already exist.
    """
    HOTPOTQA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if all((HOTPOTQA_CACHE_DIR / f"{lvl}.json").exists() for lvl in ("easy", "medium", "hard")):
        logger.info("HotpotQA cache already populated — skipping download.")
        return

    logger.info(
        f"Streaming HotpotQA "
        f"(max easy={max_easy}, medium={max_medium}, hard={max_hard})..."
    )

    from datasets import load_dataset
    easy, medium, hard = [], [], []

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="train",
        streaming=True,
        token=get_settings().hf_token,
    )

    for sample in tqdm(dataset, desc="HotpotQA"):
        ctx  = sample["context"]
        item = {
            "id":         sample.get("id", ""),
            "question":   sample.get("question", ""),
            "answer":     sample.get("answer", ""),
            "type":       sample.get("type", ""),
            "level":      sample.get("level", ""),
            "titles":     ctx.get("title", []),
            "references": ["\n".join(s) for s in ctx.get("sentences", [])],
        }

        level = item["level"]
        if   level == "easy"   and len(easy)   < max_easy:   easy.append(item)
        elif level == "medium" and len(medium) < max_medium: medium.append(item)
        elif level == "hard"   and len(hard)   < max_hard:   hard.append(item)

        if len(easy) >= max_easy and len(medium) >= max_medium and len(hard) >= max_hard:
            break

    for name, data in [("easy", easy), ("medium", medium), ("hard", hard)]:
        path = HOTPOTQA_CACHE_DIR / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    logger.info(
        f"HotpotQA cached — "
        f"easy: {len(easy)}, medium: {len(medium)}, hard: {len(hard)}"
    )


def _load_hotpotqa_from_cache(num_easy: int, num_medium: int, num_hard: int) -> list[dict]:
    """Read slices of the cached HotpotQA files and return a flat list of QA items."""
    result = []
    for name, count in [("easy", num_easy), ("medium", num_medium), ("hard", num_hard)]:
        path = HOTPOTQA_CACHE_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"HotpotQA cache file not found: {path}. Run build_cache() first."
            )
        result.extend(json.loads(path.read_text(encoding="utf-8"))[:count])
    return result


def _extract_unique_titles(golden_dataset: list[dict]) -> list[str]:
    """Collect every unique Wikipedia title referenced across a golden dataset."""
    return list({title for item in golden_dataset for title in item["titles"]})


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — build_cache()
# ─────────────────────────────────────────────────────────────────────────────

def build_cache(
    max_easy:   int = STAGE_CONFIG.stage_3.easy,
    max_medium: int = STAGE_CONFIG.stage_3.medium,
    max_hard:   int = STAGE_CONFIG.stage_3.hard,
) -> None:
    """
    PHASE 1 — Call once from main before anything else.

    Step 1: Stream HotpotQA → cache up to (max_easy + max_medium + max_hard) items.
    Step 2: Extract all unique Wikipedia titles from those items.
    Step 3: Fetch those Wikipedia articles via API (batched, resumable, no 20GB download).

    Fully idempotent — safe to re-run if interrupted at any point.

    Args:
        max_easy:   Max easy QA items to collect   (default: 300)
        max_medium: Max medium QA items to collect (default: 450)
        max_hard:   Max hard QA items to collect   (default: 450)
    """
    total = max_easy + max_medium + max_hard
    logger.info(
        f"━━ build_cache() — target {total} QA items "
        f"(easy={max_easy}, medium={max_medium}, hard={max_hard})"
    )

    # Step 1: HotpotQA
    _stream_and_cache_hotpotqa(max_easy, max_medium, max_hard)

    # Step 2: Collect all Wikipedia titles we'll ever need (across all stages)
    full_dataset  = _load_hotpotqa_from_cache(max_easy, max_medium, max_hard)
    unique_titles = _extract_unique_titles(full_dataset)
    logger.info(f"Unique Wikipedia titles needed: {len(unique_titles)}")

    # Step 3: Fetch from Wikipedia API (batched 50/request, ~40 requests total)
    fetch_wikipedia_articles(
        titles=unique_titles,
        cache_dir=WIKIPEDIA_CACHE_DIR,
    )

    logger.info("━━ build_cache() complete. Data layer is ready.")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — DataLoader  (cache-only, no network)
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    PHASE 2 — Load data from local cache for downstream RAG tasks.

    Never makes network calls. Assumes build_cache() has already been run.
    """

    def __init__(self, tracer: LangfuseTracer | None = None) -> None:
        self.tracer = get_safe_tracer(tracer)

    def load_stage(self, stage: int) -> tuple[list[Document], list[dict]]:
        """
        Load the corpus and golden QA set for a given evaluation stage (1, 2, or 3).

        Returns:
            documents:      list[Document]  — Wikipedia articles as LlamaIndex Documents
            golden_dataset: list[dict]      — QA items with question/answer/titles/references
        """
        stage_map = {
            1: STAGE_CONFIG.stage_1,
            2: STAGE_CONFIG.stage_2,
            3: STAGE_CONFIG.stage_3,
        }
        if stage not in stage_map:
            raise ValueError(f"Invalid stage '{stage}'. Choose 1, 2, or 3.")

        cfg = stage_map[stage]
        logger.info(
            f"Loading stage {stage} — "
            f"easy={cfg.easy}, medium={cfg.medium}, hard={cfg.hard} "
            f"(total={cfg.total})"
        )

        span_ctx = (
            self.tracer.span(
                name=f"dataloader.load_stage_{stage}",
                input={"stage": stage, "easy": cfg.easy,
                       "medium": cfg.medium, "hard": cfg.hard},
            )
            if self.tracer else None
        )

        try:
            golden_dataset = _load_hotpotqa_from_cache(cfg.easy, cfg.medium, cfg.hard)
            titles         = _extract_unique_titles(golden_dataset)
            documents      = self._load_documents(titles)

            logger.info(
                f"Stage {stage} ready — "
                f"{len(documents)} documents, {len(golden_dataset)} QA pairs."
            )

            if span_ctx:
                span_ctx.update(output={
                    "documents": len(documents),
                    "qa_pairs":  len(golden_dataset),
                })

            return documents, golden_dataset

        except Exception as e:
            err = f"Failed to load stage {stage}: {e}"
            logger.error(err)
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=err)
            raise

    def _load_documents(self, titles: list[str]) -> list[Document]:
        """
        Load Wikipedia Documents from the local cache.
        Warns on missing titles — does not raise, so partial results still work.
        """
        documents = []
        missing   = []

        for title in titles:
            path = _cache_path(title)
            if not path.exists():
                missing.append(title)
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                documents.append(Document(
                    text=data["text"],
                    metadata={"title": data["title"], "url": data.get("url", "")},
                ))
            except Exception as e:
                logger.warning(f"Could not read cache for '{title}': {e}")
                missing.append(title)

        if missing:
            logger.warning(
                f"{len(missing)} articles missing from cache "
                f"(run build_cache() to fix): {missing[:5]}"
            )

        return documents