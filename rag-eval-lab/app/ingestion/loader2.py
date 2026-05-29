from dataclasses import dataclass
from pathlib import Path
import json

from llama_index.core import Document
from app.tracing.langfuse import LangfuseTracer, get_safe_tracer
from .loader_helper import (
    STAGE_CONFIG, 
    _load_hotpotqa_from_cache, 
    _extract_unique_titles,
    _cache_path,
)

from app.utils.logger import get_logger
logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — DataLoader  (cache-only, no network)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentLoader:
    """
    PHASE 2 — Load data from local cache for downstream RAG tasks.

    Never makes network calls. Assumes build_cache() has already been run.
    """

    def __init__(self, tracer: LangfuseTracer | None = None) -> None:
        self.tracer = get_safe_tracer(tracer)

    def load_stage(self, stage: int) -> tuple[list[Document], list[dict]]:
        """
        Load corpus and golden QA set for a given evaluation stage (1, 2, or 3).

        Returns:
            documents:      list[Document]  — full-text Wikipedia articles as LlamaIndex Documents
            golden_dataset: list[dict]      — QA items with question, answer, titles, references
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
                input={"stage": stage, "config": str(cfg)},
            )
            if self.tracer else None
        )

        try:
            golden_dataset = _load_hotpotqa_from_cache(cfg.easy, cfg.medium, cfg.hard)
            titles         = _extract_unique_titles(golden_dataset)
            documents      = self._load_documents(titles)

            logger.info(
                f"Stage {stage} loaded — "
                f"{len(documents)} documents, {len(golden_dataset)} QA pairs."
            )

            if span_ctx:
                span_ctx.update(output={
                    "documents":  len(documents),
                    "qa_pairs":   len(golden_dataset),
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
        Load Wikipedia Documents from the local cache for the given titles.
        Logs a warning for any titles not found — does NOT raise.
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