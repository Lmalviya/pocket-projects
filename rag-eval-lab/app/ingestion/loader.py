"""
app/ingestion/loader.py
========================
Data ingestion loader for Wikipedia and HotpotQA datasets.

📚 LESSON — Ingestion, Chunking, and Evaluation Alignment:
In a RAG evaluation lab, we must benchmark our chunkers (Fixed, Recursive, Semantic)
on GENUINE, long-form documents. 
- Using pre-extracted short paragraphs (snippets) from datasets directly is a mistake,
  as they are already "pre-chunked".
- We systematically extract the TITLES of the Ground Truth (gold) and distractor articles
  needed for each stage, and download their COMPLETE, full-text Wikipedia pages.
- The chunking layer splits these complete, long-form pages into TextNodes.
- The golden dataset JSON includes the questions, answers, gold titles, AND the
  original context sentences (snippets) so that evaluators can verify retrieval accuracy.
"""

from dataclasses import dataclass
from tqdm import tqdm
import json
import os
import random
import time
import re
from typing import Any
from pathlib import Path

from datasets import load_dataset
from llama_index.core import Document
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.readers.wikipedia import WikipediaReader

from app.tracing.langfuse import LangfuseTracer, get_safe_tracer
from app.utils.logger import get_logger
from app.config.settings import get_settings
logger = get_logger(__name__)

@dataclass(frozen=True)
class Stage:
    easy: int
    medium: int
    hard: int

@dataclass
class StageConfig:
    stage_1: Stage = Stage(easy=50, medium=37, hard=38)
    stage_2: Stage = Stage(easy=150, medium=175, hard=175)
    stage_3: Stage = Stage(easy=300, medium=450, hard=450)

# ── Checkpoint helpers ────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("wiki_articles")          # where .json files are saved
CHECKPOINT   = Path("checkpoint.json")        # tracks completed titles

def load_checkpoint() -> set:
    if CHECKPOINT.exists():
        data = json.loads(CHECKPOINT.read_text())
        logger.info(f"Resuming — {len(data['done'])} titles already done")
        return set(data["done"])
    return set()
 
def save_checkpoint(done: set):
    CHECKPOINT.write_text(json.dumps({"done": list(done)}, indent=2))
 

class DocumentLoader:
    """
    Handles loading raw document corpora from Wikipedia, directories, and HotpotQA.
    Compiles fully-aligned golden QA sets and long-form Wikipedia article indexes.
    """

    def __init__(self, tracer: LangfuseTracer | None = None) -> None:
        """
        Initialize the loader with optional tracer.
        """
        self.tracer = get_safe_tracer(tracer)
        self.wikipedia_cache = "data/wikipedia_cache"
        self.hotpotqa_cache = "data/hotpot_cache"

    def get_safe_title(self, title: str) -> str:
        return re.sub(r'[^\w\s-]', '_', title).strip().replace(' ', '_')
        
    def _load_wikipedia_cache(self) -> None:
        cache_dir = Path(self.wikipedia_cache)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_files = list(cache_dir.glob("*.json"))
        if len(cache_files) > 0:
            logger.info("Wikipedia granular cache already populated with {count} files.", count=len(cache_files))
            return

        try:
            logger.info("Wikipedia granular cache is empty. Initiating cache setup...")
            dataset = load_dataset(
                "wikimedia/wikipedia",
                "20231101.en", 
                split="train",
                # streaming=True,
                cache_dir=Path("data/hf_cache"),
                token=get_settings().hf_token
            )
            
            logger.info("Downloading Wikipedia articles. Writing directly to granular cache...")
            for sample in tqdm(dataset,  desc="Processing Wikipedia"):
                title = sample.get("title")
                text = sample.get("text")
                if title and text:
                    # 1. Directly save to granular cache!
                    safe_title = self.get_safe_title(title)
                    cache_path = cache_dir / f"{safe_title}.json"
                    cached_data = {
                        "title": title,
                        "text": text,
                        "url": sample.get("url", None)
                    }
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cached_data, f, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to stream Wikipedia dataset from Hugging Face Hub: {err}", err=str(e))
            raise ConnectionError(f"❌ Failed to stream Wikipedia dataset from Hugging Face Hub: {str(e)}") from e


    def _create_hotpotqa_cache(self) -> None:
        cache_dir = Path(self.hotpotqa_cache)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_files = list(cache_dir.glob("*.json"))
        if len(cache_files) > 0:
            logger.info("HotpotQA cache already populated with {count} files.", count=len(cache_files))
            return

        logger.info("Local HotpotQA dataset not found under '{path}'. Downloading from Hugging Face...", path=cache_dir)
        logger.info("Streaming HotpotQA 'distractor' dataset from HF Hub to compile local cache...")
        try:
            easy = []
            medium = []
            hard = []
            stream_dataset = load_dataset(
                "hotpotqa/hotpot_qa", 
                "distractor", 
                split="train", 
                streaming=True,
                token=get_settings().hf_token
            )
            for sample in tqdm(stream_dataset, desc="Processing HotpotQA"):
                context = sample["context"]
                item = {
                    "id": sample.get("id", ""),
                    "question": sample.get("question", ""),
                    "answer": sample.get("answer", ""),
                    "type": sample.get("type", ""),
                    "level": sample.get("level", ""),
                    "titles": context.get("title", []),
                    "references": ["\n".join(sentence) for sentence in context.get("sentences", [])] 
                }

                max_easy = StageConfig.stage_3.easy
                max_medium = StageConfig.stage_3.medium
                max_hard = StageConfig.stage_3.hard
                if item["level"] == "easy" and len(easy) <= max_easy:
                    easy.append(item)
                elif item["level"] == "medium" and len(medium) <= max_medium:
                    medium.append(item)
                elif item["level"] == "hard" and len(hard) <= max_hard:
                    hard.append(item)
                
                if len(easy) > max_easy and len(medium) > max_hard and len(hard) > max_hard:
                    break
            
            with open(cache_dir / "easy.json", "w", encoding="utf-8") as f:
                json.dump(easy, f, ensure_ascii=False)
            
            with open(cache_dir / "medium.json", "w", encoding="utf-8") as f:
                json.dump(medium, f, ensure_ascii=False)
            
            with open(cache_dir / "hard.json", "w", encoding="utf-8") as f:
                json.dump(hard, f, ensure_ascii=False)
                
            logger.info("Successfully downloaded and cached HotpotQA dataset locally under '{path}'", path=cache_dir)
        except Exception as hf_err:
            err_msg = f"❌ Failed to load or cache HotpotQA dataset: {str(hf_err)}"
            logger.error(err_msg)
            raise ConnectionError(err_msg) from hf_err
        
    def load_from_wikipedia_cache(self, titles: list[str]) -> list[Document]:
        """
        Loads full-text articles strictly from the local granular file-based cache.
        All requested articles must exist in the cache.
        """
        logger.info("Loading {count} Wikipedia articles from granular cache...", count=len(titles))
        span_ctx = self.tracer.span(name="ingestion.load.wikipedia_cache", input={"titles": titles, "count": len(titles)}) if self.tracer else None

        try:
            documents = []
            cache_dir = Path(self.wikipedia_cache)
 
            missing_titles = []
            for title in titles:
                safe_title = self.get_safe_title(title)
                cache_path = cache_dir / f"{safe_title}.json"
                
                if cache_path.exists():
                    try:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            cached_data = json.load(f)
                        doc = Document(
                            text=cached_data["text"],
                            metadata=cached_data.get("metadata", {"title": title, "file_name": f"{title}.txt"})
                        )
                        documents.append(doc)
                    except Exception as err:
                        logger.warning("Failed to read cache file for '{title}': {err}", title=title, err=str(err))
                        missing_titles.append(title)
                else:
                    missing_titles.append(title)

            if len(missing_titles) > 0:
                logger.warning("{count} requested Wikipedia articles were missing in cache: {missing}", count=len(missing_titles), missing=missing_titles[:10])

            logger.info("Successfully loaded {count} documents from granular cache", count=len(documents))
            if span_ctx:
                span_ctx.update(output={"document_count": len(documents)})
            return documents

        except Exception as e:
            err_msg = f"❌ Failed to load documents from Wikipedia granular cache: {str(e)}"
            logger.error(err_msg)
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=err_msg)
            raise
    
    def load_from_hotpot_cache(self, numEasy: int, numMedium: int, numHard: int) -> tuple[list[Document], list[dict]]:
        easy_path = Path(self.hotpotqa_cache) / "easy.json"
        medium_path = Path(self.hotpotqa_cache) / "medium.json"
        hard_path = Path(self.hotpotqa_cache) / "hard.json"

        documents = []
        golden_dataset = []

        try:
            easy_data = json.load(open(easy_path, "r", encoding="utf-8"))[:numEasy]
            medium_data = json.load(open(medium_path, "r", encoding="utf-8"))[:numMedium]
            hard_data = json.load(open(hard_path, "r", encoding="utf-8"))[:numHard]
        except FileNotFoundError as err:
            err_msg = f"❌ Failed to load HotpotQA cache: {str(err)}"
            logger.error(err_msg)
            raise

        article_titles = []
        golden_dataset = easy_data + medium_data + hard_data
        logger.info("Total dataset size: {size} (easy: {easy}, medium: {medium}, hard: {hard})", size=len(golden_dataset), easy=len(easy_data), medium=len(medium_data), hard=len(hard_data))
        for item in golden_dataset:
            article_titles.extend(item["titles"])

        unique_article_titles = list(set(article_titles))
        logger.info("Total unique article titles: {size}", size=len(unique_article_titles))
        documents = self.load_from_wikipedia_cache(unique_article_titles)
        return documents, golden_dataset


    def compile_hotpotqa_stage(self, stage: int) -> tuple[list[Document], list[dict]]:
        """
        Compiles the corpus (full raw articles) and golden dataset (questions, answers, gold snippets)
        for a specific evaluation stage.
        """
        stage_config = {
            1: StageConfig.stage_1,
            2: StageConfig.stage_2,
            3: StageConfig.stage_3,
        }

        if stage not in stage_config:
            raise ValueError(f"Invalid Stage number: {stage}. Choose 1, 2, or 3.")

        cfg = stage_config[stage]
        total_questions = cfg.easy + cfg.medium + cfg.hard

        logger.info(
            "Compiling HotpotQA Stage {stage}: Target {questions} QA pairs",
            stage=stage,
            questions=total_questions,
        )

        span_ctx = (
            self.tracer.span(
                name="ingestion.compile_hotpotqa",
                input={"stage": stage, "config": cfg},
            )
            if self.tracer
            else None
        )

        try:
            # 1. Ensure Wikipedia granular cache is populated
            self._create_wikipedia_cache()
            self._create_hotpotqa_cache()
            documents, golden_dataset = self.load_from_hotpot_cache(cfg.easy, cfg.medium, cfg.hard)

            logger.info("Documents count: {count}", count=len(documents))
            logger.info("Golden dataset size: {size}", size=len(golden_dataset))

            # Save the Golden Question Set to Disk
            os.makedirs("data", exist_ok=True)
            gold_file_path = f"data/golden_set_stage_{stage}.json"
            
            with open(gold_file_path, "w", encoding="utf-8") as f:
                json.dump(golden_dataset, f, indent=2, ensure_ascii=False)
            logger.info("Saved {count} golden questions to '{path}'", count=len(golden_dataset), path=gold_file_path)

            if span_ctx:
                span_ctx.update(
                    output={
                        "documents_count": len(documents),
                        "questions_count": len(golden_dataset),
                        "golden_questions_file": gold_file_path,
                    }
                )

            return documents, golden_dataset

        except Exception as e:
            logger.error("HotpotQA stage compilation failed: {err}", err=str(e))
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=str(e))
            raise


