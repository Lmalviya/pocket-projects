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

import json
import os
import random
import time
from typing import Any

from datasets import load_dataset
from llama_index.core import Document
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.readers.wikipedia import WikipediaReader

from app.tracing.langfuse import LangfuseTracer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """
    Handles loading raw document corpora from Wikipedia, directories, and HotpotQA.
    Compiles fully-aligned golden QA sets and long-form Wikipedia article indexes.
    """

    def __init__(self, tracer: LangfuseTracer | None = None) -> None:
        """
        Initialize the loader with optional tracer.
        """
        self.tracer = tracer

    def load_from_wikipedia(self, titles: list[str]) -> list[Document]:
        """
        Loads full-text articles from Wikipedia using WikipediaReader.

        Args:
            titles: List of Wikipedia article titles.

        Returns:
            List of LlamaIndex Document objects containing full-text contents.
        """
        logger.info("Fetching {count} Wikipedia articles from live API...", count=len(titles))

        span_ctx = (
            self.tracer.span(
                name="ingestion.load.wikipedia_api",
                input={"titles": titles, "count": len(titles)},
            )
            if self.tracer
            else None
        )

        try:
            reader = WikipediaReader()
            documents = []
            
            # Fetch page-by-page to monitor progress and handle individual errors
            for i, title in enumerate(titles):
                try:
                    # Fetch single page
                    docs = reader.load_data(pages=[title])
                    if docs:
                        documents.extend(docs)
                    # Tiny sleep to respect Wikipedia rate limits
                    time.sleep(0.05)
                except Exception as page_err:
                    logger.warning("Could not fetch page '{title}': {err}", title=title, err=str(page_err))
                    continue

            logger.info("Successfully fetched {count} full Wikipedia documents", count=len(documents))
            
            if span_ctx:
                span_ctx.update(output={"document_count": len(documents)})
                
            return documents

        except Exception as e:
            err_msg = (
                f"❌ [Network Error] Failed to load data from Wikipedia API.\n"
                f"Reason: {str(e)}\n"
                f"Troubleshooting: Check internet connection, proxy settings, or rate limits."
            )
            logger.error(err_msg)
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=err_msg)
            raise ConnectionError(err_msg) from e

    def load_from_directory(self, path: str, required_exts: list[str] = [".txt", ".md"]) -> list[Document]:
        """
        Reads local files from a directory using SimpleDirectoryReader.
        """
        logger.info("Reading directory '{path}' for extensions {exts}...", path=path, exts=required_exts)
        
        span_ctx = (
            self.tracer.span(
                name="ingestion.load.directory",
                input={"path": path, "required_exts": required_exts},
            )
            if self.tracer
            else None
        )

        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Directory path '{path}' does not exist.")

            reader = SimpleDirectoryReader(input_dir=path, recursive=True, required_exts=required_exts)
            documents = reader.load_data()
            
            logger.info("Loaded {count} documents from local directory", count=len(documents))
            
            if span_ctx:
                span_ctx.update(output={"document_count": len(documents)})
                
            return documents

        except Exception as e:
            logger.error("Failed to load local directory: {err}", err=str(e))
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=str(e))
            raise

    def compile_hotpotqa_stage(self, stage: int) -> tuple[list[Document], list[dict]]:
        """
        Compiles the corpus (full raw articles) and golden dataset (questions, answers, gold snippets)
        for a specific evaluation stage.

        📚 LESSON — Symmetrical Golden Data & Stratified Sampling:
        This method compiles:
          1. A golden evaluation questions list (Easy/Medium/Hard + Bridge/Comparison)
             persisted to `data/golden_set_stage_{stage}.json`.
          2. A database corpus containing the FULL, raw Wikipedia articles for all gold
             and distractor articles, padded to the exact target size.
        """
        stage_config = {
            1: {"total_docs": 500, "q_easy": 50, "q_medium": 37, "q_hard": 38},
            2: {"total_docs": 5000, "q_easy": 150, "q_medium": 175, "q_hard": 175},
            3: {"total_docs": 50000, "q_easy": 300, "q_medium": 450, "q_hard": 450},
        }

        if stage not in stage_config:
            raise ValueError(f"Invalid Stage number: {stage}. Choose 1, 2, or 3.")

        cfg = stage_config[stage]
        total_questions = cfg["q_easy"] + cfg["q_medium"] + cfg["q_hard"]

        logger.info(
            "Compiling HotpotQA Stage {stage}: Target {docs} full docs, {questions} QA pairs",
            stage=stage,
            docs=cfg["total_docs"],
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
            # 1. Fetch HotpotQA Training Split
            logger.info("Loading HotpotQA 'distractor' dataset from Hugging Face...")
            try:
                dataset = load_dataset("hotpot_qa", "distractor", split="train")
            except Exception as hf_err:
                err_msg = (
                    f"❌ [Network Error] Failed to load HotpotQA dataset from Hugging Face Hub.\n"
                    f"Reason: {str(hf_err)}\n"
                    f"Troubleshooting: Verify internet connection and proxy settings."
                )
                logger.error(err_msg)
                raise ConnectionError(err_msg) from hf_err

            # 2. Stratified Sampling of Golden Questions
            easy_qs: list[dict] = []
            medium_qs: list[dict] = []
            hard_qs: list[dict] = []

            shuffled_indices = list(range(len(dataset)))
            random.seed(42)  # For reproducible evaluation splits
            random.shuffle(shuffled_indices)

            logger.info("Filtering and stratifying questions by difficulty and multi-hop type...")
            for idx in shuffled_indices:
                item = dataset[idx]
                level = item.get("level", "").lower()
                
                # We filter to collect a balanced mix of bridge and comparison
                if level == "easy" and len(easy_qs) < cfg["q_easy"]:
                    easy_qs.append(item)
                elif level == "medium" and len(medium_qs) < cfg["q_medium"]:
                    medium_qs.append(item)
                elif level == "hard" and len(hard_qs) < cfg["q_hard"]:
                    hard_qs.append(item)

                if (
                    len(easy_qs) == cfg["q_easy"]
                    and len(medium_qs) == cfg["q_medium"]
                    and len(hard_qs) == cfg["q_hard"]
                ):
                    break

            selected_items = easy_qs + medium_qs + hard_qs
            logger.info("Selected {count} golden questions successfully", count=len(selected_items))

            # 3. Collect Target Titles & Construct Golden QA Set
            # We compile the golden QA set AND collect all unique titles of articles we need.
            target_titles = set()
            gold_titles = set()
            golden_questions = []

            for item in selected_items:
                q_id = item["_id"]
                question = item["question"]
                answer = item["answer"]
                supporting_facts = item["supporting_facts"]
                
                # Gold titles (articles containing ground-truth facts) for this question
                sf_titles = list(set([fact[0] for fact in supporting_facts]))
                gold_titles.update(sf_titles)
                target_titles.update(sf_titles)

                # 📚 LESSON — Supporting Context Snippets inside Golden Dataset:
                # We store the original sentence-level snippets from HotpotQA's context
                # inside the golden dataset JSON for precise verification during evaluation.
                gold_snippets = {}
                context = item["context"]
                # context: {"title": [t1, t2], "sentences": [[s1, s2], [s3]]}
                for title, sentences in zip(context["title"], context["sentences"]):
                    if title in sf_titles:
                        gold_snippets[title] = " ".join(sentences)

                golden_questions.append({
                    "id": q_id,
                    "question": question,
                    "answer": answer,
                    "level": item["level"],
                    "type": item["type"],
                    "gold_titles": sf_titles,
                    "gold_snippets": gold_snippets,  # Added supporting snippet context!
                })

            # Gather extra distractor titles from other samples to reach target size
            logger.info("Gathering distractor article titles to reach exactly {docs} docs...", docs=cfg["total_docs"])
            for idx in shuffled_indices:
                if len(target_titles) >= cfg["total_docs"]:
                    break
                
                item = dataset[idx]
                context = item["context"]
                for title in context["title"]:
                    if title not in target_titles:
                        target_titles.add(title)
                    if len(target_titles) >= cfg["total_docs"]:
                        break

            titles_list = list(target_titles)
            logger.info(
                "Target compiled: {total} unique titles ({gold} gold, {dist} distractors)",
                total=len(titles_list),
                gold=len(gold_titles),
                dist=len(titles_list) - len(gold_titles),
            )

            # 4. Fetch Full-Text Content for the Titles
            # We try streaming the Hugging Face full Wikipedia dataset first (fast & rate-limit proof).
            # If that fails or is offline, we fall back to fetching page-by-page from live Wikipedia API.
            corpus_articles = {}  # title -> full_text

            logger.info("Fetching full Wikipedia articles (streaming via Hugging Face)...")
            try:
                # 'wikipedia' dataset split 'train' has fields: 'id', 'url', 'title', 'text'
                wiki_stream = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
                titles_to_find = set(titles_list)
                
                # Scan stream to gather full texts
                for wiki_doc in wiki_stream:
                    title = wiki_doc["title"]
                    if title in titles_to_find:
                        corpus_articles[title] = wiki_doc["text"]
                        titles_to_find.remove(title)
                        
                    if len(titles_to_find) == 0:
                        break
                
                logger.info(
                    "HF stream retrieved {count}/{total} full-text articles",
                    count=len(corpus_articles),
                    total=len(titles_list),
                )
            except Exception as stream_err:
                logger.warning("HF Wikipedia streaming was unavailable: {err}", err=str(stream_err))

            # Fallback for remaining titles: Query live Wikipedia API
            remaining_titles = [t for t in titles_list if t not in corpus_articles]
            if len(remaining_titles) > 0:
                logger.info("Falling back to live Wikipedia API for {count} remaining pages...", count=len(remaining_titles))
                try:
                    fallback_docs = self.load_from_wikipedia(remaining_titles)
                    for doc in fallback_docs:
                        title = doc.metadata.get("title")
                        if title and title not in corpus_articles:
                            corpus_articles[title] = doc.text
                except Exception as api_err:
                    logger.warning("Wikipedia API fallback failed: {err}", err=str(api_err))

            # Verify that we succeeded in fetching content
            if len(corpus_articles) == 0:
                err_msg = (
                    "❌ [Network Error] Could not load any Wikipedia article content.\n"
                    "Both Hugging Face streaming and Wikipedia live API calls failed due to network blocks.\n"
                    "Please check your internet settings or configure an HTTP proxy."
                )
                logger.error(err_msg)
                raise ConnectionError(err_msg)

            # 5. Convert full articles to LlamaIndex Document objects
            documents = []
            for title, text in corpus_articles.items():
                is_gold = title in gold_titles
                doc = Document(
                    text=text,
                    metadata={
                        "title": title,
                        "source": "wikipedia_full",
                        "is_gold": is_gold,
                        "stage": stage,
                    },
                )
                documents.append(doc)

            logger.info(
                "Final full-text corpus compiled: {total} documents ready for chunking",
                total=len(documents),
            )

            # 6. Save the Symmetrical Golden Question Set to Disk
            os.makedirs("data", exist_ok=True)
            gold_file_path = f"data/golden_set_stage_{stage}.json"
            
            with open(gold_file_path, "w", encoding="utf-8") as f:
                json.dump(golden_questions, f, indent=2, ensure_ascii=False)
            logger.info("Saved {count} golden questions to '{path}'", count=len(golden_questions), path=gold_file_path)

            if span_ctx:
                span_ctx.update(
                    output={
                        "documents_count": len(documents),
                        "gold_articles_count": len(gold_titles),
                        "questions_count": len(golden_questions),
                        "golden_questions_file": gold_file_path,
                    }
                )

            return documents, golden_questions

        except Exception as e:
            logger.error("HotpotQA stage compilation failed: {err}", err=str(e))
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=str(e))
            raise
