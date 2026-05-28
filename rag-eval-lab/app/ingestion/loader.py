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

from app.tracing.langfuse import LangfuseTracer, get_safe_tracer
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
        self.tracer = get_safe_tracer(tracer)
        
        # Inject hf_token into process environment for authenticated HF Hub requests
        try:
            from app.config.settings import get_settings
            settings = get_settings()
            if settings.hf_token:
                import os
                os.environ["HF_TOKEN"] = settings.hf_token
        except Exception as e:
            logger.warning("Could not automatically export HF_TOKEN to environment: {err}", err=str(e))

    def load_from_wikipedia(self, titles: list[str]) -> list[Document]:
        """
        Loads full-text articles from Wikipedia using optimal batch query API requests.
        Utilizes a local file-based cache to avoid network requests, and fetches
        missing pages in optimal batches of 50 to avoid rate limits entirely.

        Args:
            titles: List of Wikipedia article titles.

        Returns:
            List of LlamaIndex Document objects containing full-text contents.
        """
        logger.info("Loading {count} Wikipedia articles...", count=len(titles))

        span_ctx = (
            self.tracer.span(
                name="ingestion.load.wikipedia_api",
                input={"titles": titles, "count": len(titles)},
            )
            if self.tracer
            else None
        )

        try:
            documents = []
            
            # Ensure local cache directory exists in workspace (ignored by git)
            cache_dir = os.path.join("data", "wikipedia_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            import re
            import requests
            
            # Helper to generate safe cache file name
            def get_cache_path(title: str) -> str:
                safe_title = re.sub(r'[^\w\s-]', '_', title).strip().replace(' ', '_')
                return os.path.join(cache_dir, f"{safe_title}.json")
            
            cached_count = 0
            uncached_titles = []
            
            # 1. First, check local disk cache for all titles
            for title in titles:
                cache_path = get_cache_path(title)
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            cached_data = json.load(f)
                        
                        # Reconstruct the LlamaIndex Document object
                        doc = Document(
                            text=cached_data["text"],
                            metadata=cached_data.get("metadata", {"title": title, "file_name": f"{title}.txt"})
                        )
                        documents.append(doc)
                        cached_count += 1
                    except Exception as cache_err:
                        logger.warning("Failed to read cache for '{title}' (will re-fetch): {err}", title=title, err=str(cache_err))
                        uncached_titles.append(title)
                else:
                    uncached_titles.append(title)
            
            logger.info("Cache search complete: {cached} loaded from cache, {missed} cache misses", cached=cached_count, missed=len(uncached_titles))
            
            # 2. Fetch cache misses in optimal batches of 50 to avoid rate limits
            if len(uncached_titles) > 0:
                batch_size = 50
                for start_idx in range(0, len(uncached_titles), batch_size):
                    batch_chunk = uncached_titles[start_idx:start_idx + batch_size]
                    
                    max_retries = 3
                    retry_delay = 2.0
                    
                    for attempt in range(max_retries):
                        try:
                            # Direct Wikipedia Query API batch request
                            batch_titles_str = "|".join(batch_chunk)
                            url = "https://en.wikipedia.org/w/api.php"
                            params = {
                                "action": "query",
                                "prop": "extracts",
                                "explaintext": 1,     # return plain text, not HTML
                                "titles": batch_titles_str,
                                "format": "json",
                                "redirects": 1,       # resolve redirects
                            }
                            headers = {
                                "User-Agent": "RAGEvalLab/1.0 (contact: admin@pocket-projects.com)"
                            }
                            
                            logger.info(
                                "Fetching live Wikipedia batch ({start}-{end}/{total}) (Attempt {attempt}/{max})...",
                                start=start_idx + 1,
                                end=min(start_idx + batch_size, len(uncached_titles)),
                                total=len(uncached_titles),
                                attempt=attempt + 1,
                                max=max_retries
                            )
                            
                            response = requests.get(url, params=params, headers=headers, timeout=20)
                            response.raise_for_status()
                            
                            data = response.json()
                            pages = data.get("query", {}).get("pages", {})
                            
                            # Process and cache each page in the batch response
                            for page_id, page_info in pages.items():
                                if "missing" in page_info:
                                    continue
                                page_title = page_info.get("title")
                                extract = page_info.get("extract")
                                
                                if page_title and extract:
                                    # Save to cache
                                    cache_path = get_cache_path(page_title)
                                    cached_data = {
                                        "title": page_title,
                                        "text": extract,
                                        "metadata": {
                                            "title": page_title,
                                            "file_name": f"{page_title}.txt"
                                        }
                                    }
                                    with open(cache_path, "w", encoding="utf-8") as f:
                                        json.dump(cached_data, f, ensure_ascii=False, indent=2)
                                    
                                    # Add to documents
                                    doc = Document(
                                        text=extract,
                                        metadata={
                                            "title": page_title,
                                            "file_name": f"{page_title}.txt"
                                        }
                                    )
                                    documents.append(doc)
                            
                            break  # success, break retry loop!
                            
                        except Exception as batch_err:
                            logger.warning(
                                "Wikipedia API batch query failed on attempt {attempt}: {err}",
                                attempt=attempt + 1,
                                err=str(batch_err)
                            )
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)
                                retry_delay *= 2.0
                    
                    # Polite sleep delay between distinct batch requests
                    if start_idx + batch_size < len(uncached_titles):
                        time.sleep(1.5)

            logger.info(
                "Successfully loaded {total} Wikipedia documents ({cached} from cache, {fetched} fetched from batch API)",
                total=len(documents),
                cached=cached_count,
                fetched=len(documents) - cached_count
            )
            
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
            # 1. Fetch HotpotQA Training Split as a stream to avoid 1.4 GB local downloads
            logger.info("Streaming HotpotQA 'distractor' dataset from Hugging Face Hub...")
            try:
                dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True)
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

            logger.info("Filtering and stratifying questions by difficulty and multi-hop type...")
            # We shuffle the streamed dataset using a buffer for reproducible deterministic splits
            shuffled_dataset = dataset.shuffle(seed=42, buffer_size=1000)

            for item in shuffled_dataset:
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
                q_id = item.get("id") or item.get("_id")
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
            if len(target_titles) < cfg["total_docs"]:
                for item in shuffled_dataset:
                    context = item["context"]
                    for title in context["title"]:
                        if title not in target_titles:
                            target_titles.add(title)
                        if len(target_titles) >= cfg["total_docs"]:
                            break
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

            # logger.info("Fetching full Wikipedia articles (streaming via Hugging Face)...")
            # try:
            #     # 'wikipedia' dataset split 'train' has fields: 'id', 'url', 'title', 'text'
            #     wiki_stream = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
            #     titles_to_find = set(titles_list)
                
            #     # Scan stream to gather full texts
            #     for wiki_doc in wiki_stream:
            #         title = wiki_doc["title"]
            #         if title in titles_to_find:
            #             corpus_articles[title] = wiki_doc["text"]
            #             titles_to_find.remove(title)
                        
            #         if len(titles_to_find) == 0:
            #             break
                
            #     logger.info(
            #         "HF stream retrieved {count}/{total} full-text articles",
            #         count=len(corpus_articles),
            #         total=len(titles_list),
            #     )
            # except Exception as stream_err:
            #     logger.warning("HF Wikipedia streaming was unavailable: {err}", err=str(stream_err))

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
