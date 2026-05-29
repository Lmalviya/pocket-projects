"""
main.py
========
Central entrypoint and interactive CLI coordinator for the RAG Eval Lab.

📚 LESSON — End-to-End Bootstrap Orchestration:
This is the master coordinator. When you run `python main.py`, the system:
  1. Loads `.env` credentials and active `experiment.yaml` parameters.
  2. Initializes Langfuse distributed tracing.
  3. **Auto-Bootstraps Indexing**: It pings Qdrant. If the stage-isolated collection 
     (e.g., `rag_eval_lab_stage_1_fixed_splade`) doesn't exist, it compiles the 
     HotpotQA dataset, downloads full-text Wikipedia articles, chunks them using the 
     configured strategy, and indexes them in Qdrant.
  4. **Builds Pipeline**: Wires together the selected retriever (Dense/Sparse/Hybrid),
     reranker (CrossEncoder/Cohere/None), and generator (Single-Turn/Multi-Turn).
  5. **Interactive Console CLI**: Starts a terminal Q&A loop, presenting beautiful
     answers, listing retrieved node sources, and printing the Langfuse trace link!
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from typing import Any
from app.config.settings import get_settings
from app.generation.multi_turn import MultiTurnGenerator
from app.generation.single_turn import SingleTurnGenerator
from app.ingestion.chunking.base import ChunkingConfig
from app.ingestion.chunking.factory import get_chunker
from app.ingestion.indexing.dense import QdrantHybridIndexer
# from app.ingestion.loader import DocumentLoader
from app.ingestion.loader2 import DocumentLoader
from app.ingestion.loader_helper import build_cache
from app.pipeline.multi_turn import MultiTurnPipeline
from app.pipeline.single_turn import SingleTurnPipeline
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker.factory import get_reranker
from app.retrieval.sparse import SparseRetriever
from app.tracing.langfuse import LangfuseTracer
from app.utils.config_loader import load_experiment_config
from app.utils.logger import get_logger

logger = get_logger(__name__)


def print_banner(experiment_name: str, mode: str, retrieval: str, reranker: str) -> None:
    """Prints a premium ASCII terminal banner for the RAG Eval Lab."""
    banner = f"""
    🌌 ========================================================================= 🌌
    🌌                           RAG EVALUATION LAB                              🌌
    🌌 ========================================================================= 🌌
      🧪 ACTIVE EXPERIMENT : {experiment_name.upper()}
      ⚙️  MODE              : {mode.upper()}
      🔍 RETRIEVER         : {retrieval.upper()}
      🎯 RERANKER          : {reranker.upper()}
    🌌 ========================================================================= 🌌
    """
    print(banner)


def bootstrap_database(settings: Any, config: Any, tracer: LangfuseTracer) -> Any:
    """
    Checks Qdrant collections. If the target collection is missing, it automatically
    compiles HotpotQA raw documents, chunks them, and indexes them.
    """
    indexer = QdrantHybridIndexer(
        settings=settings,
        stage=config.ingestion.stage,
        chunking_strategy=config.ingestion.chunking,
        sparse_strategy=config.retrieval.sparse_strategy,
        tracer=tracer,
    )

    # Check if this isolated experiment collection already exists AND has data
    collection_exists = indexer.client.collection_exists(indexer.collection_name)
    collection_empty = False
    if collection_exists:
        info = indexer.client.get_collection(indexer.collection_name)
        collection_empty = info.points_count == 0

    if not collection_exists or collection_empty:
        if collection_empty:
            logger.warning(
                "Collection '{col}' exists but is EMPTY (previous index run may have failed). Rebuilding...",
                col=indexer.collection_name,
            )
        else:
            logger.warning(
                "Collection '{col}' not found in Qdrant! Automatically bootstrapping index...",
                col=indexer.collection_name,
            )
        
        # 1. Fetch raw Wikipedia docs aligned with selected Stage difficulty questions
        # build_cache()
        loader = DocumentLoader(tracer=tracer)
        documents, _ = loader.load_stage(config.ingestion.stage)
        
        # 2. Chunk documents using configured strategy
        logger.info("Initializing chunker strategy '{strategy}'...", strategy=config.ingestion.chunking)
        chunk_config = ChunkingConfig(
            chunk_size=config.ingestion.chunk_size,
            chunk_overlap=config.ingestion.chunk_overlap,
        )
        chunker = get_chunker(
            strategy=config.ingestion.chunking,
            config=chunk_config,
            tracer=tracer,
        )
        
        chunker_result = chunker.chunk(documents)
        nodes = chunker_result.nodes
        
        # 3. Embed and index nodes in Qdrant
        indexer.build_index(nodes, recreate=True)
        logger.info("Database bootstrap completed successfully.")
    else:
        logger.info("Collection '{col}' exists. Index loaded directly.", col=indexer.collection_name)

    return indexer.load_index()


def main() -> None:
    """Main execution loop for RAG Eval Lab."""
    # 1. Load settings (holds .env keys) and experiment parameters
    try:
        settings = get_settings()
        config = load_experiment_config(settings.experiment_config_path)
        logger.info("Config: {config}", config=config)
    except Exception as init_err:
        print(f"\n❌ [Startup Error] Failed to load configurations:\n{str(init_err)}\n")
        sys.exit(1)

    # 2. Initialize Langfuse distributed tracer
    tracer = LangfuseTracer(
        experiment_name=config.experiment_name,
        enabled=config.tracing.enabled,
    )

    # 3. Bootstrap/Load Qdrant Hybrid Index
    try:
        index = bootstrap_database(settings=settings, config=config, tracer=tracer)
    except Exception as db_err:
        print(f"\n{str(db_err)}\n")
        sys.exit(1)

    # 4. Instantiate Retrieval Layer components
    ret_method = config.retrieval.method.lower()
    top_k = config.retrieval.top_k
    
    if ret_method == "dense":
        retriever = DenseRetriever(index=index, top_k=top_k, tracer=tracer)
    elif ret_method == "sparse":
        retriever = SparseRetriever(index=index, top_k=top_k, tracer=tracer)
    elif ret_method == "hybrid":
        # RRF k hyper-parameter defaults to 60
        retriever = HybridRetriever(index=index, top_k=top_k, rrf_k=60, tracer=tracer)
    else:
        print(f"\n❌ Invalid retrieval method '{ret_method}'. Choose: dense | sparse | hybrid\n")
        sys.exit(1)

    # 5. Instantiate Stage-2 Reranker
    try:
        reranker = get_reranker(
            strategy=config.retrieval.reranker,
            top_k=config.retrieval.reranker_top_k,
            tracer=tracer,
        )
    except Exception as r_err:
        print(f"\n{str(r_err)}\n")
        sys.exit(1)

    # 6. Instantiate Generator & Pipeline
    mode = config.generation.mode.lower()
    
    if mode == "single_turn":
        generator = SingleTurnGenerator(settings=settings, tracer=tracer)
        pipeline = SingleTurnPipeline(
            settings=settings,
            retriever=retriever,
            generator=generator,
            reranker=reranker,
            tracer=tracer,
        )
    elif mode == "multi_turn":
        generator = MultiTurnGenerator(settings=settings, tracer=tracer)
        pipeline = MultiTurnPipeline(
            settings=settings,
            retriever=retriever,
            generator=generator,
            reranker=reranker,
            tracer=tracer,
        )
    else:
        print(f"\n❌ Invalid generation mode '{mode}'. Choose: single_turn | multi_turn\n")
        sys.exit(1)

    # 7. Start Interactive Console Loop
    print_banner(
        experiment_name=config.experiment_name,
        mode=mode,
        retrieval=ret_method,
        reranker=config.retrieval.reranker,
    )

    if mode == "multi_turn":
        print("🤖 [Multi-Turn Mode]: Conversational memory enabled.")
        print("💡 Commands: Type '/clear' to reset chat memory, or '/exit' to quit.\n")
    else:
        print("🤖 [Single-Turn Mode]: Independent Q&A sessions.")
        print("💡 Commands: Type '/exit' to quit.\n")

    try:
        while True:
            # Format terminal prompt
            prompt_str = "\nYou: " if mode == "multi_turn" else "\nQ: "
            try:
                query = input(prompt_str).strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting RAG Eval Lab. Goodbye!")
                break

            if not query:
                continue

            if query.lower() == "/exit":
                print("Exiting RAG Eval Lab. Goodbye!")
                break

            if mode == "multi_turn" and query.lower() == "/clear":
                pipeline.clear_history()
                print("🧹 Conversation memory successfully cleared!")
                continue

            # Execute end-to-end pipeline
            print("⏳ Retrieving facts and generating answer...")
            try:
                res = pipeline.run(query)
                
                # Print synthesized response
                print("\n🤖 Answer:")
                print(res.answer)
                
                # Print execution metrics
                print("\n📊 Metrics:")
                print(f"  ⚡ Latency      : {res.latency_ms} ms")
                if res.trace_url:
                    print(f"  🔗 Langfuse Trace: {res.trace_url}")

                # Print retrieved document sources
                print(f"\n📖 Sources (Top {len(res.retrieved_nodes)} chunks):")
                for i, node in enumerate(res.retrieved_nodes, start=1):
                    title = node.node.metadata.get("title", "Unknown")
                    is_gold = node.node.metadata.get("is_gold", False)
                    badge = "[GOLD 🌟]" if is_gold else "[DISTRACTOR 🛡️]"
                    print(f"  {i}. {badge} Page: '{title}' (Relevance: {node.score})")
                    # Preview the text snippet
                    snippet = node.node.get_content()[:120].replace('\n', ' ')
                    print(f"     Preview: \"{snippet}...\"")

            except Exception as e:
                print(f"\n❌ [Pipeline Error] Failed to process query:\n{str(e)}\n")

    finally:
        # Flush Langfuse buffer before process exits to ensure telemetry delivery
        tracer.flush()


if __name__ == "__main__":
    main()
