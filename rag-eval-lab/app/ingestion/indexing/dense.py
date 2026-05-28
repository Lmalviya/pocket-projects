"""
app/ingestion/indexing/dense.py
================================
Dense and hybrid vector indexing using Qdrant and Ollama embeddings.

📚 LESSON — Production-Grade Native Hybrid Search:
Instead of running separate text search engines (BM25) and vector databases (Dense),
modern production systems utilize unified vector search engines like Qdrant to perform
both. This eliminates network overhead and handles ranking fusion directly on the server.

This class implements `QdrantHybridIndexer`:
  - It configures Qdrant collections with both dense (384-dim Ollama) and sparse configs.
  - It handles collection creation, dropping, and loading.
  - It integrates directly with LlamaIndex's `QdrantVectorStore` via `sparse_doc_fn`
    and `sparse_query_fn` callback hooks.
  - It caches sparse vectors in-memory during indexing, enabling fast batch-processing.
"""

from typing import Any

import qdrant_client
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import SparseVector
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app.config.settings import Settings
from app.ingestion.indexing.sparse import SparseEncoder
from app.tracing.langfuse import LangfuseTracer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantHybridIndexer:
    """
    Manages dense and hybrid indexing inside Qdrant.
    Integrates with Ollama embeddings and pluggable sparse representations.
    """

    def __init__(
        self,
        settings: Settings,
        stage: int = 1,
        chunking_strategy: str = "fixed",
        sparse_strategy: str = "splade",
        tracer: LangfuseTracer | None = None,
    ) -> None:
        """
        Initialize the Qdrant hybrid indexer.

        Args:
            settings: Central Settings instance.
            stage: Evaluation dataset stage (1, 2, or 3).
            chunking_strategy: Strategy name ("fixed", "recursive", "sentence", "semantic").
            sparse_strategy: Sparse vector strategy ("splade" | "bm25").
            tracer: Optional LangfuseTracer instance.
        """
        self.settings = settings
        self.stage = stage
        self.chunking_strategy = chunking_strategy.lower()
        self.sparse_strategy = sparse_strategy.lower()
        self.tracer = tracer

        # 📚 LESSON — Dynamically Isolated Collections:
        # To avoid blending different experiment variables, we construct isolated collections.
        self.collection_name = (
            f"{settings.qdrant_collection_name}_stage_{stage}_"
            f"{self.chunking_strategy}_{self.sparse_strategy}"
        )

        logger.info("Initializing QdrantHybridIndexer for collection: {name}", name=self.collection_name)

        # 1. Connect to Qdrant Docker Service
        try:
            if settings.qdrant_url == ":memory:":
                logger.info("Initializing in-memory Qdrant client for local testing/evaluation...")
                self.client = qdrant_client.QdrantClient(location=":memory:")
            else:
                self.client = qdrant_client.QdrantClient(url=settings.qdrant_url)
            # Ping database to verify connection
            self.client.get_collections()
        except Exception as conn_err:
            err_msg = (
                f"❌ [Connection Error] Failed to connect to Qdrant service at '{settings.qdrant_url}'.\n"
                f"Reason: {str(conn_err)}\n"
                f"Troubleshooting:\n"
                f"  1. Ensure the Qdrant Docker container is running.\n"
                f"  2. If you haven't started Qdrant, run this Docker command:\n"
                f"     docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant\n"
                f"  3. Check if local port 6333 is blocked."
            )
            logger.error(err_msg)
            raise ConnectionError(err_msg) from conn_err

        # 2. Setup Ollama Embedding Model (Dense)
        # BAAI/bge-small-en-v1.5 produces 384-dimensional dense vectors
        logger.info("Connecting to local Ollama service at {url}...", url=settings.ollama_base_url)
        self.embed_model = OllamaEmbedding(
            model_name=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )

        # 3. Setup Sparse Vector Encoder (BM25 or SPLADE)
        self.sparse_encoder = SparseEncoder(strategy=self.sparse_strategy)

        # 4. In-Memory Cache for Sparse Document Vectors
        # We pre-compute all sparse vectors during indexing to support batching.
        self._sparse_vectors_cache: dict[str, SparseVector] = {}

    def _get_sparse_doc_vector(self, text: str) -> SparseVector:
        """
        Callback hook called by LlamaIndex's QdrantVectorStore to retrieve
        the sparse vector for a document chunk.
        """
        # Retrieve from cache if pre-computed (highly recommended for performance)
        if text in self._sparse_vectors_cache:
            return self._sparse_vectors_cache[text]

        # Fallback: encode dynamically on-the-fly (e.g. during incremental additions)
        res = self.sparse_encoder.encode([text])[0]
        return SparseVector(
            indices=res["indices"],
            values=res["values"]
        )

    def _get_sparse_query_vector(self, query: str) -> SparseVector:
        """
        Callback hook called by LlamaIndex's QdrantVectorStore to retrieve
        the sparse vector for a search query.
        """
        res = self.sparse_encoder.encode([query], is_query=True)[0]
        return SparseVector(
            indices=res["indices"],
            values=res["values"]
        )

    def build_index(self, nodes: list[TextNode], recreate: bool = True) -> VectorStoreIndex:
        """
        Builds the Qdrant native hybrid index from document chunks (nodes).

        Args:
            nodes: List of split LlamaIndex TextNodes.
            recreate: If True, deletes and recreates the collection.

        Returns:
            LlamaIndex VectorStoreIndex instance connected to Qdrant.
        """
        logger.info(
            "Building hybrid index for {count} nodes in collection '{col}'...",
            count=len(nodes),
            col=self.collection_name,
        )

        span_ctx = (
            self.tracer.span(
                name="indexing.hybrid.build",
                input={
                    "collection": self.collection_name,
                    "node_count": len(nodes),
                    "sparse_strategy": self.sparse_strategy,
                },
            )
            if self.tracer
            else None
        )

        try:
            # 1. Handle Collection Lifecycle (Recreation/Setup)
            if recreate and self.client.collection_exists(self.collection_name):
                logger.warning("Recreating collection '{col}'...", col=self.collection_name)
                self.client.delete_collection(self.collection_name)

            if not self.client.collection_exists(self.collection_name):
                logger.info("Creating new Qdrant collection '{col}' with hybrid configs...", col=self.collection_name)
                
                # Configure dual vectors (Dense HNSW Hashing + Sparse Inverted Indexing)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.settings.embed_dim,  # 384
                        distance=qmodels.Distance.COSINE,
                    ),
                    sparse_vectors_config={
                        "sparse": qmodels.SparseVectorParams(
                            index=qmodels.SparseIndexParams(
                                on_disk=True  # Enables raw on-disk memory footprint limits
                            )
                        )
                    },
                )

            # 2. Batch-Compute Sparse Vectors for all nodes and cache them
            logger.info("Batch-encoding sparse vectors for {count} nodes...", count=len(nodes))
            node_texts = [node.get_content(metadata_mode="embed") for node in nodes]
            
            sparse_embs = self.sparse_encoder.encode(node_texts)
            
            self._sparse_vectors_cache.clear()
            for node_text, emb in zip(node_texts, sparse_embs):
                self._sparse_vectors_cache[node_text] = SparseVector(
                    indices=emb["indices"],
                    values=emb["values"]
                )

            # 3. Instantiate LlamaIndex Vector Store
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                enable_hybrid=True,
                sparse_doc_fn=self._get_sparse_doc_vector,
                sparse_query_fn=self._get_sparse_query_vector,
                sparse_vector_name="sparse",
            )

            # 4. Construct the Vector Store Index
            # This calls Ollama Embedding Model under the hood for dense embeddings,
            # and triggers our sparse callbacks for sparse vectors.
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
            )

            logger.info("Hybrid index successfully built in Qdrant")

            if span_ctx:
                span_ctx.update(output={"collection": self.collection_name, "status": "SUCCESS"})

            return index

        except Exception as e:
            logger.error("Failed to build hybrid index: {err}", err=str(e))
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=str(e))
            raise

    def load_index(self) -> VectorStoreIndex:
        """
        Loads the hybrid VectorStoreIndex from the existing Qdrant collection.
        """
        logger.info("Loading existing hybrid index for collection '{col}'...", col=self.collection_name)
        
        span_ctx = (
            self.tracer.span(
                name="indexing.hybrid.load",
                input={"collection": self.collection_name},
            )
            if self.tracer
            else None
        )

        try:
            if not self.client.collection_exists(self.collection_name):
                raise ValueError(f"Collection '{self.collection_name}' does not exist in Qdrant.")

            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                enable_hybrid=True,
                sparse_doc_fn=self._get_sparse_doc_vector,
                sparse_query_fn=self._get_sparse_query_vector,
                sparse_vector_name="sparse",
            )

            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embed_model,
            )

            logger.info("Hybrid index successfully loaded from Qdrant")

            if span_ctx:
                span_ctx.update(output={"collection": self.collection_name, "status": "SUCCESS"})

            return index

        except Exception as e:
            logger.error("Failed to load hybrid index: {err}", err=str(e))
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=str(e))
            raise
