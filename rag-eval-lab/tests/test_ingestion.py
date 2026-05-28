"""
tests/test_ingestion.py
========================
Unit and smoke tests for the Ingestion and Indexing layer components.

Tests:
  - DocumentLoader local directory reads.
  - SparseEncoder output formatting and validation for BM25 and SPLADE.
  - QdrantHybridIndexer collection configurations, callback hooks, and index builders using in-memory Qdrant.
"""

import os
import shutil
import tempfile
import unittest.mock as mock
import pytest
from llama_index.core.schema import TextNode

from app.config.settings import Settings
from app.ingestion.indexing.dense import QdrantHybridIndexer
from app.ingestion.indexing.sparse import SparseEncoder
from app.ingestion.loader import DocumentLoader


@pytest.fixture(autouse=True)
def mock_huggingface_assets():
    """
    📚 LESSON — Mocking where imports are USED:
    In Python, the import statement binds a name in the local module scope. 
    If a module is imported before a mock is applied, it will continue to point
    to the original unmocked object. Therefore, the golden rule of mocking is:
    "Mock the object where it is USED, not where it is defined."
    
    We patch AutoTokenizer and STSparseEncoder inside `app.ingestion.indexing.sparse`.
    """
    with mock.patch("app.ingestion.indexing.sparse.AutoTokenizer.from_pretrained") as mock_tokenizer_loader, \
         mock.patch("app.ingestion.indexing.sparse.STSparseEncoder") as mock_splade_loader:
        
        # 1. Setup Mock Tokenizer
        mock_tokenizer = mock.Mock()
        mock_tokenizer.encode.return_value = [101, 2047, 3000, 102]
        mock_tokenizer_loader.return_value = mock_tokenizer

        # 2. Setup Mock SPLADE Neural Encoder
        mock_splade = mock.Mock()
        mock_splade.encode.return_value = [
            {"indices": [101, 2047, 3000], "values": [1.45, 2.57, 0.89]}
        ]
        mock_splade_loader.return_value = mock_splade

        yield


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file loading tests."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


@pytest.fixture
def test_settings():
    """Returns a test settings instance pointing Qdrant to in-memory mode."""
    return Settings(
        nvidia_api_key="mock-key",
        langfuse_public_key="mock-key",
        langfuse_secret_key="mock-key",
        qdrant_url=":memory:",  # In-memory Qdrant avoids Docker dependency in tests
        ollama_base_url="http://localhost:11434",
    )


def test_document_loader_directory(temp_dir):
    """Test loading documents from a local directory."""
    file1 = os.path.join(temp_dir, "test1.txt")
    file2 = os.path.join(temp_dir, "test2.md")
    
    with open(file1, "w", encoding="utf-8") as f:
        f.write("Hello world! This is a simple text file.")
        
    with open(file2, "w", encoding="utf-8") as f:
        f.write("# Hello Markdown\nThis is a simple markdown file.")

    loader = DocumentLoader()
    docs = loader.load_from_directory(temp_dir, required_exts=[".txt", ".md"])
    
    assert len(docs) == 2
    titles = [doc.metadata.get("file_name") for doc in docs]
    assert "test1.txt" in titles or "test2.md" in titles


def test_sparse_encoder_bm25():
    """Test the lexical BM25 encoder."""
    encoder = SparseEncoder(strategy="bm25")
    texts = [
        "Artificial intelligence is transforming software engineering.",
        "RAG applications combine dense retrieval and vector databases."
    ]
    
    # Ingest document corpus
    embeddings = encoder.encode(texts)
    
    assert len(embeddings) == 2
    for emb in embeddings:
        assert "indices" in emb
        assert "values" in emb
        assert len(emb["indices"]) == len(emb["values"])
        assert emb["indices"] == sorted(emb["indices"])


def test_sparse_encoder_splade():
    """Test the neural SPLADE encoder (using offline mock)."""
    encoder = SparseEncoder(strategy="splade")
    texts = ["Linux was created by Linus Torvalds."]
    
    embeddings = encoder.encode(texts)
    
    assert len(embeddings) == 1
    emb = embeddings[0]
    assert "indices" in emb
    assert "values" in emb
    assert len(emb["indices"]) > 0
    assert len(emb["indices"]) == len(emb["values"])
    assert emb["indices"] == sorted(emb["indices"])


def test_qdrant_hybrid_indexer(test_settings):
    """Test indexing TextNodes using in-memory Qdrant and pluggable encoders."""
    import uuid
    nodes = [
        TextNode(text="Qdrant is a high performance vector search engine.", id_=str(uuid.uuid4())),
        TextNode(text="SPLADE performs term expansion over a vocabulary.", id_=str(uuid.uuid4()))
    ]

    indexer = QdrantHybridIndexer(
        settings=test_settings,
        stage=1,
        chunking_strategy="fixed",
        sparse_strategy="splade"
    )

    # 📚 LESSON — Mocking Pydantic Model Methods:
    # Since OllamaEmbedding inherits from Pydantic's BaseModel, it enforces strict schema
    # validation and prevents dynamic attribute assignments. We patch the methods on the
    # class level (app.ingestion.indexing.dense.OllamaEmbedding) to intercept embedding calls safely.
    with mock.patch("app.ingestion.indexing.dense.OllamaEmbedding.get_text_embedding", return_value=[0.1] * 384), \
         mock.patch("app.ingestion.indexing.dense.OllamaEmbedding.get_query_embedding", return_value=[0.1] * 384), \
         mock.patch("app.ingestion.indexing.dense.OllamaEmbedding.get_text_embedding_batch", side_effect=lambda texts, **kwargs: [[0.1] * 384 for _ in texts]):

        # Build the hybrid index in-memory
        index = indexer.build_index(nodes, recreate=True)
        assert index is not None
        assert indexer.client.collection_exists(indexer.collection_name)

        # Load the index back
        loaded_index = indexer.load_index()
        assert loaded_index is not None
