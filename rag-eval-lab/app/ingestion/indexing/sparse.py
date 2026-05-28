"""
app/ingestion/indexing/sparse.py
==================================
Sparse embedding encoders for lexical (BM25) and neural expansion (SPLADE) search.

📚 LESSON — Learned Sparse Representations & Unified Vocabularies:
A sparse vector represents a document as a set of vocabulary token importances.
To evaluate BM25 (exact keyword match) and SPLADE (learned concept expansion) fairly,
we use a unified vocabulary:
  - We use the BERT WordPiece tokenizer vocabulary (30,522 tokens) from the SPLADE model.
  - For BM25: We calculate term frequency (TF) and inverse document frequency (IDF) 
    using the exact BM25 formula, mapping words to their WordPiece token IDs.
  - For SPLADE: We run the neural SPLADE model to predict contextual token weights and 
    perform term expansion (e.g. mapping "Linux" to "Linus Torvalds" even if Torvalds is not in text).

Both methods return a sparse representation compatible with Qdrant:
  `{"indices": list[int], "values": list[float]}`
"""

import math
from typing import Any

from sentence_transformers import SparseEncoder as STSparseEncoder
from transformers import AutoTokenizer

from app.utils.logger import get_logger

logger = get_logger(__name__)


class SparseEncoder:
    """
    Encodes text corpora and search queries into sparse vectors.
    Supports BM25 (lexical) and SPLADE (neural learned expansion) strategies.
    """

    def __init__(self, strategy: str = "splade") -> None:
        """
        Initialize the sparse encoder.

        Args:
            strategy: Sparse representation strategy: "splade" | "bm25".
        """
        self.strategy = strategy.lower()
        self.model_id = "naver/splade-cocondenser-ensembledistil"

        if self.strategy not in ["splade", "bm25"]:
            raise ValueError(f"Invalid sparse strategy '{self.strategy}'. Choose 'splade' or 'bm25'.")

        logger.info("Initializing SparseEncoder using strategy: {strategy}", strategy=self.strategy)

        # 1. Load the shared tokenizer (WordPiece vocabulary)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        except Exception as e:
            err_msg = (
                f"❌ Failed to load tokenizer for model '{self.model_id}'.\n"
                f"Reason: {str(e)}\n"
                f"Please verify you have an active internet connection to download HF assets."
            )
            logger.error(err_msg)
            raise ConnectionError(err_msg) from e

        # 2. Load the SPLADE neural network model if active
        self.model = None
        if self.strategy == "splade":
            try:
                self.model = STSparseEncoder(self.model_id)
                logger.info("SPLADE neural encoder loaded successfully.")
            except Exception as e:
                err_msg = (
                    f"❌ Failed to load SPLADE model '{self.model_id}' via sentence-transformers.\n"
                    f"Reason: {str(e)}"
                )
                logger.error(err_msg)
                raise ConnectionError(err_msg) from e

    def encode(self, texts: list[str], is_query: bool = False) -> list[dict[str, Any]]:
        """
        Encodes a list of texts into Qdrant-compatible sparse vectors.

        Args:
            texts: List of strings to encode.
            is_query: If True, indicates these are search queries. 
                     (For BM25, query IDF uses the corpus stats; for SPLADE, query weights are encoded directly).

        Returns:
            List of dictionaries, each containing:
              - "indices": List of token integer IDs.
              - "values": List of floats representing weights.
        """
        if self.strategy == "splade":
            return self._encode_splade(texts)
        else:
            return self._encode_bm25(texts, is_query)

    def _encode_splade(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Runs the pretrained SPLADE model to compute sparse vectors.
        """
        # sentence-transformers SparseEncoder.encode returns a dictionary:
        # e.g., [{'indices': array([101, 102]), 'values': array([1.45, 0.89])}, ...]
        # We ensure it converts to standard Python types.
        try:
            raw_embeddings = self.model.encode(texts)
            
            # If a single string was passed, raw_embeddings might be a single dict
            if isinstance(raw_embeddings, dict):
                raw_embeddings = [raw_embeddings]
                
            formatted = []
            for emb in raw_embeddings:
                # Convert numpy arrays to Python lists
                indices = [int(i) for i in emb["indices"]]
                values = [float(v) for v in emb["values"]]
                
                # Qdrant expects indices to be sorted
                sorted_pairs = sorted(zip(indices, values))
                if sorted_pairs:
                    indices, values = zip(*sorted_pairs)
                    indices = list(indices)
                    values = list(values)
                else:
                    indices, values = [], []
                    
                formatted.append({
                    "indices": indices,
                    "values": values
                })
            return formatted
            
        except Exception as e:
            logger.error("SPLADE encoding failed: {err}", err=str(e))
            raise

    def _encode_bm25(self, texts: list[str], is_query: bool = False) -> list[dict[str, Any]]:
        """
        Computes exact keyword BM25 weights mapped to the tokenizer's vocabulary.
        """
        # 📚 LESSON — BM25 Term Weighting:
        # For a query: we just check word occurrences (usually binary weights or TF weights).
        # For documents: we compute standard BM25 weights using corpus statistics.
        #
        # If we are encoding queries, we assume TF mapping (as IDF is handled by the index).
        # For indexing documents, we compute the full document BM25 weights:
        #   Score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (doc_len / avg_doc_len)))
        
        tokenized_corpus = []
        doc_freqs: dict[int, int] = {}
        
        # 1. Tokenize corpus
        for text in texts:
            # Add_special_tokens=False ignores [CLS] and [SEP]
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            tokenized_corpus.append(tokens)
            
            # Record document frequency for IDFs
            unique_tokens = set(tokens)
            for t in unique_tokens:
                doc_freqs[t] = doc_freqs.get(t, 0) + 1

        total_docs = len(texts)
        doc_lengths = [len(tokens) for tokens in tokenized_corpus]
        avg_doc_len = sum(doc_lengths) / total_docs if total_docs > 0 else 1.0

        # 2. Compute Inverse Document Frequencies (IDF)
        idfs = {}
        for t, df in doc_freqs.items():
            # Standard BM25 IDF formula
            idfs[t] = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

        # 3. BM25 Hyperparameters
        k1 = 1.2
        b = 0.75

        # 4. Generate sparse vectors
        encoded_docs = []
        for tokens, doc_len in zip(tokenized_corpus, doc_lengths):
            # Calculate Term Frequency (TF) in this doc
            tf: dict[int, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
                
            indices = []
            values = []

            for t, count in tf.items():
                if is_query:
                    # For queries, term weight is binary or log-scaled TF
                    weight = 1.0
                else:
                    # For documents, calculate complete BM25 weight
                    idf = idfs.get(t, 0.0)
                    num = count * (k1 + 1)
                    denom = count + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
                    weight = idf * (num / denom)

                # Filter out negligible weights
                if weight > 1e-4:
                    indices.append(int(t))
                    values.append(float(weight))

            # Qdrant requires sparse index IDs to be strictly sorted ascending
            sorted_pairs = sorted(zip(indices, values))
            if sorted_pairs:
                indices, values = zip(*sorted_pairs)
                indices = list(indices)
                values = list(values)
            else:
                indices, values = [], []

            encoded_docs.append({
                "indices": indices,
                "values": values
            })

        return encoded_docs
