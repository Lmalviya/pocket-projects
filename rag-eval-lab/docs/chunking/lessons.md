# 📚 Chunking Layer Masterclass: Structural vs. Semantic Segmentation

In information retrieval, **Chunking** is the foundational bridge between a raw, unstructured document and the discrete index units stored in a database. If your chunks are too large, retrieved contexts introduce irrelevant noise, polluting the LLM's prompt window. If they are too small, critical surrounding context is lost, rendering the LLM unable to synthesize accurate answers.

This guide provides a deep-dive analysis of the four chunking strategies implemented in our **RAG Eval Lab**.

---

## 1. Fixed-Token Chunking (`SentenceSplitter`)

Fixed-token chunking slices documents into uniform spans based on token count rather than character boundaries. We use the LlamaIndex `SentenceSplitter` (which acts as a token-aware splitter).

### Mechanism
1. Text is tokenized using a target tokenizer (e.g., Llama 3 or BGP tokenizer).
2. The algorithm fills a chunk until the token limit (`chunk_size`, e.g., 512 tokens) is reached.
3. To prevent cutting off sentences in the middle, the splitter backtracks to the nearest sentence boundary (e.g., standard end-of-sentence punctuation).
4. A sliding window starts the next chunk with a predefined `chunk_overlap` (e.g., 50 tokens) from the end of the previous chunk.

```text
Chunk 1:
[ Sentence A (50 tokens) ] [ Sentence B (300 tokens) ] [ Sentence C (162 tokens) ] = 512 tokens
                                                      └─────── OVERLAP ───────┘
Chunk 2:
                                                      [ Sentence C (162 tokens) ] [ Sentence D (200 tokens) ] ...
```

### Why Chunk Overlap is Critical
When text is sliced, important connections between adjacent sentences can be severed. For example, if Sentence B ends with *"The process works as follows:"* and Sentence C details the steps, splitting between B and C renders both chunks incomplete. Keeping an overlap ensures that boundary sentences appear in **both** neighboring chunks, preserving context continuity.

---

## 2. Recursive-Character Chunking

The standard workhorse of modern RAG systems. It splits text using a hierarchical list of delimiters, scanning them sequentially from most-preferred to least-preferred (usually `["\n\n", "\n", " ", ""]`).

### The Delimiter Hierarchy
* **`\n\n` (Paragraphs)**: The highest priority boundary. It preserves paragraph-level thoughts.
* **`\n` (Lines/List Items)**: Splits at line breaks, keeping bulleted lists and single sentences intact.
* **` ` (Words)**: If a single paragraph is still larger than the target chunk size, it splits at word boundaries (spaces) to prevent slicing words in half.
* **`""` (Characters)**: The absolute last resort. Slices individual letters if no other boundaries exist.

### Recursive Execution Loop
1. The splitter checks if the entire document fits within the `chunk_size`. If yes, it stops.
2. If not, it splits the document by the first delimiter (`\n\n`).
3. It recursively evaluates each resulting paragraph. If a paragraph is smaller than the chunk size, it is finalized.
4. If any single paragraph still exceeds the chunk size, the splitter goes to the next delimiter (`\n`) for that paragraph, splitting it further.
5. It repeats this downwards through the list until all slices satisfy the chunk size constraint.

> [!TIP]
> **Why it's preferred over naive fixed splitting**: By respecting paragraphs (`\n\n`) and lines (`\n`), it keeps structurally related statements together, avoiding disjointed fragments.

---

## 3. Sentence-Boundary Chunking

Sentence-boundary chunking uses a natural language parser or custom regular expressions to isolate complete linguistic statements.

### Core Logic
* The document is parsed to identify sentence termination symbols (`.`, `?`, `!`) while ignoring abbreviations (e.g., *"Dr."*, *"e.g."*, *"U.S.A."*).
* Individual sentences are grouped together until their combined token length approaches the `chunk_size`.
* When the limit is reached, a split is made, and a new chunk begins at the next complete sentence.

### Strengths
* **Zero Fragmented Sentences**: Every retrieved chunk starts and ends with a complete, grammatically valid sentence.
* **Excellent for Sentence-Transformers**: Many embedding models are explicitly trained on single sentences or small groups of sentences. Keeping grammatical structures intact improves embedding quality.

---

## 4. Semantic Chunking

As detailed in our [Semantic Chunking Deep-Dive](file:///c:/Users/23add/workspace/pocket-projects/rag-eval-lab/docs/chunking/semantic.md), this strategy is purely **meaning-driven**.

* **How it works**: It splits text into individual sentences, generates high-dimensional embeddings for overlapping sliding sentence buffers, computes consecutive cosine distances, and inserts boundaries where the semantic distance spikes above a statistical threshold (percentile, standard deviation, or interquartile range).
* **Strategic Value**: Chunks represent coherent thematic concepts rather than arbitrary lengths of text.

---

## 5. Comparative Trade-Offs Matrix

The following table summarizes the operational and structural differences between the four chunking techniques:

| Feature / Metric | Fixed-Token (`SentenceSplitter`) | Recursive-Character | Sentence-Boundary | Semantic Chunking |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Delimiter** | Token Count + Punctuation | Hierarchical List (`\n\n`, `\n`, ` `) | Grammatical Sentence Bounds | Semantic Cosine Distance Spikes |
| **Token Size Stability** | **High** (uniform chunk lengths) | **Medium** (varies by paragraph size) | **Medium** (varies by sentence counts) | **Low** (highly variable; topic-dependent) |
| **Processing Speed** | **Ultra-Fast** (instantaneous regex/token counters) | **Fast** (simple string splittings) | **Medium** (requires parsing/regex structures) | **Slow** (demands GPU/CPU embedding calls) |
| **Retrieval Recall** | **Low-Medium** (includes edge noise) | **Medium-High** (keeps paragraph contexts) | **Medium** (grammatically clean) | **Excellent** (matches exact thematic scope) |
| **Context Window Noise** | **High** (often carries leftover words/lines) | **Medium** (preserves paragraphs) | **Low-Medium** (clean sentence starts) | **Minimal** (only contains relevant topics) |
| **Best Application** | Fast, generic RAG baseline setups | Technical code, books, standard articles | Short Q&A, sentence-level semantic indexes | Academic papers, dense reports, legal briefs |

---

## 6. Engineering Recommendations

To select the optimal chunker for your production application, apply these guidelines:

1. **Use Recursive-Character** as your default starting layout. It is extremely fast, highly predictable, and preserves lists and logical markdown paragraph groupings.
2. **Upgrade to Semantic Chunking** if you are processing complex research documents, legal briefs, or scientific literature where topics change abruptly and retrieving a cohesive, complete concept is critical for generation accuracy.
3. **Avoid Semantic Chunking** in real-time online ingestion pipelines where low-latency ingestion is a requirement, as generating embeddings for every sentence during upload is CPU/GPU heavy and introduces processing lag.
