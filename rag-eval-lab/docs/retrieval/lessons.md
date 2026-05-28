# 📚 Retrieval & Reranking Layer: Dual-Engine search & Fusion Math

In a modern, production-grade retrieval system, relying on a single retrieval vector or strategy is insufficient. A robust search layer combines **Dense Semantic** vectors with **Sparse Lexical** weights, fuses their rankings mathematically, and refines them using neural **Rerankers**.

This guide covers the core concepts, mathematical foundations, and multi-stage pipelines implemented in the **Retrieval Layer** of our RAG Eval Lab.

---

## 1. Dense Semantic Retrieval

Dense retrieval represents documents and queries as continuous float vectors in a high-dimensional vector space (e.g., 384 dimensions for `bge-small-en-v1.5` via Ollama).

### High-Dimensional Representation
* A text chunk is passed through a transformer encoder, producing a dense vector $\mathbf{v} \in \mathbb{R}^d$.
* The query is embedded using the same model to produce query vector $\mathbf{q} \in \mathbb{R}^d$.
* Similarity is calculated using **Cosine Similarity**:
  $$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$

```text
Continuous Multi-Dimensional Vector Space:
        |           * Document A (Vector index, semantic canine)
        |          / 
        |         / (Short angle theta)
        |        /
        |       * Query: "A dog running"
        |
        +----------------------------
```

### HNSW: Hierarchical Navigable Small World Graphs
Calculating cosine similarity between a query vector and millions of document vectors in real time is computationally prohibitive ($O(N)$ linear scans). To solve this, Qdrant constructs an **HNSW Graph**:
* It builds a multi-layer graph structure where the top layers have wide-range links (long "expressways" across space) and bottom layers have short-range links (local details).
* The search agent starts at the top layer, takes long jumps to get close to the target area, drops down a layer, and executes short local steps to find the exact nearest neighbors.
* This accelerates similarity search to logarithmic time: **$O(\log N)$**.

---

## 2. Sparse Lexical Search: BM25 vs. SPLADE

While dense search excels at conceptual synonyms, it fails at exact serial numbers, unique names, or code syntax. Lexical search targets exact keyword matching, weighted by word statistical importance.

### A. Classic BM25 (Best Match 25)
BM25 is a frequency-based lexical ranking function. For a document $D$ and query $Q$ containing search terms $q_1, q_2, \dots, q_n$, the score is defined as:

$$\text{Score}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
* $f(q_i, D)$ is the raw frequency of token $q_i$ inside document $D$.
* $|D|$ is the length of the document in words, and $\text{avgdl}$ is the average length of all documents across the index.
* $k_1$ is a scaling parameter (typically $1.2$ to $2.0$) that controls **term frequency saturation**. As a word repeats, its marginal score value increases on a logarithmic curve, preventing a document repeating "dog" 100 times from completely dominating the search.
* $b$ is a length normalization parameter (typically $0.75$). It penalizes long documents; a single match in a short headline is highly significant, whereas a single match in a 50-page thesis is likely noise.
* $\text{IDF}(q_i)$ is the **Inverse Document Frequency**, measuring how rare the term is across the entire corpus:
  $$\text{IDF}(q_i) = \ln\left(1 + \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}\right)$$
  Where $N$ is the total documents in the index, and $n(q_i)$ is the number of documents containing term $q_i$. Common words ("the", "and") receive a weight near $0.0$, while rare words ("torvalds") receive high values.

---

### B. SPLADE (Learned Neural Sparse Expansion)
Traditional BM25 suffers from the **vocabulary mismatch problem**: if the document contains *"dog"* and the query is *"canine"*, BM25 scores the match at exactly $0.0$. 

**SPLADE (Sparse Lexical and Expansion)** solves this by utilizing a deep neural network (a BERT-style model) to predict term importances over the **entire vocabulary** ($30,522$ tokens) for a given text.

```text
Input text: "A dog barked."
             │
             ▼
      [ SPLADE Model ]
             │
             ▼
Sparse Vector representation:
{
  "dog": 3.42,       <-- Physical word weight
  "canine": 2.15,    <-- Synonyms Concept Expansion
  "bark": 2.89,
  "leash": 1.05      <-- Relational Concept Expansion
}
```

#### The Mathematics of Neural Expansion
SPLADE processes a sequence of text to produce raw logit predictions $x_{t, j}$ for each vocabulary token $j$ at each sequence position $t$:
1. **Activation & Saturation**: We apply a **Rectified Linear Unit (ReLU)** and log-scaling to ensure weights are non-negative and non-linear:
   $$w_{t, j} = \log(1 + \max(0, x_{t, j}))$$
2. **Sequence-Level Max Pooling**: To extract a single global weight for each vocabulary token $j$ across the entire text sequence, we take the maximum value across all positions:
   $$\mathbf{w}_j = \max_{t \in \text{sequence}} w_{t, j}$$
3. **Sparsity Constraints**: During training, SPLADE forces most weights to exactly $0.0$ using a FLOPS regularizer or $L_1$ penalty:
   $$\mathcal{L}_{\text{sparsity}} = \lambda \sum_{j=1}^V |\mathbf{w}_j|$$
   This ensures that $95\%+$ of the 30,522 vocabulary scores are $0$. The remaining active tokens are written directly to Qdrant's sparse inverted index, combining the conceptual understanding of embeddings with the lightning-fast matching of inverted indices.

---

## 3. Reciprocal Rank Fusion (RRF)

When executing a hybrid query, Qdrant returns two independent ranked lists:
* **Dense Search List**: Ordered by cosine similarity.
* **Sparse/SPLADE Search List**: Ordered by keyword/expansion weights.

Because these scores are on entirely different scales, we cannot simply add them. Instead, we use **Reciprocal Rank Fusion (RRF)**. RRF ignores raw score values and merges documents based purely on their **relative rank** (position) in each list.

### Mathematical Formulation
The RRF score for a candidate document $d$ within the combined document set $D$ is:

$$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where:
* $M$ is the set of retrieval systems (Dense and Sparse).
* $r_m(d)$ is the rank position of document $d$ (1-indexed) in retrieval system $m$. If a document is absent from a list, $r_m(d) \to \infty$ (score contribution becomes $0.0$).
* $k$ is a constant smoothing parameter (standard industry baseline is $60$). It regulates the impact of top ranks; without $k$, Rank 1 scores $\frac{1}{1} = 1.0$, which vastly overpowers Rank 2 ($\frac{1}{2} = 0.5$). With $k = 60$, Rank 1 is $\frac{1}{61} \approx 0.01639$ and Rank 2 is $\frac{1}{62} \approx 0.01612$, allowing for a balanced, robust consensus.

---

### Step-by-Step Hand-Evaluated RRF Example

Suppose our query returns the following top-5 lists for dense and sparse retrieval:

| Rank Position | Dense Retrieval Output | Sparse Retrieval Output |
| :---: | :--- | :--- |
| **Rank 1** | Document A | Document C |
| **Rank 2** | Document B | Document A |
| **Rank 3** | Document C | Document E |
| **Rank 4** | Document D | Document F |
| **Rank 5** | Document G | Document B |

We calculate the combined scores for all retrieved documents using $k = 60$:

#### A. Document A
* Dense Rank: $1 \to \frac{1}{60 + 1} \approx 0.01639$
* Sparse Rank: $2 \to \frac{1}{60 + 2} \approx 0.01613$
* **RRF Score**: $0.01639 + 0.01613 = \mathbf{0.03252}$

#### B. Document B
* Dense Rank: $2 \to \frac{1}{60 + 2} \approx 0.01613$
* Sparse Rank: $5 \to \frac{1}{60 + 5} \approx 0.01538$
* **RRF Score**: $0.01613 + 0.01538 = \mathbf{0.03151}$

#### C. Document C
* Dense Rank: $3 \to \frac{1}{60 + 3} \approx 0.01587$
* Sparse Rank: $1 \to \frac{1}{60 + 1} \approx 0.01639$
* **RRF Score**: $0.01587 + 0.01639 = \mathbf{0.03226}$

#### D. Document D
* Dense Rank: $4 \to \frac{1}{60 + 4} \approx 0.01563$
* Sparse Rank: Not Ranked $\to 0.00000$
* **RRF Score**: $0.01563 + 0.0 = \mathbf{0.01563}$

#### E. Document E
* Dense Rank: Not Ranked $\to 0.00000$
* Sparse Rank: $3 \to \frac{1}{60 + 3} \approx 0.01587$
* **RRF Score**: $0.0 + 0.01587 = \mathbf{0.01587}$

---

### Final Merged Ranking
Sorting all candidates by their RRF score yields the following consensus rank list:

1. **Document A** (Score: $0.03252$)
2. **Document C** (Score: $0.03226$)
3. **Document B** (Score: $0.03151$)
4. **Document E** (Score: $0.01587$)
5. **Document D** (Score: $0.01563$)

This consensus has elevated Document A and Document C (which performed well in both lists) above Document B, providing a highly reliable hybrid result.

---

## 4. Stage-2 Reranking Layers

Even with hybrid search, bi-encoder retrieval models are restricted. They encode document vectors and query vectors **independently** (meaning the document has no attention access to the query at encoding time). This is necessary for speed, but limits accuracy.

To refine the top retrieved results, we execute a **Stage-2 Reranking Layer** using a **Cross-Encoder**.

```text
Bi-Encoder (Retriever):
[ Query ]    ──> [ Embed Model ] ──> Vector (Q) ──┐
                                                  ├── Cosine Distance
[ Document ] ──> [ Embed Model ] ──> Vector (D) ──┘

Cross-Encoder (Reranker):
[ Query + Document ] ──> [ Transformer Core (Full Self-Attention) ] ──> Score (0 to 1)
```

### A. Cross-Encoder Reranking
* **How it works**: Instead of comparing static vectors, the query and document text are concatenated together and passed into a single transformer core simultaneously.
* **Full Self-Attention**: Every word in the query has direct, full-attention alignment to every word in the document text inside the transformer layers.
* **Accuracy vs. Cost**: Because self-attention is computed across the concatenated pair, it cannot be pre-calculated or indexed. Running a cross-encoder over millions of rows is impossible. However, running a cross-encoder over just the top-25 retrieved documents is fast and yields highly accurate, context-aware scoring.

### B. Hosted Cohere Rerank API
In production environments, rather than loading a heavy cross-encoder model into local memory (demanding local GPU/CPU resources), we can offload this to the **Cohere Rerank API** (`v3` endpoint).
* The Python pipeline sends the raw query and the top retrieved text strings to Cohere over an asynchronous REST endpoint.
* Cohere's massive proprietary multi-lingual models re-score the candidates and return a re-ordered array, which our pipeline trims to `reranker_top_k` (e.g., top-5) before passing them to the generator.
