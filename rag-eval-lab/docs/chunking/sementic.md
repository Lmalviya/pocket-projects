# Semantic Chunking in LangChain — Notes

"""
app/ingestion/chunking/semantic.py
====================================
Semantic chunking using embedding-based similarity breakpoints.

📚 LESSON — Semantic Chunking: The Most Intelligent Strategy
-------------------------------------------------------------
The previous strategies (fixed, recursive, sentence) split text based on
STRUCTURE: character count, sentence boundaries, paragraphs.

Semantic chunking splits text based on MEANING:
  "Split here because the topic just changed significantly."

Algorithm:
  1. Split text into base units (usually sentences)
  2. Embed each sentence using an embedding model
  3. Compute cosine similarity between consecutive sentence embeddings
  4. Find "breakpoints" — places where similarity drops sharply
     (= topic shift)
  5. Group sentences between breakpoints into chunks

Visual example:
  Sentence 1: "Paris is the capital of France." ──┐
  Sentence 2: "It has a population of 2.1M."    ──┤ SIMILAR TOPIC → same chunk
  Sentence 3: "The Eiffel Tower was built 1889." ──┘
  Similarity drops ↓↓↓ (BREAKPOINT DETECTED)
  Sentence 4: "Python is a programming language." ──┐
  Sentence 5: "It was created by Guido van Rossum" ──┘ SIMILAR TOPIC → new chunk

Strengths:
  ✅ Chunks represent coherent topics, not arbitrary text windows
  ✅ Retrieval quality is much higher — queries match complete ideas
  ✅ Context windows contain relevant information only

Weaknesses:
  ❌ Requires an embedding model call for EVERY sentence during indexing
  ❌ Much slower than structural approaches (5-50x slower)
  ❌ Chunk sizes vary dramatically and can be very large
  ❌ Sensitive to the breakpoint_threshold parameter

When to use:
  - High-quality RAG where latency doesn't matter
  - Long documents with multiple distinct topics
  - Academic papers, books, research reports

📚 LESSON — Breakpoint Types
-----------------------------
LangChain's SemanticChunker supports three threshold methods:

  "percentile"         → split at positions where similarity is in the bottom X%
                         percentile of all similarities. Adaptive to document.
  "standard_deviation" → split where similarity drops more than Z std deviations
                         below the mean. Good for consistent topic density.
  "interquartile"      → split where similarity is below Q1 - 1.5*IQR.
                         Robust to outliers (single very-different sentence).

Default: "percentile" with threshold 95 (split the 5% most different transitions).
"""

---

# What is Semantic Chunking?

Traditional chunking splits text using:

* fixed token count
* character limits
* recursive separators

Semantic chunking instead tries to split text based on **topic changes**.

The overall pipeline is:

1. Split document into sentences
2. Generate embeddings for sentences (or sentence groups)
3. Compute semantic similarity between neighboring sentence groups
4. Detect semantic jumps
5. Create chunk boundaries at those jumps

---

# Core Concept

Semantic chunking does NOT compare:

```text
sentence1 vs all sentences
```

Instead, it compares only **neighboring sentences/groups**.

For example:

```text
S1
S2
S3
S4
S5
```

Comparisons are:

```text
S1 ↔ S2
S2 ↔ S3
S3 ↔ S4
S4 ↔ S5
```

This helps detect:

> “Where does the topic change next?”

---

# Cosine Similarity

Embeddings are vectors.

Similarity between two vectors is computed using cosine similarity.

Formula:

```latex
cos(θ) = (A · B) / (|A||B|)
```

Where:

* A · B = dot product
* |A| = vector magnitude

Similarity range:

| Similarity | Meaning            |
| ---------- | ------------------ |
| 1.0        | identical meaning  |
| 0.8        | very similar       |
| 0.5        | somewhat related   |
| 0.0        | unrelated          |
| -1.0       | opposite direction |

LangChain usually converts similarity to distance:

```python
semantic_distance = 1 - cosine_similarity
```

So:

| Similarity | Distance |
| ---------- | -------- |
| 0.95       | 0.05     |
| 0.50       | 0.50     |
| 0.10       | 0.90     |

Large distance = large topic shift.

---

# Full Hand-Evaluated Example

Suppose document contains:

```text
S1: LangChain helps build LLM applications.

S2: It provides abstractions for prompts and memory.

S3: Vector databases are useful for semantic search.

S4: Pinecone stores vector embeddings efficiently.

S5: The IPL final was exciting this year.
```

Topics:

* S1 + S2 → LangChain
* S3 + S4 → Vector DBs
* S5 → unrelated topic

---

# Step 1 — Convert Sentences to Embeddings

Assume simplified embeddings:

```python
S1 = [0.90, 0.10]
S2 = [0.85, 0.15]

S3 = [0.20, 0.80]
S4 = [0.18, 0.82]

S5 = [-0.90, 0.05]
```

In reality embeddings are typically:

* 384 dimensions
* 768 dimensions
* 1024+ dimensions

depending on embedding model.

---

# Step 2 — Compute Consecutive Similarities

## Similarity(S1, S2)

```python
S1 = [0.90, 0.10]
S2 = [0.85, 0.15]
```

Dot product:

```python
0.90×0.85 + 0.10×0.15
= 0.765 + 0.015
= 0.78
```

Magnitudes:

```python
|S1| ≈ 0.905
|S2| ≈ 0.863
```

Cosine similarity:

```python
0.78 / (0.905 × 0.863)
≈ 0.998
```

Distance:

```python
1 - 0.998 = 0.002
```

Interpretation:

```text
Very similar topic
```

---

## Similarity(S2, S3)

```python
S2 = [0.85, 0.15]
S3 = [0.20, 0.80]
```

Dot product:

```python
0.85×0.20 + 0.15×0.80
= 0.17 + 0.12
= 0.29
```

Magnitudes:

```python
|S2| ≈ 0.863
|S3| ≈ 0.825
```

Cosine similarity:

```python
0.29 / (0.863 × 0.825)
≈ 0.407
```

Distance:

```python
1 - 0.407 = 0.593
```

Interpretation:

```text
Large semantic jump
```

---

## Similarity(S3, S4)

Assume:

```python
similarity ≈ 0.999
distance ≈ 0.001
```

Interpretation:

```text
Almost same topic
```

---

## Similarity(S4, S5)

Assume:

```python
similarity ≈ -0.15
distance ≈ 1.15
```

Interpretation:

```text
Huge topic change
```

---

# Final Distance Array

```python
distances = [
  0.002,
  0.593,
  0.001,
  1.15
]
```

Each value represents semantic change between neighboring sentence groups.

---

# Thresholding

Now LangChain decides:

> “Which distances are large enough to create chunk boundaries?”

This is controlled by:

```python
breakpoint_threshold_type
breakpoint_threshold_amount
```

---

# 1. Percentile Threshold

Example:

```python
breakpoint_threshold_type="percentile"
breakpoint_threshold_amount=80
```

Algorithm:

```python
threshold = percentile(distances, 80)
```

Sorted distances:

```python
[0.001, 0.002, 0.593, 1.15]
```

80th percentile ≈ 0.82

Rule:

```python
split if distance > 0.82
```

Only:

```python
1.15
```

creates breakpoint.

Final chunks:

```text
Chunk 1:
S1
S2
S3
S4

Chunk 2:
S5
```

---

# Lower Threshold Example

Suppose threshold becomes:

```python
0.4
```

Now:

```python
0.593
1.15
```

both become breakpoints.

Chunks:

```text
Chunk 1:
S1
S2

Chunk 2:
S3
S4

Chunk 3:
S5
```

Lower threshold → more chunks.

---

# 2. Standard Deviation Threshold

Example:

```python
breakpoint_threshold_type="standard_deviation"
breakpoint_threshold_amount=1.25
```

Formula:

```latex
T = μ + kσ
```

Where:

* μ = mean distance
* σ = standard deviation
* k = multiplier

Example:

```python
mean = 0.23
std = 0.27
k = 1.25
```

Threshold:

```python
0.23 + (1.25 × 0.27)
≈ 0.5675
```

Split if:

```python
distance > 0.5675
```

Breakpoints:

```python
0.593
1.15
```

This method detects statistical outliers.

---

# 3. Interquartile Threshold

Example:

```python
breakpoint_threshold_type="interquartile"
breakpoint_threshold_amount=1.5
```

Formula:

```latex
IQR = Q3 - Q1
```

Threshold:

```latex
T = μ + k(IQR)
```

Where:

* Q1 = 25th percentile
* Q3 = 75th percentile
* IQR = interquartile range

This method is more resistant to noisy distributions.

---

# Buffered Windowing in Real LangChain

Real SemanticChunker usually does NOT compare raw individual sentences.

Instead it compares buffered windows.

Instead of:

```text
S1 ↔ S2
```

it may compare:

```text
[S1,S2,S3]
vs
[S2,S3,S4]
```

Benefits:

* smoother embeddings
* fewer noisy splits
* more stable chunking
* better topic continuity

---

# Why Consecutive Comparison Only?

Chunking only needs local transitions.

Comparing every sentence with every other sentence would:

* increase complexity to O(n²)
* introduce noisy relationships
* create unnecessary global comparisons

Semantic chunking is fundamentally:

```text
same topic
same topic
same topic
NEW TOPIC
same topic
NEW TOPIC
```

It scans text sequentially looking for topic transitions.

---

# Practical Recommendations

| Document Type     | Recommended Strategy |
| ----------------- | -------------------- |
| General RAG       | percentile 85–95     |
| Technical docs    | percentile 80–90     |
| Legal docs        | interquartile        |
| OCR-heavy docs    | interquartile        |
| Scientific papers | standard_deviation   |

Good starting configuration:

```python
SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=85,
)
```

---

# Important Takeaways

1. Semantic chunking compares neighboring text regions only.
2. Topic changes are detected using embedding distance.
3. Larger distance = larger semantic shift.
4. Thresholding decides where chunk boundaries occur.
5. Lower thresholds create smaller/more chunks.
6. Real implementations use buffered windows for stability.
7. Percentile thresholding is most commonly used in production RAG systems.
