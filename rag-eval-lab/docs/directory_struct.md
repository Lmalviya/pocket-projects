app/
├── config/
│   ├── settings.py              # central config (model names, paths, env vars)
│   └── experiment.yaml          # which strategies to use for a given run
│
├── ingestion/                   # one-time: load + chunk + index
│   ├── loader.py                # load Wikipedia articles, HotpotQA filter
│   ├── chunking/
│   │   ├── base.py              # abstract base class
│   │   ├── fixed.py             # fixed size (512 tokens, overlap 50)
│   │   ├── semantic.py          # semantic chunking (embedding similarity)
│   │   ├── recursive.py         # langchain recursive text splitter
│   │   └── sentence.py          # sentence-aware chunking
│   └── indexing/
│       ├── dense.py             # embed + store in vector db (qdrant/chroma)
│       └── sparse.py            # BM25 index (bm25s / elasticsearch)
│
├── retrieval/
│   ├── base.py                  # abstract retriever interface
│   ├── dense.py                 # dense-only retrieval
│   ├── sparse.py                # BM25-only retrieval
│   ├── hybrid.py                # BM25 + dense fusion (RRF)
│   └── reranker/
│       ├── base.py
│       ├── cross_encoder.py     # cross-encoder reranking
│       └── cohere.py            # cohere rerank API
│
├── generation/
│   ├── prompts/                 # prompt templates (versioned via promptfoo)
│   │   ├── single_turn.yaml
│   │   └── multi_turn.yaml
│   ├── single_turn.py           # single Q&A chain
│   └── multi_turn.py           # conversation chain with history
│
├── pipeline/
│   ├── base.py                  # abstract pipeline
│   ├── single_turn.py           # wires ingestion→retrieval→generation
│   └── multi_turn.py
│
├── tracing/
│   └── langfuse.py              # langfuse decorators, span wrappers
│
└── main.py                      # entry point, reads experiment.yaml, runs pipeline