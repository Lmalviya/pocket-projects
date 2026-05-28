"""
app/utils/text.py
==================
Text processing helper functions used across the RAG pipeline.

📚 LESSON — Why extract helpers into a utils file?
---------------------------------------------------
These functions are small, pure (no side effects), and needed by multiple
modules (chunkers, generators, retrievers). Keeping them in a dedicated
utils file means:
  ✅ No circular imports (utils doesn't import from app.*)
  ✅ Easy to unit test in isolation
  ✅ Reusable — add once, use everywhere
"""

from llama_index.core.schema import NodeWithScore


def truncate_text(text: str, max_chars: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum number of characters.

    Used when logging context to Langfuse — we don't want to send 10,000
    characters of context as a span input; just enough to debug.

    Args:
        text: The input string.
        max_chars: Maximum character length before truncation.
        suffix: String appended when truncation occurs.

    Returns:
        Original text if short enough, otherwise truncated + suffix.

    Example:
        truncate_text("Hello world, this is long", max_chars=10)
        # "Hello w..."
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def nodes_to_context_str(nodes: list[NodeWithScore], separator: str = "\n\n---\n\n") -> str:
    """
    Convert a list of retrieved nodes into a single context string for the LLM.

    📚 LESSON — This is the "context building" step in RAG:
      1. User asks a question
      2. We retrieve the top-K most relevant chunks (nodes)
      3. We JOIN those chunks into one big "context" string
      4. We inject that context into the LLM prompt

    The LLM then answers ONLY using information in that context — this is
    what prevents hallucination (the model can't make up facts if forced
    to cite from retrieved text).

    Args:
        nodes: List of NodeWithScore objects from retrieval.
        separator: String inserted between chunks. Default uses markdown HR.

    Returns:
        A single string with all chunk texts joined by the separator.

    Example:
        context = nodes_to_context_str(retrieved_nodes)
        # "Chunk 1 text...\n\n---\n\nChunk 2 text...\n\n---\n\nChunk 3 text..."
    """
    if not nodes:
        return ""

    parts = []
    for i, node in enumerate(nodes, start=1):
        # node.node.get_content() returns the raw text of the chunk
        # node.score is the similarity score (higher = more relevant)
        text = node.node.get_content()
        parts.append(f"[Source {i}]\n{text}")

    return separator.join(parts)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Estimate the number of tokens in a text string using tiktoken.

    📚 LESSON — Why count tokens?
      LLMs have a "context window" — a maximum number of tokens they can
      process in one call. For example, llama-3.1 has a 128K token window.
      If we inject too much context, the request will fail (or get truncated).

      Token counting lets us:
        - Warn when context is getting too large
        - Implement "context budgeting" (pick top-K chunks that fit in N tokens)
        - Track costs (most APIs charge per token)

      tiktoken is OpenAI's tokenizer, but it's a reasonable approximation
      for most modern LLMs because they all use similar BPE tokenization.

    Args:
        text: Input string to count tokens for.
        model: The model name for tokenizer selection (default: gpt-4o).

    Returns:
        Approximate token count as an integer.
    """
    try:
        import tiktoken
        # cl100k_base is the encoding used by GPT-4, GPT-3.5, and most modern models
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback if the model name isn't recognized by tiktoken
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

    return len(enc.encode(text))


def format_sources(nodes: list[NodeWithScore]) -> str:
    """
    Format retrieved sources as a human-readable summary string.

    Useful for displaying "Cited sources" below an answer in the CLI.

    Args:
        nodes: Retrieved nodes with scores.

    Returns:
        A formatted string listing each source with its score.

    Example output:
        Sources used:
          [1] score=0.87 | Wikipedia: Retrieval-Augmented Generation
          [2] score=0.81 | Wikipedia: Large language model
    """
    if not nodes:
        return "No sources retrieved."

    lines = ["Sources used:"]
    for i, node in enumerate(nodes, start=1):
        score = node.score or 0.0
        # Try to get a meaningful title from node metadata
        metadata = node.node.metadata or {}
        source = metadata.get("title") or metadata.get("source") or "Unknown"
        lines.append(f"  [{i}] score={score:.3f} | {source}")

    return "\n".join(lines)
