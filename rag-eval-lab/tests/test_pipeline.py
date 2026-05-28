"""
tests/test_pipeline.py
======================
Unit and smoke tests for end-to-end RAG execution pipelines.

Tests:
  - SingleTurnPipeline orchestration flow (retrieval -> optional reranking -> LLM generation).
  - MultiTurnPipeline conversation memory appending and session clearing.
"""

import unittest.mock as mock
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from llama_index.core.schema import NodeWithScore, TextNode

from app.config.settings import Settings
from app.generation.single_turn import GenerationResult
from app.pipeline.base import PipelineResult
from app.pipeline.multi_turn import MultiTurnPipeline
from app.pipeline.single_turn import SingleTurnPipeline


@pytest.fixture
def test_settings():
    """Returns a test settings instance."""
    return Settings(
        nvidia_api_key="mock-nv-key",
        langfuse_public_key="mock-lf-public",
        langfuse_secret_key="mock-lf-secret",
    )


@pytest.fixture
def mock_retriever():
    """Mock retriever returning a list of candidate chunks."""
    retriever = mock.Mock()
    retriever.name = "mock_retriever"
    retriever.retrieve.return_value = [
        NodeWithScore(node=TextNode(text="Linux operating system kernel."), score=0.9),
        NodeWithScore(node=TextNode(text="Git development manager."), score=0.8)
    ]
    return retriever


@pytest.fixture
def mock_reranker():
    """Mock Stage-2 reranker sorting candidate chunks."""
    reranker = mock.Mock()
    reranker.name = "mock_reranker"
    # Reverse order or keep
    reranker.rerank.return_value = [
        NodeWithScore(node=TextNode(text="Git development manager."), score=0.95),
        NodeWithScore(node=TextNode(text="Linux operating system kernel."), score=0.85)
    ]
    return reranker


@pytest.fixture
def mock_single_turn_generator():
    """Mock single-turn generator returning a generation container."""
    generator = mock.Mock()
    generator.generate.return_value = GenerationResult(
        answer="Mocked factual answer.",
        source_nodes=[],
        prompt_version="v1"
    )
    return generator


@pytest.fixture
def mock_multi_turn_generator():
    """Mock multi-turn conversational graph generator."""
    generator = mock.Mock()
    generator.generate_turn.return_value = GenerationResult(
        answer="Mocked conversational reply.",
        source_nodes=[],
        prompt_version="v1"
    )
    return generator


def test_single_turn_pipeline(test_settings, mock_retriever, mock_reranker, mock_single_turn_generator):
    """Test end-to-end single-turn execution with retrieval, reranking, and generation."""
    # 1. Assemble pipeline with reranker
    pipeline = SingleTurnPipeline(
        settings=test_settings,
        retriever=mock_retriever,
        generator=mock_single_turn_generator,
        reranker=mock_reranker,
        tracer=None
    )

    # 2. Run query
    res = pipeline.run("Who created Linux?")

    # 3. Verify assertions
    assert isinstance(res, PipelineResult)
    assert res.query == "Who created Linux?"
    assert res.answer == "Mocked factual answer."
    assert len(res.retrieved_nodes) == 2
    assert res.latency_ms > 0.0

    # Ensure coordinates called all stages in correct sequence
    mock_retriever.retrieve.assert_called_once_with("Who created Linux?")
    mock_reranker.rerank.assert_called_once()
    mock_single_turn_generator.generate.assert_called_once()


def test_multi_turn_pipeline_memory(test_settings, mock_retriever, mock_multi_turn_generator):
    """Test multi-turn conversational pipelines and history accumulations."""
    # 1. Assemble pipeline without reranker
    pipeline = MultiTurnPipeline(
        settings=test_settings,
        retriever=mock_retriever,
        generator=mock_multi_turn_generator,
        reranker=None,
        tracer=None
    )

    # Verify memory is empty at startup
    assert len(pipeline.history) == 0

    # 2. Execute Turn 1
    res1 = pipeline.run("Who created Linux?")
    assert res1.answer == "Mocked conversational reply."
    
    # Verify turn appended HumanMessage and AIMessage to memory (2 items)
    assert len(pipeline.history) == 2
    assert isinstance(pipeline.history[0], HumanMessage)
    assert pipeline.history[0].content == "Who created Linux?"
    assert isinstance(pipeline.history[1], AIMessage)
    assert pipeline.history[1].content == "Mocked conversational reply."

    # 3. Execute Turn 2
    res2 = pipeline.run("Did he also create Git?")
    assert res2.answer == "Mocked conversational reply."
    
    # Verify turn appended another pair of messages (total 4 items)
    assert len(pipeline.history) == 4
    assert pipeline.history[2].content == "Did he also create Git?"

    # 4. Clear conversation memory
    pipeline.clear_history()
    assert len(pipeline.history) == 0
