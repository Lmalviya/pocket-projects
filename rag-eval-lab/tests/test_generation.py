"""
tests/test_generation.py
=========================
Unit tests for the RAG Generation layer components.

Tests:
  - SingleTurnGenerator prompt parsing and factual answer synthesis (mocked LLM).
  - MultiTurnGenerator state machine transitions and conversational history updates in LangGraph.
"""

import unittest.mock as mock
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from llama_index.core.schema import NodeWithScore, TextNode

from app.config.settings import Settings
from app.generation.multi_turn import MultiTurnGenerator
from app.generation.single_turn import SingleTurnGenerator


@pytest.fixture
def test_settings():
    """Returns a test settings instance."""
    return Settings(
        nvidia_api_key="mock-nv-key",
        langfuse_public_key="mock-lf-public",
        langfuse_secret_key="mock-lf-secret",
    )


@pytest.fixture
def mock_nodes():
    """Returns a list of sample retrieved NodeWithScore objects."""
    return [
        NodeWithScore(
            node=TextNode(text="Linux is a free and open-source operating system kernel created by Linus Torvalds."),
            score=0.95
        ),
        NodeWithScore(
            node=TextNode(text="Git was created by Linus Torvalds in 2005 for development of the Linux kernel."),
            score=0.88
        )
    ]


def test_single_turn_generator(test_settings, mock_nodes):
    """Test the single-turn RAG generation workflow."""
    # 1. Patch ChatOpenAI inside single_turn to prevent live NVIDIA API network requests
    with mock.patch("app.generation.single_turn.ChatOpenAI") as mock_chat_openai:
        mock_llm = mock.Mock()
        # Mock LLM invoke and direct call return values (should mimic AIMessage payload)
        aimsg = AIMessage(content="Linus Torvalds created the Linux operating system kernel.")
        mock_llm.invoke.return_value = aimsg
        mock_llm.return_value = aimsg
        mock_chat_openai.return_value = mock_llm

        # 2. Instantiate and run the generator
        generator = SingleTurnGenerator(settings=test_settings, tracer=None)
        res = generator.generate(query="Who created Linux?", nodes=mock_nodes)

        # 3. Verify assertions
        assert res.answer == "Linus Torvalds created the Linux operating system kernel."
        assert len(res.source_nodes) == 2
        assert res.prompt_version == "v1"  # Matches our local single_turn.yaml version
        mock_llm.assert_called_once()


def test_multi_turn_generator_flow(test_settings, mock_nodes):
    """Test the multi-turn conversational LangGraph state transitions."""
    # 1. Patch ChatOpenAI inside multi_turn to prevent live network requests
    with mock.patch("app.generation.multi_turn.ChatOpenAI") as mock_chat_openai:
        mock_llm = mock.Mock()
        # In LangGraph message streams, the LLM takes combined messages list and returns AIMessage
        aimsg = AIMessage(content="Yes, he also created Git in 2005.")
        mock_llm.invoke.return_value = aimsg
        mock_llm.return_value = aimsg
        mock_chat_openai.return_value = mock_llm

        # 2. Instantiate and run the generator
        generator = MultiTurnGenerator(settings=test_settings, tracer=None)
        
        # We simulate a follow-up question: "Did he also create Git?"
        # The history holds the first exchange: Q: "Who created Linux?" / A: "Linus Torvalds."
        history = [
            HumanMessage(content="Who created Linux?"),
            AIMessage(content="Linus Torvalds.")
        ]
        
        res = generator.generate_turn(
            query="Did he also create Git?",
            nodes=mock_nodes,
            history=history
        )

        # 3. Verify assertions
        assert res.answer == "Yes, he also created Git in 2005."
        assert len(res.source_nodes) == 2
        assert res.prompt_version == "v1"  # Matches our local multi_turn.yaml version
        mock_llm.invoke.assert_called_once()
        
        # Verify the message structure passed to LLM (System + 2 historical + 1 new Query = 4 messages)
        actual_messages = mock_llm.invoke.call_args[0][0]
        assert len(actual_messages) == 4
        assert actual_messages[0].type == "system"  # Combines context
        assert "operating system kernel" in actual_messages[0].content
        assert actual_messages[1].content == "Who created Linux?"
        assert actual_messages[2].content == "Linus Torvalds."
        assert actual_messages[3].content == "Did he also create Git?"
