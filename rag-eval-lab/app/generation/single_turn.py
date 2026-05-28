"""
app/generation/single_turn.py
==============================
Single-turn RAG answer generator using the NVIDIA API and Langfuse Prompt Management.

📚 LESSON — Dynamic Prompt Management & Telemetry:
In a mature RAG application, you should never hardcode prompt templates in the code.
If a prompt change is needed to reduce hallucinations, editing code and redeploying
takes hours/days. Instead, we use **Langfuse Prompt Management**:
  1. At runtime, we request the prompt template dynamically from the Langfuse server:
     `client.get_prompt("rag-qna-prompt", version=...)`
  2. This allows hot-swapping prompts via the Langfuse Web UI instantly in production.
  3. We build a **local YAML fallback** so the application continues to run even if
     the Langfuse server is offline or the prompt doesn't exist yet in the registry.

We use the OpenAI-compatible LangChain ChatOpenAI adapter pointed to the NVIDIA
Nemotron API (no structured outputs supported).
"""

import os
from dataclasses import dataclass
import yaml

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from llama_index.core.schema import NodeWithScore

from app.config.settings import Settings
from app.tracing.langfuse import LangfuseTracer, get_langfuse_callback, get_langfuse_client, get_safe_tracer
from app.utils.logger import get_logger
from app.utils.text import nodes_to_context_str

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """
    Data container holding the LLM generated answer and reference parameters.
    """
    answer: str
    source_nodes: list[NodeWithScore]
    prompt_version: str
    token_usage: dict | None = None


class SingleTurnGenerator:
    """
    Orchestrates RAG answer synthesis for single-turn Q&A sessions.
    """

    def __init__(self, settings: Settings, tracer: LangfuseTracer | None = None) -> None:
        """
        Initialize the generator.

        Args:
            settings: Central Settings instance.
            tracer: Optional LangfuseTracer instance.
        """
        self.settings = settings
        self.tracer = get_safe_tracer(tracer)

        # 📚 LESSON — Pointing ChatOpenAI to custom endpoints:
        # Since the NVIDIA API is OpenAI-compatible (uses exact same JSON payloads),
        # we do not need a custom SDK. We point ChatOpenAI directly to NVIDIA's base URL.
        logger.info(
            "Connecting to NVIDIA API model '{model}' at {url}...",
            model=settings.nvidia_model,
            url=settings.nvidia_base_url,
        )
        self.llm = ChatOpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
            model=settings.nvidia_model,
            temperature=0.1,  # Low temperature for highly grounded, factual Q&A
            max_tokens=1024,
        )

        self.local_prompt_path = "app/generation/prompts/single_turn.yaml"

    def _resolve_prompt(self) -> tuple[ChatPromptTemplate, str]:
        """
        Attempts to fetch the prompt template from the Langfuse Prompt Management registry.
        Falls back to the local YAML prompt file if Langfuse is offline or the prompt is missing.

        Returns:
            Tuple of:
              - ChatPromptTemplate: The compiled LangChain prompt template.
              - str: The version string of the prompt.
        """
        # 1. Try to fetch from Langfuse Prompt Registry
        if self.settings.langfuse_public_key and self.settings.langfuse_secret_key:
            try:
                client = get_langfuse_client()
                logger.info("Attempting to fetch 'rag-qna-prompt' from Langfuse Prompt Registry...")
                # Fetching prompt from Langfuse
                langfuse_prompt = client.get_prompt("rag-qna-prompt")
                
                # In Langfuse v4 SDK, the returned prompt object has a direct conversion
                # helper get_langchain_prompt() which compiles into a ChatPromptTemplate!
                prompt_tmpl = langfuse_prompt.get_langchain_prompt()
                version = str(langfuse_prompt.version)
                
                logger.info("Successfully loaded 'rag-qna-prompt' from Langfuse (Version: {v})", v=version)
                return prompt_tmpl, version
            except Exception as lf_err:
                logger.warning(
                    "Could not load prompt from Langfuse registry (non-fatal, falling back to local YAML): {err}",
                    err=str(lf_err),
                )

        # 2. Fallback to Local YAML Prompt Configuration
        logger.info("Loading local fallback prompt from '{path}'...", path=self.local_prompt_path)
        try:
            if not os.path.exists(self.local_prompt_path):
                raise FileNotFoundError(f"Local prompt config file '{self.local_prompt_path}' is missing.")

            with open(self.local_prompt_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # Compile into ChatPromptTemplate
            prompt_tmpl = ChatPromptTemplate.from_messages([
                ("system", data["system"]),
                ("human", data["user"]),
            ])
            version = data.get("version", "local-fallback")

            logger.info("Local fallback prompt loaded successfully (Version: {v})", v=version)
            return prompt_tmpl, version

        except Exception as e:
            logger.error("Failed to load prompt template: {err}", err=str(e))
            raise

    def generate(self, query: str, nodes: list[NodeWithScore]) -> GenerationResult:
        """
        Synthesizes an answer based strictly on retrieved context nodes.

        Args:
            query: The user search query.
            nodes: List of retrieved NodeWithScore objects.

        Returns:
            GenerationResult container.
        """
        # 1. Convert candidate nodes into a unified context string
        context_str = nodes_to_context_str(nodes)

        # 2. Fetch active Q&A system prompts
        prompt_tmpl, prompt_version = self._resolve_prompt()

        # 3. Create the LangChain chain
        chain = prompt_tmpl | self.llm | StrOutputParser()

        logger.info("Invoking LLM for single-turn RAG generation...")

        span_ctx = (
            self.tracer.span(
                name="generation.single_turn",
                input={
                    "query": query,
                    "prompt_version": prompt_version,
                    "nodes_count": len(nodes),
                },
                as_type="generation",
            )
            if self.tracer
            else None
        )

        try:
            # Setup LangChain observability callback
            callbacks = [get_langfuse_callback()] if self.tracer else []

            # Execute LLM chain
            answer = chain.invoke(
                {"context": context_str, "question": query},
                config={"callbacks": callbacks},
            )

            logger.info("LLM generation complete")

            if span_ctx:
                span_ctx.update(output={"answer": answer})

            return GenerationResult(
                answer=answer,
                source_nodes=nodes,
                prompt_version=prompt_version,
            )

        except Exception as e:
            logger.error("LLM generation failed: {err}", err=str(e))
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=str(e))
            raise
