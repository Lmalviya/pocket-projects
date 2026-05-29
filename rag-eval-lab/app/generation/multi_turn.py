"""
app/generation/multi_turn.py
=============================
Multi-turn stateful conversational RAG generator using LangGraph and Langfuse.

📚 LESSON — Stateful Conversation Graphs (LangGraph):
Unlike single-turn systems, conversational RAG must keep track of memory. 
We model this as a state transition system using LangGraph:
  1. We define a `ConversationState` TypedDict containing the conversation history
     `messages` (which automatically appends new messages using LangGraph's `add_messages` reducer)
     and the retrieved document `context` for the current turn.
  2. We register a `generator` node inside our StateGraph.
  3. The node retrieves the system prompt, formats the retrieved context into it,
     combines it with the conversation history list, and invokes the NVIDIA LLM.
  4. The graph compiles into an executable state machine, managing conversation memory
     and context injection cleanly and robustly.
"""

import os
from typing import Annotated, Any, TypedDict

import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from llama_index.core.schema import NodeWithScore

from app.config.settings import Settings
from app.generation.single_turn import GenerationResult, CompatibleChatOpenAI
from app.tracing.langfuse import LangfuseTracer, get_langfuse_callback, get_langfuse_client, get_safe_tracer
from app.utils.logger import get_logger
from app.utils.text import nodes_to_context_str

logger = get_logger(__name__)


class ConversationState(TypedDict):
    """
    📚 LESSON — LangGraph State Schema:
    The state schema represents the memory of our graph.
    - `messages` carries the conversation history. The `Annotated[..., add_messages]`
      metadata tells LangGraph's compiler to use an "append/upsert" reducer:
      whenever a node returns `{"messages": [new_msg]}`, it appends it to the history
      rather than overwriting it.
    - `context` holds the retrieved text snippet context for the current turn.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    context: str


class MultiTurnGenerator:
    """
    Orchestrates stateful conversational RAG using a LangGraph workflow.
    """

    def __init__(self, settings: Settings, tracer: LangfuseTracer | None = None) -> None:
        """
        Initialize the multi-turn generator.

        Args:
            settings: Central Settings instance.
            tracer: Optional LangfuseTracer instance.
        """
        self.settings = settings
        self.tracer = get_safe_tracer(tracer)

        # 1. Connect to NVIDIA API (OpenAI-compatible)
        logger.info(
            "Connecting to NVIDIA API model '{model}' for multi-turn conversations...",
            model=settings.nvidia_model,
        )
        self.llm = CompatibleChatOpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
            model=settings.nvidia_model,
            temperature=0.1,  # Low temperature for factual conversational RAG
            max_tokens=1024,
        )

        self.local_prompt_path = "app/generation/prompts/multi_turn.yaml"
        self.prompt_version = "unknown"

        # 2. Compile the LangGraph state machine workflow
        workflow = StateGraph(ConversationState)
        workflow.add_node("generator", self._generation_node)
        
        # Setup graph routing: START ──> generator ──> END
        workflow.add_edge(START, "generator")
        workflow.add_edge("generator", END)
        
        self.graph = workflow.compile()
        logger.info("LangGraph multi-turn conversation graph successfully compiled.")

    def _resolve_prompt_templates(self) -> tuple[str, str, str]:
        """
        Loads the system and user templates from the Langfuse Prompt registry,
        falling back to local YAML config.

        Returns:
            Tuple of:
              - str: System prompt template string.
              - str: User prompt template string.
              - str: Active prompt version.
        """
        # 1. Try to fetch from Langfuse Prompt Registry
        if self.settings.langfuse_public_key and self.settings.langfuse_secret_key:
            try:
                client = get_langfuse_client()
                logger.info("Attempting to fetch 'rag-conversation-prompt' from Langfuse Registry...")
                langfuse_prompt = client.get_prompt("rag-conversation-prompt", label="production")
                
                # In Langfuse prompt registries, we can define custom variables.
                # We pull raw system and user strings from prompt templates.
                # (Often structured as a Chat template in Langfuse).
                # If get_langchain_prompt is not used directly, we can read raw config:
                system_tmpl = langfuse_prompt.prompt
                user_tmpl = "{question}"  # Simple user payload pass
                version = str(langfuse_prompt.version)
                
                logger.info("Loaded 'rag-conversation-prompt' from Langfuse (Version: {v})", v=version)
                return system_tmpl, user_tmpl, version
            except Exception as lf_err:
                logger.warning(
                    "Could not load conversation prompt from Langfuse (falling back to local YAML): {err}",
                    err=str(lf_err),
                )

        # 2. Fallback to Local YAML Prompt Configuration
        logger.info("Loading local conversational prompt from '{path}'...", path=self.local_prompt_path)
        try:
            if not os.path.exists(self.local_prompt_path):
                raise FileNotFoundError(f"Local prompt config file '{self.local_prompt_path}' is missing.")

            with open(self.local_prompt_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            system_tmpl = data["system"]
            user_tmpl = data["user"]
            version = data.get("version", "local-fallback")

            logger.info("Local fallback prompt loaded successfully (Version: {v})", v=version)
            return system_tmpl, user_tmpl, version

        except Exception as e:
            logger.error("Failed to load conversational prompt: {err}", err=str(e))
            raise

    def _generation_node(self, state: ConversationState, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        LangGraph node execution: compiles the conversational context,
        incorporates chat history, and invokes the NVIDIA LLM.
        """
        # 1. Resolve active conversational prompt templates
        system_tmpl, _, version = self._resolve_prompt_templates()
        self.prompt_version = version

        # 2. Format retrieved context into the system prompt
        system_content = system_tmpl.format(context=state["context"])
        system_msg = SystemMessage(content=system_content)

        # 3. Assemble full LLM chat message payloads: System Prompt + Chat History
        # We append our custom RAG system instructions to the front of the conversation messages.
        chat_messages = [system_msg] + state["messages"]

        logger.info("Invoking LLM inside conversational LangGraph generator node...")
        # Invoke LLM
        ai_response = self.llm.invoke(chat_messages, config=config)

        # Return updated state: the AI response will be appended to 'messages' by LangGraph
        return {"messages": [ai_response]}

    def generate_turn(
        self,
        query: str,
        nodes: list[NodeWithScore],
        history: list[BaseMessage],
    ) -> GenerationResult:
        """
        Executes a single turn of the conversational RAG graph.

        Args:
            query: Latest user query.
            nodes: Newly retrieved document chunks for this turn.
            history: Accumulation of past messages in the conversation session.

        Returns:
            GenerationResult container.
        """
        # 1. Compile retrieved nodes for this turn
        context_str = nodes_to_context_str(nodes)

        # 2. Convert query to LangChain HumanMessage and construct initial state
        new_messages = history + [HumanMessage(content=query)]
        initial_state = {
            "messages": new_messages,
            "context": context_str,
        }

        logger.info("Executing conversational RAG LangGraph turn...")

        span_ctx = (
            self.tracer.span(
                name="generation.multi_turn",
                input={
                    "query": query,
                    "history_turns": len(history) // 2,  # Approximate number of Q&A exchanges
                    "nodes_count": len(nodes),
                },
                as_type="generation",
            )
            if self.tracer
            else None
        )

        try:
            # Set up callbacks
            callbacks = [get_langfuse_callback()] if self.tracer else []
            config = {"callbacks": callbacks}

            # 3. Run the compiled LangGraph workflow state machine
            final_state = self.graph.invoke(initial_state, config=config)

            # 4. Extract generated AI response (which is the last message in the history)
            ai_message = final_state["messages"][-1]
            answer = ai_message.content

            logger.info("Conversational RAG turn completed successfully")

            if span_ctx:
                span_ctx.update(output={"answer": answer, "history_size": len(final_state["messages"])})

            return GenerationResult(
                answer=answer,
                source_nodes=nodes,
                prompt_version=self.prompt_version,
            )

        except Exception as e:
            logger.error("Stateful conversational RAG generation failed: {err}", err=str(e))
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=str(e))
            raise
