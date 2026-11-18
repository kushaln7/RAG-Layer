"""End-to-End RAG Generation Pipeline.

Orchestrates user query retrieval, ranked context formatting, prompt construction,
and LLM completion generation with DeepSeek and pluggable backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import settings
from generation.llm_factory import LLMGenerationError, get_llm
from prompts.rag_prompt import RAG_PROMPT
from query.query_schemas import QueryContext
from retrieval.ranking_pipeline import RankedContext, RankingOptions

if TYPE_CHECKING:
    from services.query_service import QueryService

logger = logging.getLogger(__name__)


FALLBACK_NO_CONTEXT_ANSWER = "I couldn't find this information in the available knowledge base."


@dataclass
class RAGResponse:
    """Structured response container returned by RAGPipeline.

    Attributes:
        answer: Generated textual answer from the LLM.
        query: Original user query string.
        context_text: Retained and assembled context passed into the prompt.
        sources: List of unique source document names cited in the context.
        results_count: Number of retained document chunks used in generation.
        ranked_context: Full RankedContext diagnostic object (optional).
        provider: Name of the LLM provider used (e.g. 'DeepSeek').
        model_name: Name of the LLM model used (e.g. 'deepseek-flash').
        duration_seconds: End-to-end pipeline processing latency in seconds.
        metadata: Additional diagnostic or token consumption metadata.
    """

    answer: str
    query: str
    context_text: str = ""
    sources: list[str] = field(default_factory=list)
    results_count: int = 0
    ranked_context: RankedContext | None = None
    provider: str = ""
    model_name: str = ""
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """Coordinates retrieval, ranking, context formatting, and LLM answer generation."""

    def __init__(
        self,
        query_service: QueryService | None = None,
        llm: BaseChatModel | None = None,
        prompt_template: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize RAGPipeline with optional dependencies.

        Args:
            query_service: QueryService instance. Lazily loaded if None.
            llm: Configured BaseChatModel instance. Lazily loaded via get_llm() if None.
            prompt_template: Custom prompt template string. Defaults to RAG_PROMPT.
            provider: LLM provider name override. Defaults to settings.LLM_PROVIDER.
            model: LLM model name override. Defaults to settings.LLM_MODEL.
        """
        self._query_service = query_service
        self._llm = llm
        self.prompt_template = prompt_template or RAG_PROMPT
        self.provider = provider or getattr(settings, "LLM_PROVIDER", "DeepSeek")
        self.model_name = model or getattr(settings, "LLM_MODEL", "deepseek-flash")

    @property
    def query_service(self) -> QueryService:
        """Return or lazily instantiate QueryService."""
        if self._query_service is None:
            from services.query_service import query_service
            self._query_service = query_service
        return self._query_service

    @property
    def llm(self) -> BaseChatModel:
        """Return or lazily instantiate LLM client."""
        if self._llm is None:
            self._llm = get_llm(provider=self.provider, model=self.model_name)
        return self._llm

    def format_prompt(self, question: str, context: str) -> str:
        """Assemble the complete prompt string using prompt template.

        Args:
            question: Cleaned user question string.
            context: Formatted context blocks from retrieved chunks.

        Returns:
            Fully rendered prompt ready for LLM consumption.
        """
        return self.prompt_template.format(
            question=question.strip(),
            context=context.strip() if context else "No context available.",
        )

    def generate_answer(
        self,
        query: str,
        context: QueryContext | None = None,
        ranking_options: RankingOptions | None = None,
    ) -> RAGResponse:
        """Execute end-to-end RAG: retrieve, rank, construct prompt, and generate answer.

        Args:
            query: Raw user query string.
            context: Optional QueryContext configuring roles, overrides, and filters.
            ranking_options: Optional RankingOptions configuring top_k and thresholds.

        Returns:
            RAGResponse containing generated answer, cited sources, and diagnostics.
        """
        start_time = time.perf_counter()
        logger.info("Starting RAG generation pipeline for query: '%s'", query)

        # 1. Retrieve and rank relevant context chunks
        ranked_context: RankedContext = self.query_service.get_ranked_context(
            query=query,
            context=context,
            options=ranking_options,
        )

        # 2. Guard: If no relevant chunks retrieved, return standard no-context response
        if not ranked_context.results or not ranked_context.context_text.strip():
            logger.info("No context chunks retrieved for query '%s'. Returning fallback answer.", query)
            duration = time.perf_counter() - start_time
            return RAGResponse(
                answer=FALLBACK_NO_CONTEXT_ANSWER,
                query=query,
                context_text="",
                sources=[],
                results_count=0,
                ranked_context=ranked_context,
                provider=self.provider,
                model_name=self.model_name,
                duration_seconds=duration,
                metadata={"status": "no_context_found"},
            )

        # 3. Format full prompt with retrieved context
        rendered_prompt = self.format_prompt(
            question=query,
            context=ranked_context.context_text,
        )

        # 4. Invoke LLM for answer generation
        try:
            logger.info("Invoking LLM (%s / %s) with %d context chars", self.provider, self.model_name, len(ranked_context.context_text))
            llm_result = self.llm.invoke(rendered_prompt)
            answer_text = str(llm_result.content).strip()
        except Exception as exc:
            logger.error("LLM generation failed for query '%s': %s", query, exc, exc_info=True)
            raise LLMGenerationError(f"LLM generation failed: {exc}") from exc

        duration = time.perf_counter() - start_time
        logger.info("RAG generation completed in %.3fs with %d sources cited", duration, len(ranked_context.sources_cited))

        return RAGResponse(
            answer=answer_text,
            query=query,
            context_text=ranked_context.context_text,
            sources=ranked_context.sources_cited,
            results_count=len(ranked_context.results),
            ranked_context=ranked_context,
            provider=self.provider,
            model_name=self.model_name,
            duration_seconds=duration,
            metadata={
                "chars_used": ranked_context.total_chars_used,
                "remaining_budget": ranked_context.remaining_budget,
                "prompt_length": len(rendered_prompt),
            },
        )

    def answer_with_context(
        self,
        query: str,
        context_text: str,
        sources: list[str] | None = None,
    ) -> RAGResponse:
        """Generate answer directly from provided context string without vector retrieval.

        Args:
            query: User query string.
            context_text: Formatted context string.
            sources: Optional explicit list of source names.

        Returns:
            RAGResponse containing generated answer.
        """
        start_time = time.perf_counter()

        if not context_text or not context_text.strip():
            duration = time.perf_counter() - start_time
            return RAGResponse(
                answer=FALLBACK_NO_CONTEXT_ANSWER,
                query=query,
                context_text="",
                sources=[],
                results_count=0,
                provider=self.provider,
                model_name=self.model_name,
                duration_seconds=duration,
            )

        rendered_prompt = self.format_prompt(question=query, context=context_text)

        try:
            llm_result = self.llm.invoke(rendered_prompt)
            answer_text = str(llm_result.content).strip()
        except Exception as exc:
            raise LLMGenerationError(f"LLM generation failed: {exc}") from exc

        duration = time.perf_counter() - start_time

        return RAGResponse(
            answer=answer_text,
            query=query,
            context_text=context_text,
            sources=sources or [],
            results_count=1,
            provider=self.provider,
            model_name=self.model_name,
            duration_seconds=duration,
        )
