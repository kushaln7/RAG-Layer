"""RAG Service coordinating end-to-end question answering and LLM response generation."""

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from generation.rag_pipeline import RAGPipeline, RAGResponse
from query.query_schemas import QueryContext
from retrieval.ranking_pipeline import RankingOptions
from services.query_service import QueryService

logger = logging.getLogger(__name__)


class RAGService:
    """High-level service orchestrating retrieval and LLM response generation."""

    def __init__(
        self,
        rag_pipeline: RAGPipeline | None = None,
        query_service: QueryService | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        """Initialize RAGService with optional injected pipeline or dependencies."""
        self._rag_pipeline = rag_pipeline
        self._query_service = query_service
        self._llm = llm

    @property
    def pipeline(self) -> RAGPipeline:
        """Return or lazily instantiate RAGPipeline."""
        if self._rag_pipeline is None:
            self._rag_pipeline = RAGPipeline(
                query_service=self._query_service,
                llm=self._llm,
            )
        return self._rag_pipeline

    def answer_query(
        self,
        query: str,
        context: QueryContext | None = None,
        ranking_options: RankingOptions | None = None,
    ) -> RAGResponse:
        """Execute complete RAG pipeline for a user question.

        Args:
            query: User's question or search query.
            context: Optional QueryContext configuring roles, overrides, and limits.
            ranking_options: Optional RankingOptions for filtering and deduplication.

        Returns:
            RAGResponse with generated answer, citations, and pipeline diagnostics.
        """
        return self.pipeline.generate_answer(
            query=query,
            context=context,
            ranking_options=ranking_options,
        )

    def answer_with_context(
        self,
        query: str,
        context_text: str,
        sources: list[str] | None = None,
    ) -> RAGResponse:
        """Generate answer directly from provided context string.

        Args:
            query: User question string.
            context_text: Pre-formatted context text.
            sources: Optional source document citations.

        Returns:
            RAGResponse containing generated answer.
        """
        return self.pipeline.answer_with_context(
            query=query,
            context_text=context_text,
            sources=sources,
        )


# Module-level singleton instance
rag_service = RAGService()

