"""Unit and Integration Tests for RAG Generation Pipeline and LLM Service."""

import os
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
import pytest

from config.settings import settings
from generation.llm_factory import (
    LLMConfigurationError,
    LLMGenerationError,
    clear_llm_cache,
    get_llm,
)
from generation.rag_pipeline import (
    FALLBACK_NO_CONTEXT_ANSWER,
    RAGPipeline,
    RAGResponse,
)
from prompts.rag_prompt import RAG_PROMPT
from query.query_schemas import QueryContext
from retrieval.ranking_pipeline import RankedContext, RankingOptions
from retrieval.retrieval_schemas import SearchResult
from services.query_service import QueryService
from services.rag_service import RAGService, rag_service


@pytest.fixture(autouse=True)
def reset_cache() -> None:
    """Clear LLM cache before each test."""
    clear_llm_cache()


def test_1_llm_factory_returns_fake_model() -> None:
    """1. Test that get_llm with provider='fake' returns a mock model for tests."""
    llm = get_llm(provider="fake", use_cache=False)
    assert isinstance(llm, FakeListChatModel)
    resp = llm.invoke("Test query")
    assert resp.content is not None


def test_2_llm_factory_returns_deepseek_chat_openai() -> None:
    """2. Test that get_llm with provider='DeepSeek' returns a configured ChatOpenAI instance."""
    from langchain_openai import ChatOpenAI

    llm = get_llm(
        provider="DeepSeek",
        model="deepseek-flash",
        api_key="sk-mock-key-12345",
        base_url="https://api.deepseek.com/",
        use_cache=False,
    )
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "deepseek-flash"
    assert "deepseek.com" in str(llm.openai_api_base)


def test_3_llm_factory_missing_api_key_raises_error() -> None:
    """3. Test that missing DeepSeek API key raises LLMConfigurationError."""
    with pytest.raises(LLMConfigurationError, match="DeepSeek API key is missing"):
        get_llm(provider="DeepSeek", api_key="", use_cache=False)


def test_4_llm_factory_unsupported_provider_raises_error() -> None:
    """4. Test that unsupported provider raises LLMConfigurationError."""
    with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider"):
        get_llm(provider="invalid_provider_xyz", api_key="test", use_cache=False)


def test_5_prompt_formatting_structure() -> None:
    """5. Test that format_prompt properly injects context and question into RAG_PROMPT."""
    pipeline = RAGPipeline(llm=FakeListChatModel(responses=["Test response"]))
    question = "What is the password policy?"
    context = "Doc 1: Passwords must be at least 12 characters."

    formatted = pipeline.format_prompt(question=question, context=context)

    assert "Retrieved Context" in formatted
    assert context in formatted
    assert "User Question" in formatted
    assert question in formatted
    assert "Canonix AI" in formatted


def test_6_empty_context_returns_standard_fallback() -> None:
    """6. Test that empty context immediately returns the standard fallback message."""
    mock_qs = MagicMock(spec=QueryService)
    mock_qs.get_ranked_context.return_value = RankedContext(
        results=[],
        context_text="",
        total_candidates=0,
        retained_count=0,
    )

    pipeline = RAGPipeline(
        query_service=mock_qs,
        llm=FakeListChatModel(responses=["Unexpected LLM response"]),
    )

    response = pipeline.generate_answer(query="Non-existent topic")

    assert response.answer == FALLBACK_NO_CONTEXT_ANSWER
    assert response.results_count == 0
    assert response.sources == []
    assert response.metadata.get("status") == "no_context_found"


def test_7_full_rag_pipeline_with_mock_llm() -> None:
    """7. Test end-to-end RAG generation flow with mocked retrieval and fake LLM."""
    doc = Document(
        page_content="Canonix data retention policy requires 7 years archival for financial logs.",
        metadata={"source_name": "retention_policy.pdf", "chunk_id": "c1", "category": "policy"},
    )
    sr = SearchResult(document=doc, score=0.95)

    mock_qs = MagicMock(spec=QueryService)
    mock_qs.get_ranked_context.return_value = RankedContext(
        results=[sr],
        context_text="--- Document 1: retention_policy.pdf ---\nCanonix data retention policy requires 7 years archival.",
        total_candidates=1,
        retained_count=1,
        sources_cited=["retention_policy.pdf"],
    )

    fake_llm = FakeListChatModel(
        responses=["Financial logs must be archived for 7 years according to retention_policy.pdf."]
    )

    pipeline = RAGPipeline(
        query_service=mock_qs,
        llm=fake_llm,
        provider="DeepSeek",
        model="deepseek-flash",
    )

    resp = pipeline.generate_answer("How long must financial logs be kept?")

    assert "7 years" in resp.answer
    assert resp.sources == ["retention_policy.pdf"]
    assert resp.results_count == 1
    assert resp.provider == "DeepSeek"
    assert resp.model_name == "deepseek-flash"
    assert resp.duration_seconds >= 0.0


def test_8_answer_with_context_direct_generation() -> None:
    """8. Test answer_with_context generates answers directly without vector search."""
    fake_llm = FakeListChatModel(
        responses=["Encryption at rest is required for all customer databases."]
    )
    svc = RAGService(rag_pipeline=RAGPipeline(llm=fake_llm))

    res = svc.answer_with_context(
        query="What encryption is required?",
        context_text="Database security: Encryption at rest is mandatory.",
        sources=["security_spec.docx"],
    )

    assert "Encryption at rest" in res.answer
    assert res.sources == ["security_spec.docx"]


def test_9_query_service_answer_query_delegates_to_rag_service() -> None:
    """9. Test that QueryService.answer_query delegates to RAGService."""
    qs = QueryService()

    with patch.object(rag_service, "answer_query") as mock_rag:
        mock_rag.return_value = RAGResponse(
            answer="Mocked RAG response",
            query="Test query",
            sources=["test.txt"],
        )
        res = qs.answer_query("Test query")

        assert res.answer == "Mocked RAG response"
        mock_rag.assert_called_once_with(query="Test query", context=None, ranking_options=None)


def test_10_llm_invocation_failure_raises_llm_generation_error() -> None:
    """10. Test that LLM exceptions are caught and wrapped in LLMGenerationError."""
    doc = Document(page_content="Valid context", metadata={"source_name": "doc.pdf"})
    mock_qs = MagicMock(spec=QueryService)
    mock_qs.get_ranked_context.return_value = RankedContext(
        results=[SearchResult(document=doc, score=0.9)],
        context_text="Valid context",
        total_candidates=1,
        retained_count=1,
    )

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("API connection timeout")

    pipeline = RAGPipeline(query_service=mock_qs, llm=mock_llm)

    with pytest.raises(LLMGenerationError, match="LLM generation failed"):
        pipeline.generate_answer("Trigger failure")

