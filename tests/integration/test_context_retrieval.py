"""Integration test for context query workflow."""

import pytest
from repo_kgraph.services.query_processor import QueryProcessor
from repo_kgraph.services.retriever import ContextRetriever

@pytest.mark.integration
async def test_full_context_retrieval_workflow(sample_repository, sample_code_entities):
    """Test complete context retrieval from query to results."""
    # This will fail until we implement the services
    
    query_processor = QueryProcessor()
    retriever = ContextRetriever()
    
    # Process a natural language query
    query_result = await query_processor.process_query(
        repository_id=sample_repository.id,
        task_description="Add authentication middleware to Express API",
        max_results=10
    )
    
    assert query_result is not None
    assert len(query_result.results) > 0
    assert query_result.confidence_score > 0.0
    
    # Results should be ranked by relevance
    scores = [r.relevance_score for r in query_result.results]
    assert scores == sorted(scores, reverse=True)

@pytest.mark.integration
async def test_hybrid_search_workflow():
    """Test hybrid search combining semantic and graph traversal."""
    retriever = ContextRetriever()
    
    results = await retriever.hybrid_search(
        query_embedding=[0.1] * 384,
        graph_context={"entity_types": ["FUNCTION", "CLASS"]},
        max_results=20
    )
    
    assert isinstance(results, list)
    # Will fail until implemented

@pytest.mark.integration
async def test_multilanguage_context_retrieval():
    """Test context retrieval across multiple languages."""
    # Will be implemented after services exist
    pass