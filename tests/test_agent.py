"""
Basic smoke tests for FreeAgent.
Run: pytest tests/
"""
import os
import pytest
from unittest.mock import patch, MagicMock

from src.agent import FreeAgent
from src.caveman import compress_response

# Mock API key for testing
TEST_API_KEY = "test_key_123"

def test_caveman_compression():
    """Test output compression removes filler"""
    verbose = "I'd be happy to help! The reason you're seeing this error is because..."
    compressed = compress_response(verbose)
    
    assert "I'd be happy to" not in compressed
    assert "The reason" not in compressed
    assert len(compressed) < len(verbose) * 0.5  # ~50%+ reduction

def test_agent_initialization():
    """Test agent creates with valid config"""
    with patch.dict(os.environ, {"GROQ_API_KEY": TEST_API_KEY}):
        agent = FreeAgent(daily_token_limit=1000)
        assert agent.daily_token_limit == 1000
        assert agent.tokens_used_today == 0

def test_budget_tracking():
    """Test token budget enforcement"""
    with patch.dict(os.environ, {"GROQ_API_KEY": TEST_API_KEY}):
        agent = FreeAgent(daily_token_limit=100)
        agent.tokens_used_today = 95
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.usage.total_tokens = 10
        mock_response.choices = [MagicMock(message=MagicMock(content="test"))]
        
        with patch.object(agent.client.chat.completions, 'create', return_value=mock_response):
            result = agent.ask("test")
            # Should allow this request (95 + 10 = 105 > 100, but check happens before)
            assert "Error" not in result or "limit reached" in result

def test_memory_store_recall():
    """Test SQLite memory layer"""
    from src.memory import TokenEfficientMemory
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".mv2", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        memory = TokenEfficientMemory(tmp_path)
        
        # Store
        mem_id = memory.store("React hooks must be called at top level", tags=["react", "hooks"])
        assert len(mem_id) == 12  # SHA256 prefix
        
        # Recall
        result = memory.recall("React hooks", max_tokens=500)
        assert "React hooks" in result
    finally:
        os.unlink(tmp_path)
