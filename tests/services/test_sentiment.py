"""Unit tests for the sentiment analysis module"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from src.services.sentiment import SentimentAnalyzer, SentimentScore

@pytest.fixture
def mock_nlp():
    """Create a mock NLP pipeline"""
    mock = MagicMock()
    mock.return_value = [{'label': 'POS', 'score': 0.8}]
    return mock

@pytest.fixture
def sentiment_analyzer(mock_nlp):
    """Create a sentiment analyzer with mocked NLP pipeline"""
    with patch('src.services.sentiment.pipeline', return_value=mock_nlp):
        analyzer = SentimentAnalyzer()
        yield analyzer
        analyzer.cleanup()

@pytest.mark.asyncio
async def test_analyze_text_positive(sentiment_analyzer, mock_nlp):
    """Test analyzing positive sentiment text"""
    mock_nlp.return_value = [{'label': 'POS', 'score': 0.9}]
    
    result = await sentiment_analyzer.analyze_text("Great project!")
    assert result['score'] == 1.0  # POS maps to 1.0
    assert result['confidence'] == 0.9

@pytest.mark.asyncio
async def test_analyze_text_negative(sentiment_analyzer, mock_nlp):
    """Test analyzing negative sentiment text"""
    mock_nlp.return_value = [{'label': 'NEG', 'score': 0.8}]
    
    result = await sentiment_analyzer.analyze_text("Bad project!")
    assert result['score'] == -1.0  # NEG maps to -1.0
    assert result['confidence'] == 0.8

@pytest.mark.asyncio
async def test_analyze_text_neutral(sentiment_analyzer, mock_nlp):
    """Test analyzing neutral sentiment text"""
    mock_nlp.return_value = [{'label': 'NEU', 'score': 0.7}]
    
    result = await sentiment_analyzer.analyze_text("Project launched.")
    assert result['score'] == 0.0  # NEU maps to 0.0
    assert result['confidence'] == 0.7

@pytest.mark.asyncio
async def test_analyze_text_empty(sentiment_analyzer):
    """Test analyzing empty text"""
    result = await sentiment_analyzer.analyze_text("")
    assert result['score'] == 0.0
    assert result['confidence'] == 0.0

@pytest.mark.asyncio
async def test_analyze_text_error(sentiment_analyzer, mock_nlp):
    """Test error handling in text analysis"""
    mock_nlp.side_effect = Exception("NLP error")
    
    result = await sentiment_analyzer.analyze_text("Some text")
    assert result['score'] == 0.0
    assert result['confidence'] == 0.0

@pytest.mark.asyncio
async def test_analyze_messages(sentiment_analyzer, mock_nlp):
    """Test analyzing multiple messages"""
    mock_nlp.side_effect = [
        [{'label': 'POS', 'score': 0.9}],
        [{'label': 'NEG', 'score': 0.8}],
        [{'label': 'NEU', 'score': 0.7}]
    ]
    
    messages = [
        "Great project!",
        "Bad investment!",
        "Project launched."
    ]
    
    result = await sentiment_analyzer.analyze_messages(messages, 'telegram')
    assert 'score' in result
    assert 'confidence' in result
    # Should be average: (1.0 + -1.0 + 0.0) / 3 = 0.0
    assert abs(result['score']) < 0.01
    # Average confidence: (0.9 + 0.8 + 0.7) / 3 = 0.8
    assert abs(result['confidence'] - 0.8) < 0.01

@pytest.mark.asyncio
async def test_analyze_messages_empty(sentiment_analyzer):
    """Test analyzing empty message list"""
    result = await sentiment_analyzer.analyze_messages([], 'telegram')
    assert result['score'] == 0.0
    assert result['confidence'] == 0.0

@pytest.mark.asyncio
async def test_get_token_sentiment(sentiment_analyzer, mock_nlp):
    """Test getting overall token sentiment"""
    # Setup mock responses for different sources
    mock_nlp.side_effect = [
        [{'label': 'POS', 'score': 0.9}],  # Telegram message 1
        [{'label': 'POS', 'score': 0.8}],  # Telegram message 2
        [{'label': 'NEG', 'score': 0.7}],  # Discord message
    ]
    
    messages = {
        'telegram': ["Great project!", "Amazing team!"],
        'discord': ["Not convinced."]
    }
    
    result = await sentiment_analyzer.get_token_sentiment(
        "0x123...",
        messages
    )
    
    assert isinstance(result, SentimentScore)
    assert -1.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.sources.keys()) == {'telegram', 'discord'}

@pytest.mark.asyncio
async def test_get_token_sentiment_empty(sentiment_analyzer):
    """Test getting sentiment with no messages"""
    result = await sentiment_analyzer.get_token_sentiment(
        "0x123...",
        {}
    )
    
    assert isinstance(result, SentimentScore)
    assert result.score == 0.0
    assert result.confidence == 0.0
    assert result.sources == {}

@pytest.mark.asyncio
async def test_get_token_sentiment_single_source(sentiment_analyzer, mock_nlp):
    """Test getting sentiment from a single source"""
    mock_nlp.side_effect = [
        [{'label': 'POS', 'score': 0.9}],
        [{'label': 'POS', 'score': 0.8}]
    ]
    
    messages = {
        'telegram': ["Great project!", "Amazing team!"]
    }
    
    result = await sentiment_analyzer.get_token_sentiment(
        "0x123...",
        messages
    )
    
    assert isinstance(result, SentimentScore)
    assert result.score > 0  # Should be positive
    assert 'telegram' in result.sources
    assert len(result.sources) == 1

@pytest.mark.asyncio
async def test_sentiment_score_consistency(sentiment_analyzer, mock_nlp):
    """Test consistency of sentiment scores"""
    # Same text should get same score
    text = "Great project!"
    mock_nlp.return_value = [{'label': 'POS', 'score': 0.9}]
    
    result1 = await sentiment_analyzer.analyze_text(text)
    result2 = await sentiment_analyzer.analyze_text(text)
    
    assert result1['score'] == result2['score']
    assert result1['confidence'] == result2['confidence']

def test_cleanup(sentiment_analyzer):
    """Test cleanup of resources"""
    # Cleanup should not raise any errors
    sentiment_analyzer.cleanup()
    assert sentiment_analyzer._executor._shutdown 