import pytest
import asyncio
from unittest.mock import patch, MagicMock
from src.monitoring.notification_manager import NotificationManager

@pytest.fixture
def notification_manager():
    return NotificationManager(
        bot_token="test_token",
        chat_id="test_chat_id"
    )

@pytest.mark.asyncio
async def test_format_message_normal_priority(notification_manager):
    """Test message formatting with normal priority"""
    message = "Test message"
    formatted = notification_manager._format_message(
        message=message,
        priority="normal"
    )
    
    assert "ℹ️" in formatted
    assert "[NORMAL]" in formatted
    assert message in formatted

@pytest.mark.asyncio
async def test_format_message_with_metadata(notification_manager):
    """Test message formatting with metadata"""
    message = "Test message"
    metadata = {
        "error_code": "500",
        "component": "api"
    }
    
    formatted = notification_manager._format_message(
        message=message,
        priority="critical",
        metadata=metadata
    )
    
    assert "🚨" in formatted
    assert "[CRITICAL]" in formatted
    assert message in formatted
    assert "Additional Info" in formatted
    assert "error_code" in formatted
    assert "500" in formatted
    assert "component" in formatted
    assert "api" in formatted

@pytest.mark.asyncio
async def test_get_priority_emoji():
    """Test priority emoji mapping"""
    manager = NotificationManager("token", "chat_id")
    
    assert manager._get_priority_emoji("critical") == "🚨"
    assert manager._get_priority_emoji("high") == "⚠️"
    assert manager._get_priority_emoji("normal") == "ℹ️"
    assert manager._get_priority_emoji("low") == "💡"
    assert manager._get_priority_emoji("info") == "📝"
    assert manager._get_priority_emoji("unknown") == "ℹ️"  # Default emoji

@pytest.mark.asyncio
async def test_send_notification_success(notification_manager):
    """Test successful notification sending"""
    mock_response = MagicMock()
    mock_response.status = 200
    
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_response
        
        await notification_manager.send_notification(
            message="Test alert",
            priority="high",
            metadata={"test": "value"}
        )
        
        # Verify API call
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        assert "json" in call_kwargs
        
        # Verify payload
        payload = call_kwargs["json"]
        assert payload["chat_id"] == "test_chat_id"
        assert "Test alert" in payload["text"]
        assert "⚠️" in payload["text"]  # High priority emoji
        assert "test" in payload["text"]
        assert "value" in payload["text"]

@pytest.mark.asyncio
async def test_send_notification_api_error(notification_manager):
    """Test handling of API errors"""
    mock_response = MagicMock()
    mock_response.status = 400
    
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Should not raise exception but log error
        await notification_manager.send_notification(
            message="Test alert",
            priority="normal"
        )
        
        # Verify error metric was incremented
        assert notification_manager.notification_errors._value == 1

@pytest.mark.asyncio
async def test_notification_metrics(notification_manager):
    """Test metric recording"""
    mock_response = MagicMock()
    mock_response.status = 200
    
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_response
        
        await notification_manager.send_notification(
            message="Test alert",
            priority="critical"
        )
        
        # Verify metrics
        assert notification_manager.notifications_sent.labels(
            priority="critical"
        )._value == 1
        assert notification_manager.notification_latency._value > 0

@pytest.mark.asyncio
async def test_concurrent_notifications(notification_manager):
    """Test sending multiple notifications concurrently"""
    mock_response = MagicMock()
    mock_response.status = 200
    
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Send multiple notifications concurrently
        tasks = [
            notification_manager.send_notification(
                message=f"Test {i}",
                priority="normal"
            )
            for i in range(5)
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify all notifications were sent
        assert mock_post.call_count == 5
        assert notification_manager.notifications_sent.labels(
            priority="normal"
        )._value == 5 