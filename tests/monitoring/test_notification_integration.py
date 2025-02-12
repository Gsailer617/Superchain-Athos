import pytest
import asyncio
import os
from datetime import datetime
from src.monitoring.notification_manager import NotificationManager

# Get test configuration from environment variables
TEST_BOT_TOKEN = os.getenv("TEST_BOT_TOKEN")
TEST_CHAT_ID = os.getenv("TEST_CHAT_ID")

# Skip tests if credentials not configured
requires_telegram = pytest.mark.skipif(
    not (TEST_BOT_TOKEN and TEST_CHAT_ID),
    reason="Telegram credentials not configured"
)

@pytest.fixture
async def notification_manager():
    """Create notification manager with test credentials"""
    if not (TEST_BOT_TOKEN and TEST_CHAT_ID):
        pytest.skip("Telegram credentials not configured")
        
    manager = NotificationManager(
        bot_token=TEST_BOT_TOKEN,
        chat_id=TEST_CHAT_ID
    )
    return manager

@requires_telegram
@pytest.mark.asyncio
async def test_basic_notification(notification_manager):
    """Test sending a basic notification"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"Integration Test Notification [{timestamp}]"
    
    await notification_manager.send_notification(
        message=message,
        priority="normal"
    )
    
    # Verify metrics
    assert notification_manager.notifications_sent.labels(
        priority="normal"
    )._value == 1
    assert notification_manager.notification_latency._value > 0

@requires_telegram
@pytest.mark.asyncio
async def test_priority_notifications(notification_manager):
    """Test notifications with different priorities"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    priorities = ["critical", "high", "normal", "low", "info"]
    for priority in priorities:
        message = f"{priority.title()} Priority Test [{timestamp}]"
        await notification_manager.send_notification(
            message=message,
            priority=priority
        )
        
        # Small delay to prevent rate limiting
        await asyncio.sleep(1)
    
    # Verify all notifications were sent
    for priority in priorities:
        assert notification_manager.notifications_sent.labels(
            priority=priority
        )._value == 1

@requires_telegram
@pytest.mark.asyncio
async def test_notification_with_metadata(notification_manager):
    """Test notification with detailed metadata"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    metadata = {
        "test_id": "integration_001",
        "component": "notification_system",
        "timestamp": timestamp,
        "details": "Testing metadata formatting"
    }
    
    await notification_manager.send_notification(
        message="Metadata Test Notification",
        priority="info",
        metadata=metadata
    )
    
    assert notification_manager.notifications_sent.labels(
        priority="info"
    )._value == 1

@requires_telegram
@pytest.mark.asyncio
async def test_notification_rate_handling(notification_manager):
    """Test handling of rapid notification sending"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Send multiple notifications rapidly
    messages = [
        f"Rate Test {i} [{timestamp}]"
        for i in range(5)
    ]
    
    tasks = [
        notification_manager.send_notification(
            message=msg,
            priority="normal"
        )
        for msg in messages
    ]
    
    # Send all notifications concurrently
    await asyncio.gather(*tasks)
    
    # Verify all notifications were sent
    assert notification_manager.notifications_sent.labels(
        priority="normal"
    )._value == 5
    
    # Check for any rate limit errors
    assert notification_manager.notification_errors._value == 0

@requires_telegram
@pytest.mark.asyncio
async def test_long_message_handling(notification_manager):
    """Test handling of long messages"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a long message
    long_message = "Long Message Test\n" + "-" * 100 + "\n"
    long_message += "\n".join([f"Line {i}" for i in range(20)])
    long_message += f"\nTimestamp: {timestamp}"
    
    metadata = {
        "test_type": "long_message",
        "length": len(long_message)
    }
    
    await notification_manager.send_notification(
        message=long_message,
        priority="normal",
        metadata=metadata
    )
    
    assert notification_manager.notifications_sent.labels(
        priority="normal"
    )._value == 1
    assert notification_manager.notification_errors._value == 0 