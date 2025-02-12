import aiohttp
import json
from typing import Dict, Optional
import structlog
from prometheus_client import Counter, Gauge
import asyncio

logger = structlog.get_logger(__name__)

class NotificationManager:
    def __init__(
        self,
        bot_token: str,
        chat_id: str
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        
        # Notification metrics
        self.notifications_sent = Counter(
            'telegram_notifications_sent_total',
            'Number of notifications sent via Telegram',
            ['priority']
        )
        
        self.notification_errors = Counter(
            'telegram_notification_errors_total',
            'Number of Telegram notification errors',
            ['error_type']
        )
        
        self.notification_latency = Gauge(
            'telegram_notification_latency_seconds',
            'Latency of Telegram notification delivery'
        )

    async def send_notification(
        self,
        message: str,
        priority: str = "normal",
        metadata: Optional[Dict] = None
    ):
        """Send notification via Telegram"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Format message with metadata
            formatted_message = self._format_message(message, priority, metadata)
            
            # Send message
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": formatted_message,
                "parse_mode": "HTML"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        raise Exception(f"Telegram API error: {response.status}")
            
            # Record metrics
            end_time = asyncio.get_event_loop().time()
            self.notification_latency.set(end_time - start_time)
            self.notifications_sent.labels(priority=priority).inc()
            
        except Exception as e:
            logger.error("Error sending Telegram notification",
                        error=str(e))
            self.notification_errors.labels(
                error_type=type(e).__name__
            ).inc()

    def _format_message(
        self,
        message: str,
        priority: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Format notification message with priority and metadata"""
        # Add priority emoji
        priority_emoji = self._get_priority_emoji(priority)
        formatted_message = f"{priority_emoji} <b>[{priority.upper()}]</b>\n\n{message}"
        
        # Add metadata if provided
        if metadata:
            formatted_message += "\n\n<b>Additional Info:</b>"
            for key, value in metadata.items():
                formatted_message += f"\n• <b>{key}:</b> {value}"
        
        return formatted_message

    def _get_priority_emoji(self, priority: str) -> str:
        """Get emoji for priority level"""
        emojis = {
            "critical": "🚨",
            "high": "⚠️",
            "normal": "ℹ️",
            "low": "💡",
            "info": "📝"
        }
        return emojis.get(priority.lower(), "ℹ️") 