import pytest
import asyncio
import time
import statistics
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

class NotificationBenchmark:
    """Class to track benchmark metrics"""
    def __init__(self):
        self.latencies = []
        self.start_time = None
        self.end_time = None
        self.total_notifications = 0
        self.failed_notifications = 0
        
    @property
    def duration(self):
        """Total duration of the benchmark in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
        
    @property
    def throughput(self):
        """Notifications per second"""
        if self.duration > 0:
            return self.total_notifications / self.duration
        return 0
        
    @property
    def success_rate(self):
        """Percentage of successful notifications"""
        if self.total_notifications > 0:
            return ((self.total_notifications - self.failed_notifications) 
                   / self.total_notifications * 100)
        return 0
        
    @property
    def avg_latency(self):
        """Average latency in seconds"""
        if self.latencies:
            return statistics.mean(self.latencies)
        return 0
        
    @property
    def p95_latency(self):
        """95th percentile latency in seconds"""
        if self.latencies:
            return statistics.quantiles(self.latencies, n=20)[-1]
        return 0
        
    def print_results(self):
        """Print benchmark results"""
        print("\nBenchmark Results:")
        print(f"Total Duration: {self.duration:.2f}s")
        print(f"Total Notifications: {self.total_notifications}")
        print(f"Failed Notifications: {self.failed_notifications}")
        print(f"Throughput: {self.throughput:.2f} notifications/s")
        print(f"Success Rate: {self.success_rate:.1f}%")
        print(f"Average Latency: {self.avg_latency:.3f}s")
        print(f"P95 Latency: {self.p95_latency:.3f}s")

@requires_telegram
@pytest.mark.asyncio
async def test_notification_throughput(notification_manager):
    """Benchmark notification throughput and latency"""
    benchmark = NotificationBenchmark()
    num_notifications = 20  # Adjust based on rate limits
    
    async def send_with_timing(message, priority="normal"):
        try:
            start = time.time()
            await notification_manager.send_notification(
                message=message,
                priority=priority
            )
            latency = time.time() - start
            benchmark.latencies.append(latency)
            benchmark.total_notifications += 1
        except Exception:
            benchmark.failed_notifications += 1
    
    # Prepare test messages
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    messages = [
        f"Benchmark Test {i} [{timestamp}]"
        for i in range(num_notifications)
    ]
    
    # Run benchmark
    benchmark.start_time = time.time()
    
    # Send notifications with controlled concurrency
    chunk_size = 5  # Number of concurrent notifications
    for i in range(0, len(messages), chunk_size):
        chunk = messages[i:i + chunk_size]
        tasks = [send_with_timing(msg) for msg in chunk]
        await asyncio.gather(*tasks)
        await asyncio.sleep(2)  # Rate limit compliance
        
    benchmark.end_time = time.time()
    
    # Print results
    benchmark.print_results()
    
    # Basic assertions
    assert benchmark.success_rate > 95  # At least 95% success
    assert benchmark.avg_latency < 2.0  # Average latency under 2s

@requires_telegram
@pytest.mark.asyncio
async def test_notification_stress(notification_manager):
    """Stress test with varied message types and priorities"""
    benchmark = NotificationBenchmark()
    
    async def send_varied_message(index):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            priority = ["critical", "high", "normal", "low", "info"][index % 5]
            
            # Vary message size and complexity
            if index % 3 == 0:
                message = f"Short message {index}"
            elif index % 3 == 1:
                message = f"Medium message {index}\n" + "-" * 50
            else:
                message = (f"Long message {index}\n" + "-" * 100 + 
                         "\n".join([f"Line {i}" for i in range(10)]))
                
            metadata = {
                "test_id": f"stress_{index}",
                "timestamp": timestamp,
                "complexity": len(message)
            }
            
            start = time.time()
            await notification_manager.send_notification(
                message=message,
                priority=priority,
                metadata=metadata
            )
            latency = time.time() - start
            benchmark.latencies.append(latency)
            benchmark.total_notifications += 1
            
        except Exception:
            benchmark.failed_notifications += 1
    
    # Run stress test
    num_messages = 15  # Adjust based on rate limits
    benchmark.start_time = time.time()
    
    # Send varied messages with controlled concurrency
    chunk_size = 3  # Smaller chunks for stress test
    for i in range(0, num_messages, chunk_size):
        tasks = [send_varied_message(j) 
                for j in range(i, min(i + chunk_size, num_messages))]
        await asyncio.gather(*tasks)
        await asyncio.sleep(2)  # Rate limit compliance
        
    benchmark.end_time = time.time()
    
    # Print results
    benchmark.print_results()
    
    # Assertions for stress test
    assert benchmark.success_rate > 90  # Allow slightly lower success rate
    assert benchmark.avg_latency < 2.5  # Allow slightly higher latency 