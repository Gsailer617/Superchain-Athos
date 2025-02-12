"""Advanced observability and monitoring system"""

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.aiohttp import AioHttpIntegration
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.prometheus import PrometheusInstrumentor
import structlog
from typing import Dict, Any, Optional
import logging
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ObservabilityConfig:
    """Configuration for observability system"""
    sentry_dsn: str
    jaeger_host: str = "localhost"
    jaeger_port: int = 6831
    log_level: str = "INFO"
    log_path: Path = Path("logs")
    enable_tracing: bool = True
    enable_metrics: bool = True
    environment: str = "production"

class ObservabilityManager:
    """Manages advanced observability features"""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self._setup_logging()
        self._setup_sentry()
        self._setup_tracing()
        self._setup_metrics()
        
    def _setup_logging(self):
        """Setup structured logging with advanced processors"""
        # Create log directory
        self.config.log_path.mkdir(parents=True, exist_ok=True)
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            logger_factory=structlog.PrintLoggerFactory(),
            wrapper_class=structlog.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(
            self.config.log_path / "application.log"
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Setup JSON handler for structured logging
        json_handler = logging.FileHandler(
            self.config.log_path / "structured.json"
        )
        json_handler.setFormatter(
            logging.Formatter('%(message)s')
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.log_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(json_handler)
        
    def _setup_sentry(self):
        """Initialize Sentry for error tracking"""
        sentry_sdk.init(
            dsn=self.config.sentry_dsn,
            environment=self.config.environment,
            integrations=[
                LoggingIntegration(
                    level=logging.INFO,
                    event_level=logging.ERROR
                ),
                AioHttpIntegration(),
            ],
            traces_sample_rate=1.0,
            send_default_pii=False,
            before_send=self._before_send_sentry
        )
        
    def _setup_tracing(self):
        """Setup distributed tracing with OpenTelemetry"""
        if not self.config.enable_tracing:
            return
            
        # Create tracer provider
        trace.set_tracer_provider(TracerProvider())
        
        # Setup Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=self.config.jaeger_host,
            agent_port=self.config.jaeger_port,
        )
        
        # Add span processor
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(jaeger_exporter)
        )
        
        # Instrument libraries
        AioHttpClientInstrumentor().instrument()
        RedisInstrumentor().instrument()
        PrometheusInstrumentor().instrument()
        
    def _setup_metrics(self):
        """Setup advanced metrics collection"""
        if not self.config.enable_metrics:
            return
            
        # Additional metric instrumentation can be added here
        pass
        
    def _before_send_sentry(
        self,
        event: Dict[str, Any],
        hint: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process event before sending to Sentry"""
        # Add custom context
        if 'extra' not in event:
            event['extra'] = {}
            
        event['extra'].update({
            'environment': self.config.environment,
            'runtime_context': self._get_runtime_context()
        })
        
        # Filter sensitive data
        if 'request' in event and 'headers' in event['request']:
            headers = event['request']['headers']
            if 'Authorization' in headers:
                headers['Authorization'] = '[FILTERED]'
                
        return event
        
    def _get_runtime_context(self) -> Dict[str, Any]:
        """Get current runtime context"""
        import psutil
        import platform
        
        process = psutil.Process()
        
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_percent': process.memory_percent(),
            'open_files': len(process.open_files()),
            'threads': process.num_threads(),
            'python_version': platform.python_version(),
            'platform': platform.platform()
        }
        
    def create_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> trace.Span:
        """Create a new trace span"""
        tracer = trace.get_tracer(__name__)
        return tracer.start_span(
            name,
            attributes=attributes or {}
        )
        
    def log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        level: str = "INFO"
    ):
        """Log an event with structured data"""
        logger = structlog.get_logger()
        logger.bind(event_type=event_type).log(
            level,
            "Event logged",
            **data
        )
        
    def capture_exception(
        self,
        exc: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """Capture and report an exception"""
        with sentry_sdk.push_scope() as scope:
            if context:
                for key, value in context.items():
                    scope.set_extra(key, value)
            sentry_sdk.capture_exception(exc)
            
    def monitor_operation(self, name: str):
        """Decorator for monitoring operations"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                with self.create_span(name) as span:
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("success", True)
                        return result
                    except Exception as e:
                        span.set_attribute("success", False)
                        span.set_attribute("error", str(e))
                        self.capture_exception(e, {
                            "operation": name,
                            "args": args,
                            "kwargs": kwargs
                        })
                        raise
            return wrapper
        return decorator
        
    async def cleanup(self):
        """Cleanup observability resources"""
        # Flush any pending traces
        trace.get_tracer_provider().shutdown()
        
        # Close Sentry client
        client = sentry_sdk.Hub.current.client
        if client is not None:
            client.close(timeout=2.0)
            
class OperationMonitor:
    """Context manager for monitoring operations"""
    
    def __init__(
        self,
        observability: ObservabilityManager,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        self.observability = observability
        self.operation_name = operation_name
        self.attributes = attributes or {}
        self.span = None
        
    async def __aenter__(self):
        """Start monitoring operation"""
        self.span = self.observability.create_span(
            self.operation_name,
            self.attributes
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End monitoring operation"""
        if exc_val is not None:
            self.span.set_attribute("success", False)
            self.span.set_attribute("error", str(exc_val))
            self.observability.capture_exception(exc_val, {
                "operation": self.operation_name,
                "attributes": self.attributes
            })
        else:
            self.span.set_attribute("success", True)
            
        self.span.end() 