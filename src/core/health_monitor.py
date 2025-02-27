"""
Health Monitoring System

This module provides health monitoring for the bridge system:
- Component health checks
- System-wide status reporting
- Performance metrics collection
- Alerting mechanism
- Historical data for analysis
"""

import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import threading
import json
from prometheus_client import Gauge, Counter, Histogram, Summary
import functools

from .bridge_adapter import BridgeState
from .circuit_breaker import CircuitBreakerRegistry
from .error_handling import ErrorHandler
from .adapter_optimization import AdapterOptimizerRegistry
from .adaptive_timeout import AdaptiveTimeoutRegistry

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class HealthCheckResult:
    """Result of a health check"""
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SystemHealthSnapshot:
    """Snapshot of system health at a point in time"""
    overall_status: HealthStatus
    component_status: Dict[str, HealthCheckResult]
    resource_usage: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Alert:
    """System alert information"""
    level: AlertLevel
    component: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    resolution_message: Optional[str] = None
    resolution_timestamp: Optional[datetime] = None

class HealthCheck:
    """Health check definition"""
    
    def __init__(
        self,
        name: str,
        check_func: Callable[[], Tuple[HealthStatus, str, Dict[str, Any]]],
        interval_seconds: float = 60.0
    ):
        """Initialize health check
        
        Args:
            name: Name of the component being checked
            check_func: Function that performs the health check
            interval_seconds: How often to run the check
        """
        self.name = name
        self.check_func = check_func
        self.interval_seconds = interval_seconds
        self.last_result: Optional[HealthCheckResult] = None
        self.last_check_time: Optional[datetime] = None
        
    async def run(self) -> HealthCheckResult:
        """Run the health check
        
        Returns:
            Health check result
        """
        try:
            status, message, details = self.check_func()
            result = HealthCheckResult(
                component=self.name,
                status=status,
                message=message,
                details=details
            )
        except Exception as e:
            logger.error(f"Error running health check {self.name}: {str(e)}")
            result = HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e), "traceback": str(e.__traceback__)}
            )
            
        self.last_result = result
        self.last_check_time = datetime.now()
        return result

class HealthMonitor:
    """System health monitor"""
    
    def __init__(
        self,
        circuit_breaker_registry: CircuitBreakerRegistry,
        error_handler: ErrorHandler,
        adapter_optimizer_registry: AdapterOptimizerRegistry,
        timeout_registry: AdaptiveTimeoutRegistry
    ):
        """Initialize health monitor
        
        Args:
            circuit_breaker_registry: Circuit breaker registry
            error_handler: Error handler
            adapter_optimizer_registry: Adapter optimizer registry
            timeout_registry: Adaptive timeout registry
        """
        self.circuit_breaker_registry = circuit_breaker_registry
        self.error_handler = error_handler
        self.adapter_optimizer_registry = adapter_optimizer_registry
        self.timeout_registry = timeout_registry
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_history: List[SystemHealthSnapshot] = []
        self.alerts: List[Alert] = []
        self.running = False
        self.lock = threading.RLock()
        self._setup_metrics()
        self._setup_standard_checks()
        
    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        self.health_gauge = Gauge(
            'system_health',
            'System health status (0=healthy, 1=degraded, 2=unhealthy, 3=critical, 4=unknown)',
            ['component']
        )
        self.alert_counter = Counter(
            'system_alerts_total',
            'Total number of alerts',
            ['level', 'component']
        )
        self.check_duration = Histogram(
            'health_check_duration_seconds',
            'Time spent running health checks',
            ['component']
        )
        
    def _setup_standard_checks(self):
        """Set up standard health checks"""
        # Circuit breaker health check
        self.register_check(
            "circuit_breakers",
            lambda: self._check_circuit_breakers(),
            interval_seconds=30.0
        )
        
        # Error subsystem health check
        self.register_check(
            "error_handling",
            lambda: self._check_error_subsystem(),
            interval_seconds=60.0
        )
        
        # Adapter performance check
        self.register_check(
            "adapter_performance",
            lambda: self._check_adapter_performance(),
            interval_seconds=120.0
        )
        
        # Resource usage check
        self.register_check(
            "resource_usage",
            lambda: self._check_resource_usage(),
            interval_seconds=60.0
        )
        
        # Timeout configuration check
        self.register_check(
            "timeout_configuration",
            lambda: self._check_timeout_configuration(),
            interval_seconds=90.0
        )
        
    def register_check(
        self,
        name: str,
        check_func: Callable[[], Tuple[HealthStatus, str, Dict[str, Any]]],
        interval_seconds: float = 60.0
    ) -> None:
        """Register a new health check
        
        Args:
            name: Name of the component being checked
            check_func: Function that performs the health check
            interval_seconds: How often to run the check
        """
        with self.lock:
            if name in self.health_checks:
                raise ValueError(f"Health check {name} is already registered")
                
            self.health_checks[name] = HealthCheck(name, check_func, interval_seconds)
            
    def _check_circuit_breakers(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check circuit breaker health
        
        Returns:
            Health status, message, and details
        """
        try:
            circuit_states = {}
            open_circuits = []
            
            # Collect all circuit breakers and their states
            for name, metrics in self.circuit_breaker_registry.get_all_metrics().items():
                circuit_states[name] = metrics["state"]
                if metrics["state"] == "open":
                    open_circuits.append(name)
                    
            if len(open_circuits) == 0:
                return (
                    HealthStatus.HEALTHY,
                    "All circuit breakers are closed",
                    {"circuit_states": circuit_states}
                )
            elif len(open_circuits) < 3:
                return (
                    HealthStatus.DEGRADED,
                    f"{len(open_circuits)} circuit breakers are open",
                    {"circuit_states": circuit_states, "open_circuits": open_circuits}
                )
            else:
                return (
                    HealthStatus.UNHEALTHY,
                    f"{len(open_circuits)} circuit breakers are open",
                    {"circuit_states": circuit_states, "open_circuits": open_circuits}
                )
                
        except Exception as e:
            return (
                HealthStatus.UNKNOWN,
                f"Failed to check circuit breakers: {str(e)}",
                {"error": str(e)}
            )
            
    def _check_error_subsystem(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check error handling subsystem health
        
        Returns:
            Health status, message, and details
        """
        try:
            error_summary = self.error_handler.get_error_summary()
            
            total_errors = sum(
                count for component_data in error_summary.values()
                for severity, count in component_data["counts_by_severity"].items()
            )
            
            critical_errors = sum(
                count for component_data in error_summary.values()
                for severity, count in component_data["counts_by_severity"].items()
                if severity == "CRITICAL"
            )
            
            if total_errors == 0:
                return (
                    HealthStatus.HEALTHY,
                    "No errors reported",
                    {"error_summary": error_summary}
                )
            elif critical_errors == 0 and total_errors < 10:
                return (
                    HealthStatus.HEALTHY,
                    f"{total_errors} non-critical errors reported",
                    {"error_summary": error_summary}
                )
            elif critical_errors == 0:
                return (
                    HealthStatus.DEGRADED,
                    f"{total_errors} errors reported, but none are critical",
                    {"error_summary": error_summary}
                )
            else:
                return (
                    HealthStatus.UNHEALTHY,
                    f"{critical_errors} critical errors reported out of {total_errors} total",
                    {"error_summary": error_summary}
                )
                
        except Exception as e:
            return (
                HealthStatus.UNKNOWN,
                f"Failed to check error subsystem: {str(e)}",
                {"error": str(e)}
            )
            
    def _check_adapter_performance(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check adapter performance
        
        Returns:
            Health status, message, and details
        """
        try:
            metrics = self.adapter_optimizer_registry.get_all_metrics()
            
            slow_adapters = []
            for adapter_id, adapter_metrics in metrics.items():
                # Check if any operation is taking too long (> 5 seconds on average)
                for op, time in adapter_metrics.get("average_response_times", {}).items():
                    if time > 5.0:
                        slow_adapters.append({
                            "adapter": adapter_metrics.get("adapter_name", adapter_id),
                            "operation": op,
                            "avg_time": time
                        })
                        
            if not slow_adapters:
                return (
                    HealthStatus.HEALTHY,
                    "All adapters performing within expected parameters",
                    {"metrics": metrics}
                )
            elif len(slow_adapters) < 3:
                return (
                    HealthStatus.DEGRADED,
                    f"{len(slow_adapters)} slow adapter operations detected",
                    {"metrics": metrics, "slow_adapters": slow_adapters}
                )
            else:
                return (
                    HealthStatus.UNHEALTHY,
                    f"{len(slow_adapters)} slow adapter operations detected",
                    {"metrics": metrics, "slow_adapters": slow_adapters}
                )
                
        except Exception as e:
            return (
                HealthStatus.UNKNOWN,
                f"Failed to check adapter performance: {str(e)}",
                {"error": str(e)}
            )
            
    def _check_resource_usage(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check system resource usage
        
        Returns:
            Health status, message, and details
        """
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            if cpu_percent < 70 and memory_percent < 80 and disk_percent < 80:
                return (
                    HealthStatus.HEALTHY,
                    "Resource usage is within expected parameters",
                    {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                        "disk_percent": disk_percent
                    }
                )
            elif cpu_percent < 85 and memory_percent < 90 and disk_percent < 90:
                return (
                    HealthStatus.DEGRADED,
                    "Resource usage is elevated",
                    {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                        "disk_percent": disk_percent
                    }
                )
            else:
                return (
                    HealthStatus.UNHEALTHY,
                    "Resource usage is critically high",
                    {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                        "disk_percent": disk_percent
                    }
                )
                
        except ImportError:
            return (
                HealthStatus.UNKNOWN,
                "Resource usage check not available (psutil not installed)",
                {}
            )
        except Exception as e:
            return (
                HealthStatus.UNKNOWN,
                f"Failed to check resource usage: {str(e)}",
                {"error": str(e)}
            )
            
    def _check_timeout_configuration(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check timeout configuration health
        
        Returns:
            Health status, message, and details
        """
        try:
            timeouts = self.timeout_registry.get_all_timeouts()
            states = self.timeout_registry.get_all_states()
            
            degraded_states = []
            for chain, state in states.items():
                if state != "normal":
                    degraded_states.append({
                        "chain": chain,
                        "state": state,
                        "timeout": timeouts.get(chain, 0)
                    })
                    
            if not degraded_states:
                return (
                    HealthStatus.HEALTHY,
                    "All chains have normal network conditions",
                    {"timeouts": timeouts, "states": {k: str(v) for k, v in states.items()}}
                )
            elif all(item["state"] != "unstable" for item in degraded_states):
                return (
                    HealthStatus.DEGRADED,
                    f"{len(degraded_states)} chains with degraded network conditions",
                    {
                        "timeouts": timeouts, 
                        "states": {k: str(v) for k, v in states.items()},
                        "degraded_chains": degraded_states
                    }
                )
            else:
                return (
                    HealthStatus.UNHEALTHY,
                    f"{len(degraded_states)} chains with degraded network conditions, including unstable chains",
                    {
                        "timeouts": timeouts, 
                        "states": {k: str(v) for k, v in states.items()},
                        "degraded_chains": degraded_states
                    }
                )
                
        except Exception as e:
            return (
                HealthStatus.UNKNOWN,
                f"Failed to check timeout configuration: {str(e)}",
                {"error": str(e)}
            )
    
    async def run_check(self, check_name: str) -> HealthCheckResult:
        """Run a specific health check
        
        Args:
            check_name: Name of check to run
            
        Returns:
            Health check result
            
        Raises:
            KeyError: If check not found
        """
        with self.lock:
            if check_name not in self.health_checks:
                raise KeyError(f"Health check {check_name} not found")
                
            check = self.health_checks[check_name]
            
        # Record time spent running check
        start_time = time.time()
        try:
            result = await check.run()
        finally:
            duration = time.time() - start_time
            self.check_duration.labels(component=check_name).observe(duration)
            
        # Update metrics
        self.health_gauge.labels(component=check_name).set(
            self._health_status_to_metric(result.status)
        )
        
        return result
    
    def _health_status_to_metric(self, status: HealthStatus) -> float:
        """Convert health status to metric value
        
        Args:
            status: Health status
            
        Returns:
            Numeric value for metric
        """
        return {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.UNHEALTHY: 2,
            HealthStatus.CRITICAL: 3,
            HealthStatus.UNKNOWN: 4
        }.get(status, 4)
        
    async def run_all_checks(self) -> SystemHealthSnapshot:
        """Run all health checks
        
        Returns:
            System health snapshot
        """
        results = {}
        resource_usage = {}
        
        # Run resource usage check first to include in snapshot
        if "resource_usage" in self.health_checks:
            result = await self.run_check("resource_usage")
            results["resource_usage"] = result
            resource_usage = result.details
        
        # Run all other checks
        check_names = [name for name in self.health_checks if name != "resource_usage"]
        for name in check_names:
            results[name] = await self.run_check(name)
            
        # Determine overall status (worst of all checks)
        status_order = [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
            HealthStatus.CRITICAL,
            HealthStatus.UNKNOWN
        ]
        
        all_statuses = [result.status for result in results.values()]
        overall_status = max(all_statuses, key=lambda x: status_order.index(x))
        
        # Create snapshot
        snapshot = SystemHealthSnapshot(
            overall_status=overall_status,
            component_status=results,
            resource_usage=resource_usage
        )
        
        with self.lock:
            self.health_history.append(snapshot)
            
            # Limit history size
            if len(self.health_history) > 100:
                self.health_history = self.health_history[-100:]
                
        return snapshot
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        self.running = True
        
        try:
            while self.running:
                # Determine which checks need to be run now
                now = datetime.now()
                checks_to_run = []
                
                with self.lock:
                    for name, check in self.health_checks.items():
                        if (check.last_check_time is None or 
                            (now - check.last_check_time).total_seconds() >= check.interval_seconds):
                            checks_to_run.append(name)
                
                # Run checks
                for name in checks_to_run:
                    try:
                        result = await self.run_check(name)
                        
                        # Check for alerts
                        if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                            self._create_alert(
                                AlertLevel.ERROR if result.status == HealthStatus.UNHEALTHY else AlertLevel.CRITICAL,
                                name,
                                result.message,
                                result.details
                            )
                        elif result.status == HealthStatus.DEGRADED:
                            self._create_alert(
                                AlertLevel.WARNING,
                                name,
                                result.message,
                                result.details
                            )
                    except Exception as e:
                        logger.error(f"Error running health check {name}: {str(e)}")
                        
                # Sleep until next check
                await asyncio.sleep(5)  # Check every 5 seconds
                
        except asyncio.CancelledError:
            logger.info("Health monitor loop cancelled")
            self.running = False
            
    def _create_alert(
        self,
        level: AlertLevel,
        component: str,
        message: str,
        details: Dict[str, Any]
    ) -> Alert:
        """Create and record an alert
        
        Args:
            level: Alert level
            component: Component that triggered the alert
            message: Alert message
            details: Alert details
            
        Returns:
            Created alert
        """
        with self.lock:
            # Check if we already have a similar unresolved alert
            for alert in self.alerts:
                if (alert.component == component and 
                    alert.level == level and 
                    not alert.resolved and
                    alert.message == message):
                    # Update existing alert
                    alert.details = details
                    return alert
                    
            # Create new alert
            alert = Alert(
                level=level,
                component=component,
                message=message,
                details=details
            )
            
            self.alerts.append(alert)
            self.alert_counter.labels(level=level.value, component=component).inc()
            
            # Log alert
            log_func = {
                AlertLevel.INFO: logger.info,
                AlertLevel.WARNING: logger.warning,
                AlertLevel.ERROR: logger.error,
                AlertLevel.CRITICAL: logger.critical
            }.get(level, logger.error)
            
            log_func(f"ALERT [{level.value.upper()}] {component}: {message}")
            
            return alert
            
    def acknowledge_alert(self, alert_idx: int) -> None:
        """Acknowledge an alert
        
        Args:
            alert_idx: Index of alert in alerts list
            
        Raises:
            IndexError: If alert index is invalid
        """
        with self.lock:
            if alert_idx < 0 or alert_idx >= len(self.alerts):
                raise IndexError(f"Invalid alert index: {alert_idx}")
                
            self.alerts[alert_idx].acknowledged = True
            
    def resolve_alert(self, alert_idx: int, resolution_message: str) -> None:
        """Resolve an alert
        
        Args:
            alert_idx: Index of alert in alerts list
            resolution_message: Message describing resolution
            
        Raises:
            IndexError: If alert index is invalid
        """
        with self.lock:
            if alert_idx < 0 or alert_idx >= len(self.alerts):
                raise IndexError(f"Invalid alert index: {alert_idx}")
                
            alert = self.alerts[alert_idx]
            alert.resolved = True
            alert.resolution_message = resolution_message
            alert.resolution_timestamp = datetime.now()
            
    def get_current_health(self) -> Dict[str, Any]:
        """Get current health status
        
        Returns:
            Health status summary
        """
        with self.lock:
            # Get latest check results
            check_results = {}
            for name, check in self.health_checks.items():
                if check.last_result is not None:
                    check_results[name] = {
                        "status": check.last_result.status.value,
                        "message": check.last_result.message,
                        "last_checked": check.last_check_time.isoformat() if check.last_check_time else None
                    }
                    
            # Get active alerts
            active_alerts = []
            for idx, alert in enumerate(self.alerts):
                if not alert.resolved:
                    active_alerts.append({
                        "index": idx,
                        "level": alert.level.value,
                        "component": alert.component,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "acknowledged": alert.acknowledged
                    })
                    
            # Get overall status
            overall_status = HealthStatus.HEALTHY
            if not check_results:
                overall_status = HealthStatus.UNKNOWN
            else:
                status_values = [
                    HealthStatus[status.upper()] 
                    for status in [result["status"] for result in check_results.values()]
                ]
                status_order = [
                    HealthStatus.HEALTHY,
                    HealthStatus.DEGRADED,
                    HealthStatus.UNHEALTHY,
                    HealthStatus.CRITICAL,
                    HealthStatus.UNKNOWN
                ]
                overall_status = max(status_values, key=lambda x: status_order.index(x))
                
            return {
                "overall_status": overall_status.value,
                "components": check_results,
                "active_alerts": active_alerts,
                "alert_count": len(active_alerts),
                "last_updated": datetime.now().isoformat()
            }
            
    def get_health_history(self) -> List[Dict[str, Any]]:
        """Get health history
        
        Returns:
            List of health snapshots
        """
        with self.lock:
            return [
                {
                    "timestamp": snapshot.timestamp.isoformat(),
                    "overall_status": snapshot.overall_status.value,
                    "components": {
                        name: {
                            "status": result.status.value,
                            "message": result.message
                        }
                        for name, result in snapshot.component_status.items()
                    },
                    "resource_usage": snapshot.resource_usage
                }
                for snapshot in self.health_history
            ]
            
    def get_alerts(self, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """Get alerts
        
        Args:
            include_resolved: Whether to include resolved alerts
            
        Returns:
            List of alerts
        """
        with self.lock:
            alerts_list = []
            for idx, alert in enumerate(self.alerts):
                if include_resolved or not alert.resolved:
                    alerts_list.append({
                        "index": idx,
                        "level": alert.level.value,
                        "component": alert.component,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "acknowledged": alert.acknowledged,
                        "resolved": alert.resolved,
                        "resolution_message": alert.resolution_message,
                        "resolution_timestamp": alert.resolution_timestamp.isoformat() if alert.resolution_timestamp else None
                    })
                    
            return alerts_list
            
    def stop(self) -> None:
        """Stop health monitor"""
        self.running = False 