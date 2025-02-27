"""
Dependency Injection System

This module provides a flexible dependency injection system for the core components:
- Service registration and resolution
- Lifecycle management
- Scoped dependencies
- Configuration-based initialization
"""

from typing import Dict, Any, Optional, Type, TypeVar, Generic, Callable, Union, List, cast
import inspect
from dataclasses import dataclass, field
import logging
from enum import Enum
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)
T = TypeVar('T')
S = TypeVar('S')

class LifecycleState(Enum):
    """Lifecycle states for components"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DISPOSING = "disposing"
    DISPOSED = "disposed"

class Scope(Enum):
    """Service registration scopes"""
    SINGLETON = "singleton"  # One instance for the entire application
    SCOPED = "scoped"      # One instance per scope
    TRANSIENT = "transient"  # New instance each time

@dataclass
class ServiceRegistration(Generic[T]):
    """Service registration information"""
    service_type: Type[T]
    implementation_type: Optional[Type[T]] = None
    implementation_factory: Optional[Callable[['DependencyContainer'], T]] = None
    instance: Optional[T] = None
    scope: Scope = Scope.SINGLETON
    lifecycle_state: LifecycleState = LifecycleState.UNINITIALIZED

class DependencyContainer:
    """Main dependency container"""
    
    def __init__(self):
        """Initialize container"""
        self._registrations: Dict[Type, ServiceRegistration] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._current_scope_id: Optional[str] = None
        self._lock = threading.RLock()
        
    def register(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        scope: Scope = Scope.SINGLETON
    ) -> None:
        """Register a service with its implementation type
        
        Args:
            service_type: Interface or base type
            implementation_type: Concrete implementation type
            scope: Service lifetime scope
        """
        with self._lock:
            if service_type in self._registrations:
                raise ValueError(f"Service type {service_type.__name__} is already registered")
                
            self._registrations[service_type] = ServiceRegistration(
                service_type=service_type,
                implementation_type=implementation_type or service_type,
                scope=scope
            )
            
    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[['DependencyContainer'], T],
        scope: Scope = Scope.SINGLETON
    ) -> None:
        """Register a service with a factory function
        
        Args:
            service_type: Interface or base type
            factory: Factory function that creates the service
            scope: Service lifetime scope
        """
        with self._lock:
            if service_type in self._registrations:
                raise ValueError(f"Service type {service_type.__name__} is already registered")
                
            self._registrations[service_type] = ServiceRegistration(
                service_type=service_type,
                implementation_factory=factory,
                scope=scope
            )
            
    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """Register an existing instance
        
        Args:
            service_type: Interface or base type
            instance: Instance to register
        """
        with self._lock:
            if service_type in self._registrations:
                raise ValueError(f"Service type {service_type.__name__} is already registered")
                
            self._registrations[service_type] = ServiceRegistration(
                service_type=service_type,
                instance=instance,
                scope=Scope.SINGLETON,
                lifecycle_state=LifecycleState.ACTIVE
            )
            
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service by type
        
        Args:
            service_type: Type to resolve
            
        Returns:
            Instance of the requested service
            
        Raises:
            KeyError: If service type is not registered
            ValueError: If service cannot be created
        """
        with self._lock:
            if service_type not in self._registrations:
                raise KeyError(f"Service type {service_type.__name__} is not registered")
                
            registration = self._registrations[service_type]
            
            # Check if we already have an instance for singleton
            if registration.scope == Scope.SINGLETON and registration.instance is not None:
                return registration.instance
                
            # Check if we have a scoped instance
            if registration.scope == Scope.SCOPED and self._current_scope_id is not None:
                scope_instances = self._scoped_instances.get(self._current_scope_id, {})
                if service_type in scope_instances:
                    return scope_instances[service_type]
                    
            # Create new instance
            instance = self._create_instance(registration)
            
            # Store instance if singleton
            if registration.scope == Scope.SINGLETON:
                registration.instance = instance
                registration.lifecycle_state = LifecycleState.ACTIVE
                
            # Store instance if scoped
            if registration.scope == Scope.SCOPED and self._current_scope_id is not None:
                scope_instances = self._scoped_instances.setdefault(self._current_scope_id, {})
                scope_instances[service_type] = instance
                
            return instance
            
    def _create_instance(self, registration: ServiceRegistration[T]) -> T:
        """Create a new instance of a service
        
        Args:
            registration: Service registration information
            
        Returns:
            New instance of the service
            
        Raises:
            ValueError: If service cannot be created
        """
        # Check if we already have an instance
        if registration.instance is not None:
            return registration.instance
            
        # Check if we have a factory
        if registration.implementation_factory is not None:
            try:
                instance = registration.implementation_factory(self)
                return instance
            except Exception as e:
                raise ValueError(f"Error creating instance with factory: {str(e)}")
                
        # Try to create instance from implementation type
        if registration.implementation_type is not None:
            try:
                # Check constructor parameters
                signature = inspect.signature(registration.implementation_type.__init__)
                params = signature.parameters
                
                # Skip self parameter
                param_values = {}
                for name, param in list(params.items())[1:]:  # Skip self
                    if param.annotation != inspect.Parameter.empty:
                        # Try to resolve parameter from container
                        try:
                            param_values[name] = self.resolve(param.annotation)
                        except KeyError:
                            # If parameter has default value, use it
                            if param.default != inspect.Parameter.empty:
                                param_values[name] = param.default
                            else:
                                raise ValueError(
                                    f"Cannot resolve parameter {name} of type {param.annotation}"
                                )
                
                # Create instance
                instance = registration.implementation_type(**param_values)
                return instance
                
            except Exception as e:
                raise ValueError(f"Error creating instance of {registration.implementation_type.__name__}: {str(e)}")
                
        raise ValueError(f"Cannot create instance for {registration.service_type.__name__}")
    
    @contextmanager
    def create_scope(self, scope_id: Optional[str] = None) -> 'DependencyContainer':
        """Create a new dependency scope
        
        Args:
            scope_id: Optional identifier for the scope
            
        Returns:
            Scoped dependency container
            
        Example:
            with container.create_scope() as scope:
                service = scope.resolve(IService)
        """
        scope_id = scope_id or f"scope_{id(threading.current_thread())}_{len(self._scoped_instances)}"
        
        # Initialize scope
        self._scoped_instances[scope_id] = {}
        prev_scope_id = self._current_scope_id
        self._current_scope_id = scope_id
        
        try:
            yield self
        finally:
            # Dispose scope
            if scope_id in self._scoped_instances:
                # Dispose any scoped services that implement a dispose method
                for service in self._scoped_instances[scope_id].values():
                    if hasattr(service, 'dispose') and callable(getattr(service, 'dispose')):
                        try:
                            service.dispose()
                        except Exception as e:
                            logger.error(f"Error disposing service: {str(e)}")
                
                del self._scoped_instances[scope_id]
                
            self._current_scope_id = prev_scope_id
    
    def dispose(self) -> None:
        """Dispose all services"""
        with self._lock:
            # Dispose all singletons that implement a dispose method
            for registration in self._registrations.values():
                if registration.instance is not None:
                    if hasattr(registration.instance, 'dispose') and callable(getattr(registration.instance, 'dispose')):
                        try:
                            registration.lifecycle_state = LifecycleState.DISPOSING
                            registration.instance.dispose()
                            registration.lifecycle_state = LifecycleState.DISPOSED
                        except Exception as e:
                            logger.error(f"Error disposing service: {str(e)}")
                            
            # Dispose all scoped instances
            for scope_id in list(self._scoped_instances.keys()):
                for service in self._scoped_instances[scope_id].values():
                    if hasattr(service, 'dispose') and callable(getattr(service, 'dispose')):
                        try:
                            service.dispose()
                        except Exception as e:
                            logger.error(f"Error disposing service: {str(e)}")
                            
                del self._scoped_instances[scope_id]
                
            self._registrations.clear()


# Global container instance
container = DependencyContainer()

def register_core_services() -> None:
    """Register all core services in the container"""
    from .bridge_adapter import BridgeAdapterFactory
    from .bridge_manager import BridgeManager
    from .chain_connector import ChainConnector
    from .enhanced_chain_connector import EnhancedChainConnector
    from .config_manager import ConfigManager
    from .gas_manager import GasManager
    from .error_handling import ErrorHandler
    from .circuit_breaker import CircuitBreakerRegistry
    from .adapter_optimization import AdapterOptimizerRegistry
    from .adaptive_timeout import AdaptiveTimeoutRegistry
    from .health_monitor import HealthMonitor
    
    # Register core components
    container.register(BridgeManager)
    container.register(ChainConnector)
    container.register(ConfigManager)
    container.register(GasManager)
    container.register(ErrorHandler)
    container.register(HealthMonitor)
    
    # Register enhanced implementations
    container.register(ChainConnector, EnhancedChainConnector)
    
    # Register singleton registries
    container.register_instance(CircuitBreakerRegistry, CircuitBreakerRegistry)
    container.register_instance(AdapterOptimizerRegistry, AdapterOptimizerRegistry)
    container.register_instance(AdaptiveTimeoutRegistry, AdaptiveTimeoutRegistry)
    container.register_instance(BridgeAdapterFactory, BridgeAdapterFactory) 