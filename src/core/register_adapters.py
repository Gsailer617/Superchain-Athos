from typing import Dict, Type, Optional
import logging
from .bridge_adapter import BridgeAdapterFactory, BridgeAdapter
from .debridge_adapter import DeBridgeAdapter
from .superbridge_adapter import SuperbridgeAdapter
from .across_adapter import AcrossAdapter
from .stargate_adapter import StargateAdapter
from .layerzero_adapter import LayerZeroAdapter
from .linea_bridge_adapter import LineaBridgeAdapter
from .mantle_bridge_adapter import MantleBridgeAdapter
from .avalanche_bridge_adapter import AvalancheBridgeAdapter
from .gnosis_bridge_adapter import GnosisBridgeAdapter
from .mode_bridge_adapter import ModeBridgeAdapter
from .sonic_bridge_adapter import SonicBridgeAdapter

logger = logging.getLogger(__name__)

class AdapterRegistrationError(Exception):
    """Exception raised for errors during adapter registration"""
    pass

def validate_adapter(adapter_class: Type[BridgeAdapter]) -> bool:
    """Validate that an adapter class implements all required methods"""
    required_methods = [
        'validate_transfer',
        'estimate_fees',
        'estimate_time',
        'prepare_transfer',
        'verify_message',
        'get_bridge_state',
        'monitor_liquidity',
        'recover_failed_transfer'
    ]
    
    for method in required_methods:
        if not hasattr(adapter_class, method):
            logger.error(f"Adapter {adapter_class.__name__} missing required method: {method}")
            return False
    return True

def register_bridge_adapters() -> Dict[str, Type[BridgeAdapter]]:
    """Register all bridge adapters with the factory
    
    Returns:
        Dict[str, Type[BridgeAdapter]]: Mapping of registered adapter names to their classes
    
    Raises:
        AdapterRegistrationError: If registration fails for any adapter
    """
    registered_adapters: Dict[str, Type[BridgeAdapter]] = {}
    
    adapters_to_register = [
        ('debridge', DeBridgeAdapter),
        ('superbridge', SuperbridgeAdapter),
        ('across', AcrossAdapter),
        ('stargate', StargateAdapter),
        ('layerzero', LayerZeroAdapter),
        ('linea', LineaBridgeAdapter),
        ('mantle', MantleBridgeAdapter),
        ('avalanche', AvalancheBridgeAdapter),
        ('gnosis', GnosisBridgeAdapter),
        ('mode', ModeBridgeAdapter),
        ('sonic', SonicBridgeAdapter)
    ]
    
    for adapter_name, adapter_class in adapters_to_register:
        try:
            # Validate adapter implementation
            if not validate_adapter(adapter_class):
                raise AdapterRegistrationError(
                    f"Adapter {adapter_name} failed validation checks"
                )
            
            # Register with factory
            BridgeAdapterFactory.register_adapter(adapter_name, adapter_class)
            registered_adapters[adapter_name] = adapter_class
            
            logger.info(f"Successfully registered {adapter_name} adapter")
            
        except Exception as e:
            logger.error(f"Failed to register {adapter_name} adapter: {str(e)}")
            raise AdapterRegistrationError(f"Error registering {adapter_name}: {str(e)}")
    
    # Log registration summary
    logger.info("Registered bridge adapters:")
    for name in registered_adapters:
        logger.info(f"- {name}")
    
    return registered_adapters

def get_registered_adapters() -> Dict[str, Type[BridgeAdapter]]:
    """Get all currently registered adapters
    
    Returns:
        Dict[str, Type[BridgeAdapter]]: Mapping of registered adapter names to their classes
    """
    return BridgeAdapterFactory.get_registered_adapters()

def unregister_adapter(adapter_name: str) -> bool:
    """Unregister a specific adapter
    
    Args:
        adapter_name: Name of the adapter to unregister
    
    Returns:
        bool: True if adapter was unregistered, False if it wasn't registered
    """
    try:
        BridgeAdapterFactory.unregister_adapter(adapter_name)
        logger.info(f"Successfully unregistered {adapter_name} adapter")
        return True
    except KeyError:
        logger.warning(f"Attempted to unregister non-existent adapter: {adapter_name}")
        return False
    except Exception as e:
        logger.error(f"Error unregistering {adapter_name} adapter: {str(e)}")
        return False 