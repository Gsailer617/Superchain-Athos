import pytest
from unittest.mock import Mock, patch
from src.core.register_adapters import (
    register_bridge_adapters,
    get_registered_adapters,
    unregister_adapter,
    validate_adapter,
    AdapterRegistrationError
)
from src.core.bridge_adapter import BridgeAdapter, BridgeAdapterFactory
from src.core.debridge_adapter import DeBridgeAdapter
from src.core.superbridge_adapter import SuperbridgeAdapter
from src.core.across_adapter import AcrossAdapter

@pytest.fixture(autouse=True)
def clear_adapters():
    """Clear registered adapters before and after each test"""
    BridgeAdapterFactory.clear_adapters()
    yield
    BridgeAdapterFactory.clear_adapters()

def test_validate_adapter():
    """Test adapter validation"""
    # Should pass for valid adapter
    assert validate_adapter(DeBridgeAdapter)
    
    # Should fail for incomplete adapter
    class IncompleteAdapter(BridgeAdapter):
        pass
    
    assert not validate_adapter(IncompleteAdapter)

def test_register_bridge_adapters():
    """Test registering bridge adapters"""
    # Register adapters
    registered = register_bridge_adapters()
    
    # Verify registration
    assert len(registered) == 3
    assert 'debridge' in registered
    assert 'superbridge' in registered
    assert 'across' in registered
    
    # Verify adapter classes
    assert registered['debridge'] == DeBridgeAdapter
    assert registered['superbridge'] == SuperbridgeAdapter
    assert registered['across'] == AcrossAdapter

def test_get_registered_adapters():
    """Test getting registered adapters"""
    # Register adapters
    register_bridge_adapters()
    
    # Get registered adapters
    adapters = get_registered_adapters()
    
    # Verify adapters
    assert len(adapters) == 3
    assert 'debridge' in adapters
    assert 'superbridge' in adapters
    assert 'across' in adapters

def test_unregister_adapter():
    """Test unregistering adapters"""
    # Register adapters
    register_bridge_adapters()
    
    # Unregister one adapter
    assert unregister_adapter('debridge')
    
    # Verify it's removed
    adapters = get_registered_adapters()
    assert 'debridge' not in adapters
    assert len(adapters) == 2
    
    # Try to unregister non-existent adapter
    assert not unregister_adapter('nonexistent')

def test_register_duplicate_adapter():
    """Test registering duplicate adapter"""
    # Register adapters
    register_bridge_adapters()
    
    # Try to register again
    with pytest.raises(AdapterRegistrationError):
        register_bridge_adapters()

@patch('src.core.register_adapters.validate_adapter')
def test_register_invalid_adapter(mock_validate):
    """Test registering invalid adapter"""
    # Make validation fail
    mock_validate.return_value = False
    
    # Try to register
    with pytest.raises(AdapterRegistrationError):
        register_bridge_adapters()

def test_adapter_registration_error():
    """Test adapter registration error handling"""
    class BrokenAdapter(BridgeAdapter):
        def __init__(self, *args, **kwargs):
            raise Exception("Initialization error")
    
    # Try to register broken adapter
    with pytest.raises(AdapterRegistrationError):
        BridgeAdapterFactory.register_adapter('broken', BrokenAdapter) 