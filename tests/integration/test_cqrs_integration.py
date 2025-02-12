import pytest
import asyncio
from datetime import datetime
from uuid import uuid4
from src.cqrs.commands.token_discovery import (
    DiscoverTokenCommand,
    ValidateTokenCommand,
    TokenDiscoveryCommandHandler,
    TokenValidationCommandHandler
)
from src.cqrs.queries.token_discovery import (
    GetTokenDetailsQuery,
    GetTokenValidationStatusQuery,
    TokenDetailsQueryHandler,
    TokenValidationStatusQueryHandler,
    TokenDetails
)
from src.cqrs.events.redis_store import RedisEventStore

@pytest.mark.integration
class TestCQRSIntegration:
    """Integration tests for CQRS implementation"""
    
    @pytest.fixture
    async def event_store(self):
        """Create event store for testing"""
        store = RedisEventStore(
            redis_url="redis://localhost:6379/1",
            namespace="test_token"
        )
        await store.init()
        yield store
        
        # Cleanup test data
        if store.redis:
            await store.redis.flushdb()
            await store.redis.close()
    
    @pytest.fixture
    def token_data(self):
        """Sample token data for testing"""
        return {
            "chain_id": 1,
            "token_address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "source": "test",
            "validation_data": {
                "name": "Test Token",
                "symbol": "TEST",
                "decimals": 18
            }
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_token_discovery(self, event_store, token_data):
        """Test complete token discovery flow using CQRS"""
        # 1. Discover Token Command
        discovery_command = DiscoverTokenCommand(
            id=str(uuid4()),
            chain_id=token_data["chain_id"],
            token_address=token_data["token_address"],
            source=token_data["source"],
            validation_data=token_data["validation_data"]
        )
        
        discovery_handler = TokenDiscoveryCommandHandler(event_store)
        assert await discovery_handler.validate(discovery_command)
        await discovery_handler.handle(discovery_command)
        
        # 2. Validate Token Command
        validation_command = ValidateTokenCommand(
            id=str(uuid4()),
            chain_id=token_data["chain_id"],
            token_address=token_data["token_address"],
            validation_rules={
                "min_liquidity": 1000000,
                "min_holders": 100
            }
        )
        
        validation_handler = TokenValidationCommandHandler(event_store)
        assert await validation_handler.validate(validation_command)
        await validation_handler.handle(validation_command)
        
        # 3. Query Token Details
        details_query = GetTokenDetailsQuery(
            id=str(uuid4()),
            chain_id=token_data["chain_id"],
            token_address=token_data["token_address"]
        )
        
        details_handler = TokenDetailsQueryHandler(event_store)
        token_details = await details_handler.handle(details_query)
        
        assert isinstance(token_details, TokenDetails)
        assert token_details.token_address == token_data["token_address"]
        assert token_details.chain_id == token_data["chain_id"]
        assert token_details.source == token_data["source"]
        assert token_details.validation_status == "VALIDATED"
        
        # 4. Query Validation Status
        status_query = GetTokenValidationStatusQuery(
            id=str(uuid4()),
            chain_id=token_data["chain_id"],
            token_address=token_data["token_address"]
        )
        
        status_handler = TokenValidationStatusQueryHandler(event_store)
        status = await status_handler.handle(status_query)
        
        assert status == "VALIDATED"
    
    @pytest.mark.asyncio
    async def test_concurrent_token_operations(self, event_store, token_data):
        """Test concurrent token operations using CQRS"""
        # Create multiple commands for different tokens
        commands = []
        for i in range(5):
            token_address = f"0x{i:040x}"
            commands.append(
                DiscoverTokenCommand(
                    id=str(uuid4()),
                    chain_id=token_data["chain_id"],
                    token_address=token_address,
                    source=token_data["source"],
                    validation_data=token_data["validation_data"]
                )
            )
        
        # Execute commands concurrently
        discovery_handler = TokenDiscoveryCommandHandler(event_store)
        await asyncio.gather(
            *[discovery_handler.handle(cmd) for cmd in commands]
        )
        
        # Verify all tokens were discovered
        for cmd in commands:
            events = await event_store.get_events(cmd.token_address)
            assert len(events) == 1
            assert events[0].event_type == "TOKEN_DISCOVERED"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, event_store):
        """Test error handling in CQRS implementation"""
        # Test with invalid token address
        invalid_command = DiscoverTokenCommand(
            id=str(uuid4()),
            chain_id=1,
            token_address="invalid_address",
            source="test",
            validation_data={}
        )
        
        handler = TokenDiscoveryCommandHandler(event_store)
        assert not await handler.validate(invalid_command)
        
        with pytest.raises(Exception):
            await handler.handle(invalid_command)
        
        # Test querying non-existent token
        query = GetTokenDetailsQuery(
            id=str(uuid4()),
            chain_id=1,
            token_address="0x0000000000000000000000000000000000000000"
        )
        
        query_handler = TokenDetailsQueryHandler(event_store)
        with pytest.raises(ValueError):
            await query_handler.handle(query)
    
    @pytest.mark.asyncio
    async def test_event_versioning(self, event_store, token_data):
        """Test event versioning in CQRS implementation"""
        handler = TokenDiscoveryCommandHandler(event_store)
        
        # Create multiple events for the same token
        versions = []
        for i in range(3):
            command = DiscoverTokenCommand(
                id=str(uuid4()),
                chain_id=token_data["chain_id"],
                token_address=token_data["token_address"],
                source=f"test_{i}",
                validation_data=token_data["validation_data"]
            )
            await handler.handle(command)
            
            version = await event_store.get_latest_version(
                token_data["token_address"]
            )
            versions.append(version)
        
        # Verify versions are increasing
        assert versions[0] < versions[1] < versions[2]
        
        # Test getting events since version
        events = await event_store.get_events(
            token_data["token_address"],
            since_version=versions[0]
        )
        assert len(events) == 2 