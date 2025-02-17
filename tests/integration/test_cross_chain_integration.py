import pytest
import asyncio
from typing import Dict, Any, cast
from web3 import Web3
from eth_typing import ChecksumAddress

from src.core.bridge_adapter import BridgeConfig, BridgeState
from src.core.register_adapters import register_bridge_adapters
from src.execution.transaction_builder import CrossChainTransactionBuilder, CrossChainOpportunityType
from src.analysis.cross_chain_analyzer import CrossChainAnalyzer

@pytest.fixture
async def web3_connections():
    """Initialize Web3 connections for test networks"""
    connections = {}
    networks = {
        'ethereum': 'http://localhost:8545',  # Local Ganache
        'base': 'http://localhost:8546',      # Local Ganache
        'polygon': 'http://localhost:8547',   # Local Ganache
        'arbitrum': 'http://localhost:8548',  # Local Ganache
        'optimism': 'http://localhost:8549',  # Local Ganache
        'bnb': 'http://localhost:8550',       # Local Ganache
        'linea': 'http://localhost:8551',     # Local Ganache
        'mantle': 'http://localhost:8552',    # Local Ganache
        'avalanche': 'http://localhost:8553', # Local Ganache
        'gnosis': 'http://localhost:8554',    # Local Ganache
        'mode': 'http://localhost:8555',      # Local Ganache
        'sonic': 'http://localhost:8556'      # Local Ganache
    }
    
    for chain, url in networks.items():
        web3 = Web3(Web3.HTTPProvider(url))
        if web3.is_connected():
            connections[chain] = web3
    
    return connections

@pytest.fixture
def bridge_config():
    """Test bridge configuration"""
    return {
        'supported_chains': ['ethereum', 'base', 'polygon', 'mode', 'sonic'],
        'min_amount': 100.0,
        'max_amount': 1000000.0,
        'fee_multiplier': 1.0,
        'gas_limit_multiplier': 1.2,
        'confirmation_blocks': 1,
        'debridge_config': {
            'execution_fee_multiplier': 1.2,
            'claim_timeout': 7200,
            'auto_claim': True
        },
        'superbridge_config': {
            'lz_endpoint': '',
            'custom_adapter': '',
            'fee_tier': 'standard'
        },
        'across_config': {
            'relayer_fee_pct': 0.04,
            'lp_fee_pct': 0.02,
            'verification_gas_limit': 2000000
        },
        'mode_config': {
            'l1_bridge': '0x0000000000000000000000000000000000001010',
            'l2_bridge': '0x0000000000000000000000000000000000001010',
            'message_service': '0x0000000000000000000000000000000000001011'
        },
        'sonic_config': {
            'bridge_router': '0x0000000000000000000000000000000000001010',
            'token_bridge': '0x0000000000000000000000000000000000001011',
            'liquidity_pool': '0x0000000000000000000000000000000000001012'
        }
    }

@pytest.fixture
async def tx_builder(bridge_config):
    """Initialize transaction builder"""
    return CrossChainTransactionBuilder(bridge_config)

@pytest.fixture
async def chain_analyzer():
    """Initialize chain analyzer"""
    return CrossChainAnalyzer()

@pytest.mark.integration
class TestCrossChainIntegration:
    """Integration tests for cross-chain functionality"""
    
    @pytest.mark.asyncio
    async def test_bridge_registration(self):
        """Test registration of bridge adapters"""
        adapters = register_bridge_adapters()
        
        assert len(adapters) >= 5
        assert 'debridge' in adapters
        assert 'superbridge' in adapters
        assert 'across' in adapters
        assert 'mode' in adapters
        assert 'sonic' in adapters
    
    @pytest.mark.asyncio
    async def test_cross_chain_transfer_preparation(
        self,
        tx_builder: CrossChainTransactionBuilder,
        web3_connections: Dict[str, Web3]
    ):
        """Test preparation of cross-chain transfer"""
        opportunity = cast(CrossChainOpportunityType, {
            'source_chain': 'ethereum',
            'target_chain': 'base',
            'token_pair': ('USDC', 'USDC'),
            'amount': 1000.0,
            'recipient': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e'
        })
        
        result = await tx_builder.build_cross_chain_transaction(opportunity)
        
        assert result.success
        assert result.source_tx is not None
        assert isinstance(result.bridge_name, str)
        assert result.estimated_time > 0
        assert result.total_fee > 0
    
    @pytest.mark.asyncio
    async def test_bridge_analysis(
        self,
        chain_analyzer: CrossChainAnalyzer,
        web3_connections: Dict[str, Web3]
    ):
        """Test bridge analysis functionality"""
        token_pair = ('USDC', 'USDC')
        amount = 1000.0
        source_chain = 'ethereum'
        target_chain = 'base'
        
        analysis = await chain_analyzer._analyze_bridges(
            token_pair,
            amount,
            source_chain,
            target_chain
        )
        
        assert analysis['success']
        assert 'recommended_bridge' in analysis
        assert 'estimated_time' in analysis
        assert 'total_fee' in analysis
        assert analysis['total_fee'] > 0
    
    @pytest.mark.asyncio
    async def test_liquidity_monitoring(
        self,
        chain_analyzer: CrossChainAnalyzer,
        web3_connections: Dict[str, Web3]
    ):
        """Test liquidity monitoring across chains"""
        token_pairs = [('USDC', 'USDC'), ('USDT', 'USDT')]
        chains = ['ethereum', 'base', 'polygon']
        
        liquidity_data = await chain_analyzer.monitor_bridge_liquidity_parallel(
            token_pairs,
            chains
        )
        
        assert isinstance(liquidity_data, dict)
        for chain in chains:
            assert chain in liquidity_data
            for pair in token_pairs:
                pair_key = f"{pair[0]}_{pair[1]}"
                assert pair_key in liquidity_data[chain]
                assert liquidity_data[chain][pair_key] >= 0
    
    @pytest.mark.asyncio
    async def test_cross_chain_opportunity_analysis(
        self,
        chain_analyzer: CrossChainAnalyzer,
        web3_connections: Dict[str, Web3]
    ):
        """Test cross-chain opportunity analysis"""
        market_data = {
            'source_chain': 'ethereum',
            'target_chain': 'base',
            'market_data': await chain_analyzer.fetch_all_market_data()
        }
        
        opportunity = await chain_analyzer.analyze_cross_chain_opportunity(
            ('USDC', 'USDC'),
            1000.0,
            market_data
        )
        
        assert isinstance(opportunity, dict)
        assert 'is_viable' in opportunity
        assert 'profit_potential' in opportunity
        assert 'bridge_liquidity' in opportunity
        assert 'estimated_gas_cost' in opportunity
        assert 'execution_time' in opportunity
        assert 'risks' in opportunity
    
    @pytest.mark.asyncio
    async def test_transaction_validation(
        self,
        chain_analyzer: CrossChainAnalyzer,
        web3_connections: Dict[str, Web3]
    ):
        """Test transaction validation across chains"""
        transactions = [
            {
                'chain': 'ethereum',
                'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
                'value': 1000000000000000000,  # 1 ETH
                'gas': 21000
            },
            {
                'chain': 'base',
                'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
                'value': 500000000000000000,  # 0.5 ETH
                'gas': 21000
            }
        ]
        
        validation_results = await chain_analyzer.validate_transactions_parallel(
            transactions
        )
        
        assert isinstance(validation_results, list)
        assert len(validation_results) == len(transactions)
        for result in validation_results:
            assert 'is_valid' in result
            assert 'transaction' in result
            if not result['is_valid']:
                assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_performance_metrics(
        self,
        chain_analyzer: CrossChainAnalyzer,
        web3_connections: Dict[str, Web3]
    ):
        """Test performance metrics collection"""
        # Record some test metrics
        await chain_analyzer.record_opportunity_execution(
            chain='ethereum',
            execution_time=5.0,
            profit_realized=100.0,
            expected_profit=120.0,
            gas_used=150000,
            estimated_gas=140000,
            slippage=0.01,
            bridge_latency=60.0,
            success=True
        )
        
        # Get performance report
        report = chain_analyzer.get_performance_report()
        
        assert isinstance(report, dict)
        assert 'chain_metrics' in report
        assert 'operation_times' in report
        assert 'overall_health' in report
        
        # Check learning feedback
        feedback = chain_analyzer.get_learning_feedback()
        assert isinstance(feedback, dict)
        
        # Check recommendations
        recommendations = chain_analyzer.get_chain_recommendations('ethereum')
        assert isinstance(recommendations, list)
    
    @pytest.mark.asyncio
    async def test_mode_bridge_transfer(
        self,
        tx_builder: CrossChainTransactionBuilder,
        web3_connections: Dict[str, Web3]
    ):
        """Test Mode bridge transfer preparation"""
        opportunity = cast(CrossChainOpportunityType, {
            'source_chain': 'ethereum',
            'target_chain': 'mode',
            'token_pair': ('USDC', 'USDC'),
            'amount': 1000.0,
            'recipient': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e'
        })
        
        result = await tx_builder.build_cross_chain_transaction(opportunity)
        
        assert result.success
        assert result.source_tx is not None
        assert result.bridge_name == 'mode'
        assert result.estimated_time > 0
        assert result.total_fee > 0
    
    @pytest.mark.asyncio
    async def test_sonic_bridge_transfer(
        self,
        tx_builder: CrossChainTransactionBuilder,
        web3_connections: Dict[str, Web3]
    ):
        """Test Sonic bridge transfer preparation"""
        opportunity = cast(CrossChainOpportunityType, {
            'source_chain': 'ethereum',
            'target_chain': 'sonic',
            'token_pair': ('USDC', 'USDC'),
            'amount': 1000.0,
            'recipient': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e'
        })
        
        result = await tx_builder.build_cross_chain_transaction(opportunity)
        
        assert result.success
        assert result.source_tx is not None
        assert result.bridge_name == 'sonic'
        assert result.estimated_time > 0
        assert result.total_fee > 0
    
    @pytest.mark.asyncio
    async def test_mode_bridge_liquidity(
        self,
        chain_analyzer: CrossChainAnalyzer,
        web3_connections: Dict[str, Web3]
    ):
        """Test Mode bridge liquidity monitoring"""
        token_pairs = [('USDC', 'USDC'), ('ETH', 'ETH')]
        chains = ['ethereum', 'mode']
        
        liquidity_data = await chain_analyzer.monitor_bridge_liquidity_parallel(
            token_pairs,
            chains
        )
        
        assert isinstance(liquidity_data, dict)
        assert 'mode' in liquidity_data
        for pair in token_pairs:
            pair_key = f"{pair[0]}_{pair[1]}"
            assert pair_key in liquidity_data['mode']
            assert liquidity_data['mode'][pair_key] >= 0
    
    @pytest.mark.asyncio
    async def test_sonic_bridge_liquidity(
        self,
        chain_analyzer: CrossChainAnalyzer,
        web3_connections: Dict[str, Web3]
    ):
        """Test Sonic bridge liquidity monitoring"""
        token_pairs = [('USDC', 'USDC'), ('ETH', 'ETH')]
        chains = ['ethereum', 'sonic']
        
        liquidity_data = await chain_analyzer.monitor_bridge_liquidity_parallel(
            token_pairs,
            chains
        )
        
        assert isinstance(liquidity_data, dict)
        assert 'sonic' in liquidity_data
        for pair in token_pairs:
            pair_key = f"{pair[0]}_{pair[1]}"
            assert pair_key in liquidity_data['sonic']
            assert liquidity_data['sonic'][pair_key] >= 0
    
    @pytest.mark.asyncio
    async def test_mode_bridge_state(
        self,
        chain_analyzer: CrossChainAnalyzer,
        web3_connections: Dict[str, Web3]
    ):
        """Test Mode bridge state monitoring"""
        bridge_states = await chain_analyzer._analyze_bridges(
            ('USDC', 'USDC'),
            1000.0,
            'ethereum',
            'mode'
        )
        
        assert bridge_states['success']
        assert bridge_states['recommended_bridge'] == 'mode'
        assert bridge_states['estimated_time'] > 0
        assert bridge_states['total_fee'] > 0
    
    @pytest.mark.asyncio
    async def test_sonic_bridge_state(
        self,
        chain_analyzer: CrossChainAnalyzer,
        web3_connections: Dict[str, Web3]
    ):
        """Test Sonic bridge state monitoring"""
        bridge_states = await chain_analyzer._analyze_bridges(
            ('USDC', 'USDC'),
            1000.0,
            'ethereum',
            'sonic'
        )
        
        assert bridge_states['success']
        assert bridge_states['recommended_bridge'] == 'sonic'
        assert bridge_states['estimated_time'] > 0
        assert bridge_states['total_fee'] > 0
    
    @pytest.mark.asyncio
    async def test_all_chains_bridge_analysis(
        self,
        chain_analyzer: CrossChainAnalyzer,
        web3_connections: Dict[str, Web3]
    ):
        """Test bridge analysis for all supported chains"""
        test_cases = [
            ('ethereum', 'base'),
            ('ethereum', 'polygon'),
            ('ethereum', 'arbitrum'),
            ('ethereum', 'optimism'),
            ('ethereum', 'bnb'),
            ('ethereum', 'linea'),
            ('ethereum', 'mantle'),
            ('ethereum', 'avalanche'),
            ('ethereum', 'gnosis'),
            ('ethereum', 'mode'),
            ('ethereum', 'sonic'),
            ('base', 'optimism'),
            ('polygon', 'arbitrum'),
            ('bnb', 'polygon'),
            ('avalanche', 'polygon'),
            ('gnosis', 'ethereum')
        ]
        
        for source_chain, target_chain in test_cases:
            analysis = await chain_analyzer._analyze_bridges(
                ('USDC', 'USDC'),
                1000.0,
                source_chain,
                target_chain
            )
            
            assert analysis['success']
            assert 'recommended_bridge' in analysis
            assert 'estimated_time' in analysis
            assert 'fees' in analysis
            assert analysis['fees']['total'] > 0
    
    @pytest.mark.asyncio
    async def test_all_chains_liquidity_monitoring(
        self,
        chain_analyzer: CrossChainAnalyzer,
        web3_connections: Dict[str, Web3]
    ):
        """Test liquidity monitoring for all chains"""
        token_pairs = [('USDC', 'USDC'), ('ETH', 'ETH'), ('USDT', 'USDT')]
        chains = [
            'ethereum', 'base', 'polygon', 'arbitrum', 'optimism',
            'bnb', 'linea', 'mantle', 'avalanche', 'gnosis',
            'mode', 'sonic'
        ]
        
        liquidity_data = await chain_analyzer.monitor_bridge_liquidity_parallel(
            token_pairs,
            chains
        )
        
        assert isinstance(liquidity_data, dict)
        for chain in chains:
            assert chain in liquidity_data
            for pair in token_pairs:
                pair_key = f"{pair[0]}_{pair[1]}"
                assert pair_key in liquidity_data[chain]
                assert liquidity_data[chain][pair_key] >= 0
    
    @pytest.mark.asyncio
    async def test_all_chains_performance_metrics(
        self,
        chain_analyzer: CrossChainAnalyzer,
        web3_connections: Dict[str, Web3]
    ):
        """Test performance metrics for all chains"""
        chains = [
            'ethereum', 'base', 'polygon', 'arbitrum', 'optimism',
            'bnb', 'linea', 'mantle', 'avalanche', 'gnosis',
            'mode', 'sonic'
        ]
        
        for chain in chains:
            # Record test metrics for each chain
            await chain_analyzer.record_opportunity_execution(
                chain=chain,
                execution_time=5.0,
                profit_realized=100.0,
                expected_profit=120.0,
                gas_used=150000,
                estimated_gas=140000,
                slippage=0.01,
                bridge_latency=60.0,
                success=True,
                bridge_name=chain  # Use chain name as bridge name for testing
            )
        
        # Verify metrics were recorded
        for chain in chains:
            metrics = chain_analyzer.metrics.chain_metrics.get(chain)
            assert metrics is not None
            assert metrics.total_txs > 0
            assert metrics.success_rate > 0
            assert metrics.avg_execution_time > 0
            
            # Check chain-specific metrics
            if chain == 'mode':
                assert metrics.mode_bridge_usage > 0
                assert metrics.mode_gas_savings >= 0
            elif chain == 'sonic':
                assert metrics.sonic_bridge_volume > 0
                assert metrics.sonic_fee_savings >= 0
    
    @pytest.mark.asyncio
    async def test_all_chains_bridge_states(
        self,
        chain_analyzer: CrossChainAnalyzer,
        web3_connections: Dict[str, Web3]
    ):
        """Test bridge state monitoring for all chains"""
        test_cases = [
            ('ethereum', 'base'),
            ('ethereum', 'polygon'),
            ('ethereum', 'arbitrum'),
            ('ethereum', 'optimism'),
            ('ethereum', 'bnb'),
            ('ethereum', 'linea'),
            ('ethereum', 'mantle'),
            ('ethereum', 'avalanche'),
            ('ethereum', 'gnosis'),
            ('ethereum', 'mode'),
            ('ethereum', 'sonic')
        ]
        
        for source_chain, target_chain in test_cases:
            bridge_states = await chain_analyzer._analyze_bridges(
                ('USDC', 'USDC'),
                1000.0,
                source_chain,
                target_chain
            )
            
            assert bridge_states['success']
            assert 'recommended_bridge' in bridge_states
            assert bridge_states['estimated_time'] > 0
            assert bridge_states['fees']['total'] > 0
            
            # Check specific bridge state
            bridge_name = bridge_states['recommended_bridge']
            assert bridge_name in [
                'layerzero', 'debridge', 'superbridge', 'across',
                'mode', 'sonic', 'omni', 'plasma', 'pos'
            ] 