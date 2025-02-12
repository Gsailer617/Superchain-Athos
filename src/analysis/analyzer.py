class MarketAnalyzer:
    """Market data analysis and price discovery"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        cache: AsyncCache,
        metrics: MetricsManager
    ):
        self.config = config
        self.cache = cache
        self.metrics = metrics
        self.dex_client = DEXClient(config['dex_api_url'])
        self.token_client = TokenClient(config['token_api_url'])
        self.network_client = NetworkClient(config['network_api_url'])
        self.defi_client = DefiLlamaClient(config['defillama_api_url'])
        
    async def _fetch_dex_data(self, dex_id: str) -> Dict[str, Any]:
        """Fetch DEX data including pools, volumes, and liquidity"""
        cache_key = f"dex_data:{dex_id}"
        
        # Try cache first
        cached = await self.cache.get(cache_key)
        if cached:
            self.metrics.record_cache_hit()
            return cached
            
        try:
            # Fetch pools data
            pools = await self.dex_client.get_pools(dex_id)
            
            # Fetch 24h volume
            volume = await self.dex_client.get_volume(dex_id)
            
            # Fetch TVL
            tvl = await self.dex_client.get_tvl(dex_id)
            
            # Fetch fee data
            fees = await self.dex_client.get_fees(dex_id)
            
            data = {
                'pools': pools,
                'volume_24h': volume,
                'tvl': tvl,
                'fees': fees,
                'timestamp': time.time()
            }
            
            # Cache the results
            await self.cache.set(cache_key, data)
            self.metrics.record_cache_miss()
            
            return data
            
        except Exception as e:
            self.metrics.record_api_error('dex', str(e))
            logger.error(f"Error fetching DEX data: {str(e)}")
            return {}
    
    async def _fetch_token_data(self, token_address: str) -> Dict[str, Any]:
        """Fetch token data including price, supply, and holders"""
        cache_key = f"token_data:{token_address}"
        
        cached = await self.cache.get(cache_key)
        if cached:
            self.metrics.record_cache_hit()
            return cached
            
        try:
            # Fetch token info
            info = await self.token_client.get_token_info(token_address)
            
            # Fetch holder data
            holders = await self.token_client.get_holders(token_address)
            
            # Fetch transfers
            transfers = await self.token_client.get_transfers(
                token_address,
                limit=100
            )
            
            # Get price data
            prices = await self.token_client.get_price_history(
                token_address,
                days=7
            )
            
            data = {
                'info': info,
                'holders': holders,
                'transfers': transfers,
                'price_history': prices,
                'timestamp': time.time()
            }
            
            await self.cache.set(cache_key, data)
            self.metrics.record_cache_miss()
            
            return data
            
        except Exception as e:
            self.metrics.record_api_error('token', str(e))
            logger.error(f"Error fetching token data: {str(e)}")
            return {}
    
    async def _fetch_network_data(self) -> Dict[str, Any]:
        """Fetch network data including gas, blocks, and mempool"""
        cache_key = "network_data"
        
        cached = await self.cache.get(cache_key)
        if cached:
            self.metrics.record_cache_hit()
            return cached
            
        try:
            # Get latest block data
            block_data = await self.network_client.get_latest_block()
            
            # Get gas price data
            gas_data = await self.network_client.get_gas_prices()
            
            # Get mempool data
            mempool = await self.network_client.get_mempool()
            
            # Get network stats
            stats = await self.network_client.get_network_stats()
            
            data = {
                'latest_block': block_data,
                'gas_prices': gas_data,
                'mempool': mempool,
                'network_stats': stats,
                'timestamp': time.time()
            }
            
            await self.cache.set(cache_key, data)
            self.metrics.record_cache_miss()
            
            return data
            
        except Exception as e:
            self.metrics.record_api_error('network', str(e))
            logger.error(f"Error fetching network data: {str(e)}")
            return {}
    
    async def _get_dex_price(
        self,
        token_address: str,
        dex_id: str
    ) -> float:
        """Get token price from a specific DEX"""
        try:
            # Get pool data
            pool_data = await self.dex_client.get_token_pools(
                token_address,
                dex_id
            )
            
            if not pool_data or 'price' not in pool_data:
                return 0.0
                
            # Record metrics
            self.metrics.observe(
                'dex_price',
                pool_data['price'],
                {'dex': dex_id}
            )
            
            return float(pool_data['price'])
            
        except Exception as e:
            self.metrics.record_api_error('dex_price', str(e))
            logger.error(f"Error getting DEX price: {str(e)}")
            return 0.0
    
    async def _get_aggregator_price(
        self,
        token_address: str
    ) -> float:
        """Get token price from aggregator APIs"""
        try:
            # Try multiple price sources
            prices = await asyncio.gather(
                self.defi_client.get_token_price(token_address),
                self.token_client.get_current_price(token_address),
                return_exceptions=True
            )
            
            valid_prices = [
                p for p in prices
                if isinstance(p, (int, float)) and p > 0
            ]
            
            if not valid_prices:
                return 0.0
                
            # Use median price
            price = statistics.median(valid_prices)
            
            # Record metrics
            self.metrics.observe('aggregator_price', price)
            
            return float(price)
            
        except Exception as e:
            self.metrics.record_api_error('aggregator_price', str(e))
            logger.error(f"Error getting aggregator price: {str(e)}")
            return 0.0
    
    async def _get_cached_price(
        self,
        token_address: str,
        max_age: int = 300
    ) -> float:
        """Get cached token price if available and fresh"""
        try:
            cache_key = f"token_price:{token_address}"
            cached = await self.cache.get(cache_key)
            
            if not cached:
                return 0.0
                
            # Check if price is still fresh
            age = time.time() - cached['timestamp']
            if age > max_age:
                return 0.0
                
            return float(cached['price'])
            
        except Exception as e:
            logger.error(f"Error getting cached price: {str(e)}")
            return 0.0
    
    async def _get_dex_liquidity(
        self,
        token_address: str,
        dex_id: str
    ) -> float:
        """Get token liquidity on a specific DEX"""
        try:
            # Get pool data
            pool_data = await self.dex_client.get_token_pools(
                token_address,
                dex_id
            )
            
            if not pool_data or 'liquidity' not in pool_data:
                return 0.0
                
            # Record metrics
            self.metrics.observe(
                'dex_liquidity',
                pool_data['liquidity'],
                {'dex': dex_id}
            )
            
            return float(pool_data['liquidity'])
            
        except Exception as e:
            self.metrics.record_api_error('dex_liquidity', str(e))
            logger.error(f"Error getting DEX liquidity: {str(e)}")
            return 0.0
    
    async def _get_average_block_time(self) -> float:
        """Get average block time over last N blocks"""
        try:
            # Get latest blocks
            blocks = await self.network_client.get_latest_blocks(100)
            
            if not blocks:
                return 0.0
                
            # Calculate average time between blocks
            times = [
                b['timestamp']
                for b in blocks
                if 'timestamp' in b
            ]
            
            if len(times) < 2:
                return 0.0
                
            diffs = [
                t2 - t1
                for t1, t2 in zip(times[:-1], times[1:])
            ]
            
            avg_time = statistics.mean(diffs)
            
            # Record metrics
            self.metrics.observe('average_block_time', avg_time)
            
            return float(avg_time)
            
        except Exception as e:
            self.metrics.record_api_error('block_time', str(e))
            logger.error(f"Error getting average block time: {str(e)}")
            return 0.0
    
    async def _get_network_load(self) -> float:
        """Get current network load (0-1)"""
        try:
            # Get network stats
            stats = await self.network_client.get_network_stats()
            
            if not stats:
                return 0.0
                
            # Calculate load based on gas usage and pending txs
            gas_usage = stats.get('gas_used_percent', 0)
            pending_ratio = min(
                1.0,
                stats.get('pending_tx_count', 0) / 10000
            )
            
            # Combine metrics
            load = (gas_usage + pending_ratio) / 2
            
            # Record metrics
            self.metrics.observe('network_load', load)
            
            return float(load)
            
        except Exception as e:
            self.metrics.record_api_error('network_load', str(e))
            logger.error(f"Error getting network load: {str(e)}")
            return 0.0
    
    async def _get_pending_transactions(self) -> int:
        """Get number of pending transactions"""
        try:
            # Get mempool data
            mempool = await self.network_client.get_mempool()
            
            if not mempool:
                return 0
                
            count = mempool.get('pending_count', 0)
            
            # Record metrics
            self.metrics.observe('pending_transactions', count)
            
            return int(count)
            
        except Exception as e:
            self.metrics.record_api_error('pending_tx', str(e))
            logger.error(f"Error getting pending transactions: {str(e)}")
            return 0
    
    async def _get_current_gas_price(self) -> int:
        """Get current gas price in gwei"""
        try:
            # Get gas price data
            gas_data = await self.network_client.get_gas_prices()
            
            if not gas_data:
                return 0
                
            # Get fast gas price
            price = gas_data.get('fast', 0)
            
            # Record metrics
            self.metrics.observe('gas_price', price)
            
            return int(price)
            
        except Exception as e:
            self.metrics.record_api_error('gas_price', str(e))
            logger.error(f"Error getting gas price: {str(e)}")
            return 0 