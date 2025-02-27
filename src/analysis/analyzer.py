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
        
        # Initialize circuit breakers for external services
        self.circuit_breakers = {
            'dex_client': {'failures': 0, 'threshold': 5, 'open': False, 'last_attempt': 0},
            'token_client': {'failures': 0, 'threshold': 5, 'open': False, 'last_attempt': 0},
            'network_client': {'failures': 0, 'threshold': 5, 'open': False, 'last_attempt': 0},
            'defi_client': {'failures': 0, 'threshold': 5, 'open': False, 'last_attempt': 0}
        }
        
    async def _call_with_circuit_breaker(self, service_name: str, call_fn, *args, **kwargs):
        """Call external API with circuit breaker pattern"""
        # Check if circuit is open (too many failures)
        breaker = self.circuit_breakers.get(service_name)
        if not breaker:
            logger.warning(f"No circuit breaker configured for {service_name}")
            return await call_fn(*args, **kwargs)
            
        if breaker['open']:
            # Circuit is open, check if we should retry
            if time.time() - breaker['last_attempt'] < 60:  # Wait at least 60s
                logger.warning(f"Circuit breaker open for {service_name}, request blocked")
                self.metrics.record_api_error(service_name, "circuit_open")
                return None

            # Try to reset circuit
            logger.info(f"Attempting to reset circuit breaker for {service_name}")
            breaker['open'] = False
            breaker['failures'] = 0
        
        try:
            # Make API call
            result = await call_fn(*args, **kwargs)
            
            # Reset failures on success
            breaker['failures'] = 0
            return result
            
        except Exception as e:
            # Increment failures
            breaker['failures'] += 1
            breaker['last_attempt'] = time.time()
            
            # Open circuit if threshold reached
            if breaker['failures'] >= breaker['threshold']:
                breaker['open'] = True
                logger.warning(f"Circuit breaker opened for {service_name} due to {str(e)}")
                self.metrics.record_api_error(service_name, "circuit_tripped")
                
            # Re-raise for caller to handle
            raise
        
    async def _fetch_dex_data(self, dex_id: str) -> Dict[str, Any]:
        """Fetch DEX data including pools, volumes, and liquidity"""
        cache_key = f"dex_data:{dex_id}"
        
        # Try cache first
        cached = await self.cache.get(cache_key)
        if cached:
            self.metrics.record_cache_hit()
            return cached
            
        try:
            # Use circuit breaker for API calls
            pools = await self._call_with_circuit_breaker(
                'dex_client', 
                self.dex_client.get_pools, 
                dex_id
            )
            
            volume = await self._call_with_circuit_breaker(
                'dex_client', 
                self.dex_client.get_volume, 
                dex_id
            )
            
            tvl = await self._call_with_circuit_breaker(
                'dex_client', 
                self.dex_client.get_tvl, 
                dex_id
            )
            
            fees = await self._call_with_circuit_breaker(
                'dex_client', 
                self.dex_client.get_fees, 
                dex_id
            )
            
            data = {
                'pools': pools,
                'volume_24h': volume,
                'tvl': tvl,
                'fees': fees,
                'timestamp': time.time()
            }
            
            # Cache the results
            await self.cache.set(cache_key, data)
            
            return data
        except Exception as e:
            logger.error(f"Error fetching DEX data for {dex_id}: {str(e)}")
            self.metrics.record_api_error("dex_data", str(e))
            return None
    
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
        """Get the current gas price on the network"""
        cache_key = "gas_price"
        
        # Try cache first with a short TTL
        cached = await self.cache.get(cache_key)
        if cached:
            self.metrics.record_cache_hit()
            return cached
            
        try:
            gas_price = await self.network_client.get_gas_price()
            
            # Cache for a short period
            await self.cache.set(cache_key, gas_price, ttl=60)  # 1 minute
            self.metrics.record_cache_miss()
            
            return gas_price
        except Exception as e:
            self.metrics.record_api_error('network', str(e))
            logger.error(f"Error getting gas price: {str(e)}")
            return 0
            
    # Flash Loan Arbitrage Analysis Methods
    
    async def analyze_flash_loan_opportunity(
        self, 
        token_address: str,
        amount: float,
        loan_source: str = 'aave',  # Options: aave, dydx, balancer
        route: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of flash loan arbitrage opportunities for a token
        
        Args:
            token_address: Address of the token to analyze
            amount: Amount of tokens to use in the flash loan
            loan_source: Flash loan provider to use
            route: Optional list of exchange addresses to use in route
            
        Returns:
            Dictionary with detailed opportunity analysis
        """
        # Record start time for performance metrics
        start_time = time.time()
        
        # Get token data 
        token_data = await self._fetch_token_data(token_address)
        if not token_data:
            return {
                'viable': False,
                'reason': 'Token data not available',
                'details': {}
            }
            
        # Get current gas price
        gas_price = await self._get_current_gas_price()
        
        # Determine exchange sources based on provided route or defaults
        exchanges = route or self.config.get('default_exchanges', [
            'uniswap_v3', 'sushiswap', 'curve'
        ])
        
        # Gather price data from all exchanges
        exchange_prices = {}
        for exchange in exchanges:
            try:
                price = await self._get_dex_price(token_address, exchange)
                exchange_prices[exchange] = price
            except Exception as e:
                logger.debug(f"Could not get price from {exchange}: {str(e)}")
                
        if len(exchange_prices) < 2:
            return {
                'viable': False,
                'reason': 'Insufficient exchange data',
                'details': {
                    'exchange_prices': exchange_prices
                }
            }
            
        # Calculate price discrepancies and best arbitrage route
        arbitrage_routes = await self._calculate_arbitrage_routes(
            token_address,
            amount,
            exchange_prices
        )
        
        # Calculate flash loan fees
        loan_fee = await self._calculate_flash_loan_fee(amount, loan_source)
        
        # Estimate gas costs for the arbitrage
        gas_cost = await self._estimate_arbitrage_gas_cost(
            token_address,
            amount,
            arbitrage_routes[0] if arbitrage_routes else None,
            loan_source,
            gas_price
        )
        
        # Calculate potential profit
        best_route = arbitrage_routes[0] if arbitrage_routes else None
        if best_route:
            profit = best_route['profit'] - loan_fee - gas_cost
            
            result = {
                'viable': profit > 0,
                'profit': profit,
                'token_address': token_address,
                'amount': amount,
                'loan_source': loan_source,
                'loan_fee': loan_fee,
                'gas_cost': gas_cost,
                'gas_price': gas_price,
                'best_route': best_route,
                'all_routes': arbitrage_routes[:5],  # Top 5 routes only
                'exchange_prices': exchange_prices,
                'token_data': {
                    'symbol': token_data.get('symbol'),
                    'decimals': token_data.get('decimals'),
                    'liquidity': token_data.get('liquidity', {})
                },
                'analysis_time': time.time() - start_time
            }
        else:
            result = {
                'viable': False,
                'reason': 'No profitable arbitrage routes found',
                'details': {
                    'exchange_prices': exchange_prices,
                    'loan_fee': loan_fee,
                    'gas_cost': gas_cost,
                }
            }
            
        return result
            
    async def _calculate_arbitrage_routes(
        self,
        token_address: str,
        amount: float,
        exchange_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Calculate all possible arbitrage routes and their profitability"""
        routes = []
        
        # Find all pairs of exchanges with price differences
        exchanges = list(exchange_prices.keys())
        
        for buy_exchange in exchanges:
            buy_price = exchange_prices[buy_exchange]
            
            for sell_exchange in exchanges:
                if buy_exchange == sell_exchange:
                    continue
                    
                sell_price = exchange_prices[sell_exchange]
                
                # Calculate potential profit (before fees and gas)
                price_diff = sell_price - buy_price
                profit_potential = price_diff * amount
                
                if profit_potential > 0:
                    # Get liquidity data to ensure the trade is possible
                    buy_liquidity = await self._get_dex_liquidity(token_address, buy_exchange)
                    sell_liquidity = await self._get_dex_liquidity(token_address, sell_exchange)
                    
                    # Calculate slippage based on liquidity
                    buy_slippage = min(1.0, (amount / buy_liquidity) * 0.1) if buy_liquidity else 1.0
                    sell_slippage = min(1.0, (amount / sell_liquidity) * 0.1) if sell_liquidity else 1.0
                    
                    # Adjust profit for slippage
                    adjusted_buy_price = buy_price * (1 + buy_slippage)
                    adjusted_sell_price = sell_price * (1 - sell_slippage)
                    adjusted_price_diff = adjusted_sell_price - adjusted_buy_price
                    adjusted_profit = adjusted_price_diff * amount
                    
                    if adjusted_profit > 0:
                        routes.append({
                            'buy_exchange': buy_exchange,
                            'sell_exchange': sell_exchange,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'price_diff': price_diff,
                            'profit': adjusted_profit,
                            'buy_slippage': buy_slippage,
                            'sell_slippage': sell_slippage,
                            'buy_liquidity': buy_liquidity,
                            'sell_liquidity': sell_liquidity
                        })
        
        # Sort routes by profit (highest first)
        routes.sort(key=lambda x: x['profit'], reverse=True)
        
        return routes
        
    async def _calculate_flash_loan_fee(
        self,
        amount: float,
        loan_source: str
    ) -> float:
        """Calculate the flash loan fee for a given amount and provider"""
        fee_rates = {
            'aave': 0.0009,      # 0.09%
            'dydx': 0.0,         # 0% (gas only)
            'balancer': 0.0006,  # 0.06%
            'uniswap': 0.0005,   # 0.05%
            'maker': 0.0008,     # 0.08%
        }
        
        fee_rate = fee_rates.get(loan_source.lower(), 0.001)  # Default 0.1%
        return amount * fee_rate
        
    async def _estimate_arbitrage_gas_cost(
        self,
        token_address: str,
        amount: float,
        route: Optional[Dict[str, Any]],
        loan_source: str,
        gas_price: int
    ) -> float:
        """Estimate the gas cost for an arbitrage transaction"""
        if not route:
            return 0
            
        # Base gas costs for different operations
        gas_estimates = {
            'flash_loan_borrow': {
                'aave': 300000,
                'dydx': 400000,
                'balancer': 250000,
                'uniswap': 200000,
                'maker': 350000
            },
            'exchange_swap': {
                'uniswap_v2': 150000,
                'uniswap_v3': 180000,
                'sushiswap': 150000,
                'curve': 200000,
                'balancer': 160000
            },
            'flash_loan_repay': 80000,
            'overhead': 50000
        }
        
        # Calculate total gas estimate
        loan_gas = gas_estimates['flash_loan_borrow'].get(loan_source.lower(), 350000)
        buy_gas = gas_estimates['exchange_swap'].get(route['buy_exchange'], 180000)
        sell_gas = gas_estimates['exchange_swap'].get(route['sell_exchange'], 180000)
        repay_gas = gas_estimates['flash_loan_repay']
        overhead_gas = gas_estimates['overhead']
        
        total_gas = loan_gas + buy_gas + sell_gas + repay_gas + overhead_gas
        
        # Convert gas to ETH cost
        gas_price_in_eth = gas_price / 1e9  # Convert from wei to gwei
        eth_cost = total_gas * gas_price_in_eth / 1e9  # Convert to ETH
        
        # Get ETH price to convert to USD
        eth_price = await self._get_aggregator_price('0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2')  # WETH address
        usd_cost = eth_cost * eth_price
        
        return usd_cost
        
    # Yield Farming Analysis Methods
    
    async def analyze_yield_farming_opportunities(
        self,
        token_address: str,
        amount: float,
        time_horizon: int = 30  # days
    ) -> Dict[str, Any]:
        """
        Analyze yield farming opportunities for a given token
        
        Args:
            token_address: Address of token to analyze
            amount: Amount of tokens to farm
            time_horizon: Investment time horizon in days
            
        Returns:
            Dictionary with detailed yield farming analysis
        """
        # Record start time for performance metrics
        start_time = time.time()
        
        # Get token data
        token_data = await self._fetch_token_data(token_address)
        if not token_data:
            return {
                'viable': False,
                'reason': 'Token data not available',
                'details': {}
            }
            
        # Get current gas price
        gas_price = await self._get_current_gas_price()
        
        # Get network data
        network_data = await self._fetch_network_data()
        
        # Fetch DeFi protocol data from DefiLlama
        try:
            protocols = await self._call_with_circuit_breaker(
                'defi_client',
                self.defi_client.get_yields
            )
        except Exception as e:
            logger.error(f"Error fetching protocol data: {str(e)}")
            protocols = []
            
        # Find protocols that support this token
        compatible_protocols = []
        for protocol in protocols:
            if await self._protocol_supports_token(protocol, token_address):
                compatible_protocols.append(protocol)
                
        if not compatible_protocols:
            return {
                'viable': False,
                'reason': 'No compatible yield farming protocols found',
                'details': {
                    'token': token_data.get('symbol', token_address)
                }
            }
            
        # Analyze each protocol's yield opportunities
        opportunities = []
        for protocol in compatible_protocols:
            try:
                yields = await self._analyze_protocol_yields(
                    protocol,
                    token_address,
                    amount,
                    time_horizon
                )
                if yields:
                    opportunities.append(yields)
            except Exception as e:
                logger.error(f"Error analyzing {protocol['name']} yields: {str(e)}")
                
        # Sort opportunities by adjusted APY (highest first)
        opportunities.sort(key=lambda x: x['adjusted_apy'], reverse=True)
        
        result = {
            'viable': len(opportunities) > 0,
            'token_address': token_address,
            'token_data': {
                'symbol': token_data.get('symbol'),
                'decimals': token_data.get('decimals'),
                'price': token_data.get('price', 0)
            },
            'amount': amount,
            'time_horizon': time_horizon,
            'opportunities': opportunities,
            'gas_price': gas_price,
            'network_congestion': network_data.get('congestion', 0),
            'analysis_time': time.time() - start_time
        }
        
        return result
        
    async def _protocol_supports_token(
        self,
        protocol: Dict[str, Any],
        token_address: str
    ) -> bool:
        """Check if a protocol supports a specific token"""
        # This would require protocol-specific logic
        # For now, we'll implement a simplified version
        
        # Check if protocol has tokens list
        if 'tokens' in protocol:
            return token_address.lower() in [t.lower() for t in protocol['tokens']]
            
        # For some protocols, we need to check pools
        if 'pools' in protocol:
            for pool in protocol['pools']:
                if 'tokens' in pool and token_address.lower() in [t.lower() for t in pool['tokens']]:
                    return True
                    
        # Default to checking by querying the protocol directly
        try:
            supported = await self._call_with_circuit_breaker(
                'defi_client',
                self.defi_client.check_token_support,
                protocol['id'],
                token_address
            )
            return supported
        except Exception:
            # If we can't determine, assume not supported
            return False
            
    async def _analyze_protocol_yields(
        self,
        protocol: Dict[str, Any],
        token_address: str,
        amount: float,
        time_horizon: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze yield farming opportunities in a specific protocol"""
        try:
            # Get protocol APY data
            apy_data = await self._call_with_circuit_breaker(
                'defi_client',
                self.defi_client.get_protocol_apy,
                protocol['id']
            )
            
            if not apy_data:
                return None
                
            # Find the best pool for this token
            best_pool = None
            best_apy = 0
            
            for pool in apy_data.get('pools', []):
                if token_address.lower() in [t.lower() for t in pool.get('tokens', [])]:
                    pool_apy = pool.get('apy', 0)
                    if pool_apy > best_apy:
                        best_apy = pool_apy
                        best_pool = pool
                        
            if not best_pool:
                return None
                
            # Calculate entry/exit costs
            gas_price = await self._get_current_gas_price()
            entry_cost = await self._estimate_farming_entry_cost(protocol, token_address, gas_price)
            exit_cost = await self._estimate_farming_exit_cost(protocol, token_address, gas_price)
            
            # Calculate impermanent loss risk
            il_risk = await self._calculate_impermanent_loss_risk(
                protocol,
                best_pool,
                token_address,
                time_horizon
            )
            
            # Calculate projected earnings
            apy = best_pool.get('apy', 0)
            daily_rate = apy / 365
            earnings = amount * (1 + daily_rate) ** time_horizon - amount
            
            # Adjust for costs and impermanent loss
            adjusted_earnings = earnings - entry_cost - exit_cost - (earnings * il_risk)
            adjusted_apy = (adjusted_earnings / amount) * (365 / time_horizon)
            
            return {
                'protocol': protocol['name'],
                'protocol_id': protocol['id'],
                'pool': best_pool.get('name', 'Unknown Pool'),
                'pool_id': best_pool.get('id'),
                'apy': apy,
                'adjusted_apy': adjusted_apy,
                'projected_earnings': earnings,
                'adjusted_earnings': adjusted_earnings,
                'entry_cost': entry_cost,
                'exit_cost': exit_cost,
                'impermanent_loss_risk': il_risk,
                'token_share': await self._calculate_token_share(best_pool, token_address),
                'rewards': best_pool.get('rewards', []),
                'tvl': best_pool.get('tvl', 0),
                'strategy': await self._generate_farming_strategy(protocol, best_pool, token_address)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing protocol yields: {str(e)}")
            return None
            
    async def _estimate_farming_entry_cost(
        self,
        protocol: Dict[str, Any],
        token_address: str,
        gas_price: int
    ) -> float:
        """Estimate the cost to enter a farming position"""
        # Simplified implementation
        # This would be protocol-specific in reality
        
        # Base gas usage for common operations
        gas_estimates = {
            'approve': 45000,
            'single_stake': 150000,
            'lp_provide': 200000,
            'compound': 180000,
            # Protocol-specific overrides
            'curve': 250000,
            'yearn': 220000,
            'convex': 280000
        }
        
        # Get protocol-specific estimate or use default
        protocol_id = protocol.get('id', '').lower()
        gas_estimate = gas_estimates.get(protocol_id, gas_estimates['single_stake'])
        
        # Adjust based on current gas price
        gas_price_in_eth = gas_price / 1e9  # Convert from wei to gwei
        eth_cost = gas_estimate * gas_price_in_eth / 1e9  # Convert to ETH
        
        # Get ETH price
        eth_price = await self._get_aggregator_price('0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2')
        usd_cost = eth_cost * eth_price
        
        return usd_cost
        
    async def _estimate_farming_exit_cost(
        self,
        protocol: Dict[str, Any],
        token_address: str,
        gas_price: int
    ) -> float:
        """Estimate the cost to exit a farming position"""
        # Similar to entry cost but ~20% less gas usually
        entry_cost = await self._estimate_farming_entry_cost(protocol, token_address, gas_price)
        return entry_cost * 0.8  # Exit is usually a bit cheaper than entry
        
    async def _calculate_impermanent_loss_risk(
        self,
        protocol: Dict[str, Any],
        pool: Dict[str, Any],
        token_address: str,
        time_horizon: int
    ) -> float:
        """Calculate impermanent loss risk for a pool"""
        # Simple model - could be much more sophisticated with price history
        
        # If single-asset pool, no IL
        if len(pool.get('tokens', [])) == 1:
            return 0.0
            
        # Get token volatility
        try:
            token_data = await self._fetch_token_data(token_address)
            volatility = token_data.get('volatility_30d', 0.05)  # Default 5%
        except Exception:
            volatility = 0.05  # Default if we can't get data
            
        # Simple IL model based on volatility and pool type
        if 'stablecoin' in pool.get('type', '').lower():
            # Stablecoin pools have lower IL risk
            return volatility * 0.1 * min(time_horizon / 30, 1)
        else:
            # Regular pools have higher IL risk
            return volatility * 0.5 * min(time_horizon / 30, 1)
            
    async def _calculate_token_share(
        self,
        pool: Dict[str, Any],
        token_address: str
    ) -> float:
        """Calculate the share of the token in the pool"""
        tokens = pool.get('tokens', [])
        if not tokens:
            return 1.0  # Default to 100% if no data
            
        if len(tokens) == 1:
            return 1.0  # Single token pool
            
        # Get token weights if available
        weights = pool.get('weights', [])
        if weights and len(weights) == len(tokens):
            token_idx = [t.lower() for t in tokens].index(token_address.lower())
            return weights[token_idx]
            
        # Default to equal weighting
        return 1.0 / len(tokens)
        
    async def _generate_farming_strategy(
        self,
        protocol: Dict[str, Any],
        pool: Dict[str, Any],
        token_address: str
    ) -> Dict[str, Any]:
        """Generate a step-by-step farming strategy"""
        protocol_name = protocol.get('name', 'Unknown')
        pool_name = pool.get('name', 'Unknown')
        
        # Generic strategy steps
        strategy = {
            'name': f"{protocol_name} {pool_name} Strategy",
            'steps': [
                {
                    'step': 1,
                    'action': f"Approve {protocol_name} contract to use your tokens",
                    'details': f"Call approve() on the token contract at {token_address}"
                },
                {
                    'step': 2,
                    'action': f"Deposit tokens into {pool_name}",
                    'details': f"Call deposit() on the pool contract at {pool.get('address', 'Unknown')}"
                }
            ],
            'harvesting': {
                'frequency': 'Daily',
                'action': "Harvest rewards",
                'gas_efficiency': "More efficient to harvest less frequently when gas prices are high"
            },
            'risks': [
                "Smart contract risk",
                "Impermanent loss risk" if len(pool.get('tokens', [])) > 1 else None,
                "APY fluctuation risk",
                "Protocol risk"
            ],
            'links': {
                'protocol': protocol.get('url'),
                'docs': protocol.get('docs_url'),
                'pool': pool.get('url')
            }
        }
        
        # Filter out None values from risks
        strategy['risks'] = [r for r in strategy['risks'] if r is not None]
        
        return strategy 