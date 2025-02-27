// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@aave/core-v3/contracts/flashloan/base/FlashLoanSimpleReceiverBase.sol";
import "@aave/core-v3/contracts/interfaces/IPoolAddressesProvider.sol";
import "@balancer-labs/v2-interfaces/contracts/vault/IVault.sol";
import "@balancer-labs/v2-interfaces/contracts/vault/IFlashLoanRecipient.sol";
import "./SwapHelper.sol";

interface IUniswapV2Router {
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
}

interface IUniswapV3Router {
    struct ExactInputParams {
        bytes path;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
    }

    function exactInput(ExactInputParams calldata params) external payable returns (uint256 amountOut);
}

interface IMaverickRouter {
    function exactInput(
        bytes calldata path,
        address recipient,
        uint256 deadline,
        uint256 amountIn,
        uint256 amountOutMinimum
    ) external returns (uint256 amountOut);
}

// Radiant Flash Loan Interface
interface IRadiantFlashLoan {
    function flashLoan(
        address receiverAddress,
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata modes,
        address onBehalfOf,
        bytes calldata params,
        uint16 referralCode
    ) external;
}

// Add missing protocol interfaces
interface IVelodromeRouter {
    function swapExactTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
}

interface ICurvePool {
    function exchange(int128 i, int128 j, uint256 dx, uint256 min_dy) external returns (uint256);
    function get_dy(int128 i, int128 j, uint256 dx) external view returns (uint256);
}

// Add Curve interface
interface ICurveRouter {
    function exchange(
        address pool,
        address from,
        address to,
        uint256 amount,
        uint256 expected,
        address to
    ) external payable returns (uint256);
    
    function get_dy(
        address pool,
        address from,
        address to,
        uint256 amount
    ) external view returns (uint256);
}

// Add Morpho interface
interface IMorphoRouter {
    function supply(
        address loanToken,
        uint256 amount,
        address onBehalfOf,
        uint16 referralCode
    ) external;
    
    function withdraw(
        address asset,
        uint256 amount,
        address to
    ) external returns (uint256);
    
    function flashLoan(
        address receiverAddress,
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata modes,
        address onBehalfOf,
        bytes calldata params,
        uint16 referralCode
    ) external;
}

contract FlashLoanArbitrage is 
    Initializable, 
    UUPSUpgradeable, 
    OwnableUpgradeable, 
    ReentrancyGuardUpgradeable,
    FlashLoanSimpleReceiverBase,
    IFlashLoanRecipient
{
    using SafeERC20 for IERC20;
    using SwapHelper for *;

    // Constants
    uint256 private constant DEADLINE_BUFFER = 180;  // 3 minutes
    uint256 private constant MAX_BPS = 10000;
    uint256 private constant GRACE_PERIOD = 300;  // 5 minutes for emergency actions
    
    // Struct packing for gas optimization
    struct ProviderStats {
        uint128 reliability;     // Reliability score (0-100)
        uint128 successRate;     // Success rate (0-100)
        uint96 avgGasUsed;      // Average gas used
        uint32 executionCount;  // Number of executions
        bool enabled;           // Provider status
    }
    
    // Enhanced Flash Loan Provider Configuration
    struct FlashLoanConfig {
        address provider;
        address router;
        uint128 minAmount;
        uint128 fee;
        uint128 maxAmount;         // Maximum amount that can be borrowed
        uint128 utilizationRate;   // Current utilization rate
        uint64 cooldownPeriod;     // Cooldown period between loans
        uint64 lastLoanTimestamp;  // Timestamp of last loan
        bool enabled;
        ProviderStats stats;
    }

    // Enhanced security settings
    struct SecuritySettings {
        uint256 maxSlippagePercent;     // Maximum allowed slippage (in basis points)
        uint256 maxGasPrice;            // Maximum allowed gas price
        uint256 minProfitRatio;         // Minimum profit ratio required
        uint256 maxPositionSize;        // Maximum position size per trade
        uint256 emergencyTimelock;      // Timelock for emergency actions
        uint256 maxDailyVolume;         // Maximum daily volume
        uint256 maxTxCount;             // Maximum transactions per day
        uint256 cooldownPeriod;         // Cooldown between transactions
        mapping(address => bool) trustedTokens;  // O(1) lookup for trusted tokens
        mapping(address => bool) blockedAddresses;  // Blocked addresses
        mapping(address => uint256) dailyVolumes;   // Track daily volumes
        mapping(address => uint256) txCounts;       // Track transaction counts
        mapping(address => uint256) lastTxTime;     // Last transaction timestamp
    }

    // Protocol health monitoring
    struct ProtocolHealth {
        uint256 totalValueLocked;    // Total value locked
        uint256 dailyVolume;         // 24h volume
        uint256 utilizationRate;     // Current utilization rate
        uint256 lastUpdateTime;      // Last update timestamp
        bool isHealthy;              // Overall health status
        // Add configurable thresholds
        uint256 minTvlThreshold;     // Minimum TVL required
        uint256 maxUtilization;      // Maximum utilization rate (in basis points)
        uint256 minDailyVolume;      // Minimum daily volume required
        uint256 healthCheckInterval; // Time between health checks
    }

    // Events with indexed parameters for efficient filtering
    event FlashLoanExecuted(
        FlashLoanProvider indexed provider,
        address indexed token,
        uint256 amount,
        uint256 fee
    );
    
    // Storage optimization - pack related variables
    struct ExecutionState {
        bool isPaused;
        uint64 lastExecutionTime;
        uint64 emergencyTimeout;
        uint128 minProfitThreshold;
    }
    ExecutionState private state;
    
    // Optimized mappings
    mapping(FlashLoanProvider => FlashLoanConfig) private flashLoanConfigs;
    mapping(bytes32 => uint256) private executionNonces;
    mapping(address => bool) private whitelistedCallers;
    
    // Modifiers with reentrancy protection
    modifier whenNotPaused() {
        require(!state.isPaused, "Contract is paused");
        _;
    }
    
    modifier onlyWhitelisted() {
        require(whitelistedCallers[msg.sender], "Caller not whitelisted");
        _;
    }
    
    modifier withDeadline(uint256 deadline) {
        require(deadline >= block.timestamp + DEADLINE_BUFFER, "Deadline too short");
        require(deadline <= block.timestamp + 1 hours, "Deadline too long");
        _;
    }
    
    // Flash Loan Provider enum
    enum FlashLoanProvider { AAVE, BALANCER, RADIANT }

    // DEX Types
    enum DexType { 
        UniswapV2, 
        UniswapV3, 
        Maverick, 
        Curve, 
        Velodrome,
        GMX,
        Morpho
    }
    
    // Enhanced DEX Configuration with order book depth and ranking
    struct DexConfig {
        address router;
        DexType dexType;
        uint24[] feeTiers;
        bool enabled;
        bool isStablePair;
        uint128 rankingScore;      // Dynamic ranking score (0-10000)
        uint128 depthScore;        // Liquidity depth score (0-10000)
        uint64 lastUpdateTime;     // Last ranking update timestamp
        uint32 successfulTrades;   // Number of successful trades
        uint32 failedTrades;       // Number of failed trades
    }

    // Order Book Depth Analysis
    struct OrderBookDepth {
        uint128 bidDepth;      // Total bid liquidity
        uint128 askDepth;      // Total ask liquidity
        uint64 lastUpdateTime; // Last update timestamp
        uint32 updateCount;    // Number of updates
        mapping(uint24 => uint256) feeTierLiquidity; // Liquidity per fee tier
    }

    // Protocol Performance Tracking
    struct ProtocolPerformance {
        uint128 reliabilityScore;  // Protocol reliability (0-10000)
        uint128 utilizationRate;   // Current utilization rate
        uint96 avgGasUsed;        // Average gas used
        uint96 avgProfitRatio;    // Average profit ratio
        uint32 totalExecutions;   // Total number of executions
        uint32 successfulExecs;   // Successful executions
        bool active;              // Protocol active status
    }

    // Storage for new tracking mechanisms
    mapping(string => OrderBookDepth) private orderBookDepths;
    mapping(string => ProtocolPerformance) private protocolPerformance;
    mapping(address => uint256) private tokenLiquidityScores;

    // Events for tracking
    event DexRankingUpdated(string indexed dex, uint256 newRanking);
    event ProtocolPerformanceUpdated(string indexed protocol, uint256 reliabilityScore, uint256 utilizationRate);
    event OrderBookDepthUpdated(
        address indexed dex,
        uint256 bidDepth,
        uint256 askDepth,
        uint256[] feeTierLiquidity
    );

    // Configurable parameters
    uint256 public minProfitThreshold;  // Minimum profit threshold
    uint256 public maxSlippage;         // Maximum allowed slippage
    uint256 public minLiquidityUSD;     // Minimum pool liquidity in USD
    uint256 public maxPositionSizeETH;  // Maximum position size in ETH
    
    // Supported DEX routers
    mapping(string => DexConfig) public dexConfigs;
    
    // Strategy storage
    mapping(bytes32 => bytes) public strategies;
    
    // Provider performance tracking
    mapping(FlashLoanProvider => uint256) public providerSuccesses;
    mapping(FlashLoanProvider => uint256) public providerFailures;
    mapping(FlashLoanProvider => uint256) public providerTotalGas;

    // Events
    event ProviderStatsUpdated(FlashLoanProvider provider, uint256 reliability, uint256 successRate);
    event ProviderEnabled(FlashLoanProvider provider, bool enabled);
    event ArbitrageExecuted(address[] path, uint256 profit);
    event EmergencyWithdraw(address token, uint256 amount);
    event DexRouterUpdated(string indexed dex, address router);
    event StrategyUpdated(string strategyType, bytes strategyData);
    event ProfitThresholdUpdated(uint256 newThreshold);
    event EmergencyPaused(address indexed by);
    event EmergencyUnpaused(address indexed by);

    // Constants for Base mainnet
    address public constant WETH = 0x4200000000000000000000000000000000000006;
    
    // Storage variables for enhanced security
    SecuritySettings private securitySettings;
    mapping(string => ProtocolHealth) private protocolHealth;
    mapping(address => uint256) private lastInteractionTime;
    
    // Events for monitoring
    event SecuritySettingsUpdated(uint256 maxSlippage, uint256 maxGasPrice);
    event ProtocolHealthUpdated(string indexed protocol, bool isHealthy);
    event FlashLoanExecuted(string indexed protocol, address indexed token, uint256 amount, uint256 fee);
    event EmergencyActionTriggered(string indexed reason);

    // Add protocol-specific configuration
    struct ProtocolConfig {
        uint256 maxLeverage;        // Maximum leverage allowed
        uint256 minCollateral;      // Minimum collateral required
        uint256 optimalUtilization; // Optimal utilization rate
        uint256 maxDrawdown;        // Maximum allowed drawdown
        bool supportsFLashMint;     // Whether protocol supports flash minting
        mapping(address => bool) supportedCollateral; // Supported collateral tokens
    }

    // Add to storage variables
    mapping(string => ProtocolConfig) private protocolConfigs;

    // Add new security-related state variables
    uint256 private constant MAX_DAILY_VOLUME = 1000000 ether;  // 1M ETH
    uint256 private constant MAX_TX_COUNT = 100;                // 100 txs per day
    uint256 private constant MIN_COOLDOWN = 5 minutes;          // 5 min cooldown
    
    // Add security events
    event SecurityLimitReached(string indexed limitType, uint256 currentValue, uint256 maxValue);
    event AddressBlocked(address indexed addr, string reason);
    event EmergencyShutdown(address indexed initiator, string reason);
    event SecuritySettingsUpdated(
        uint256 maxSlippage,
        uint256 maxGasPrice,
        uint256 maxDailyVolume,
        uint256 maxTxCount
    );

    // Add events for health monitoring
    event ProtocolHealthUpdated(
        uint256 indexed timestamp,
        uint256 tvl,
        uint256 dailyVolume,
        uint256 utilizationRate,
        bool isHealthy
    );

    event HealthThresholdsUpdated(
        uint256 minTvlThreshold,
        uint256 maxUtilization,
        uint256 minDailyVolume,
        uint256 healthCheckInterval
    );

    // Constructor remains unchanged
    constructor(address _addressProvider) 
        FlashLoanSimpleReceiverBase(IPoolAddressesProvider(_addressProvider)) 
    {
        _disableInitializers();
    }

    /**
     * @dev Required by the OZ UUPS module
     */
    function _authorizeUpgrade(address newImplementation) internal override onlyOwner {
        // Validate the new implementation
        require(newImplementation != address(0), "Invalid implementation address");
        require(newImplementation != address(this), "Cannot upgrade to self");
        
        // Ensure there's a cooldown period between upgrades
        require(
            block.timestamp >= state.lastExecutionTime + 7 days,
            "Upgrade cooldown not elapsed"
        );
        
        // Update last execution time
        state.lastExecutionTime = uint64(block.timestamp);
        
        // Emit upgrade event
        emit ContractUpgraded(address(this), newImplementation);
    }

    /**
     * @dev Enhanced initialization with comprehensive validation
     */
    function initialize(
        address initialOwner,
        uint256 _minProfitThreshold,
        uint256 _maxSlippage,
        uint256 _minLiquidityUSD,
        uint256 _maxPositionSizeETH
    ) public initializer {
        // Enhanced input validation
        require(initialOwner != address(0), "Invalid owner");
        require(_minProfitThreshold > 0, "Invalid profit threshold");
        require(_maxSlippage > 0 && _maxSlippage <= 5000, "Invalid slippage");  // Max 50%
        require(_minLiquidityUSD > 0, "Invalid liquidity threshold");
        require(_maxPositionSizeETH > 0, "Invalid position size");
        require(_maxPositionSizeETH <= 10000 ether, "Position size too large"); // Safety cap
        
        // Initialize OpenZeppelin contracts
        __Ownable_init();
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();
        
        // Transfer ownership with validation
        _validateAndTransferOwnership(initialOwner);
        
        // Initialize state variables with validation
        _initializeStateVariables(
            _minProfitThreshold,
            _maxSlippage,
            _minLiquidityUSD,
            _maxPositionSizeETH
        );
        
        // Initialize configurations
        _initializeProtocolConfigs();
        _initializeSecuritySettings();
        _initializeDexConfigs();
        _initializeFlashLoanConfigs();
        
        // Emit initialization event
        emit ContractInitialized(
            initialOwner,
            _minProfitThreshold,
            _maxSlippage,
            _minLiquidityUSD,
            _maxPositionSizeETH
        );
    }

    /**
     * @dev Initialize state variables with validation
     */
    function _initializeStateVariables(
        uint256 _minProfitThreshold,
        uint256 _maxSlippage,
        uint256 _minLiquidityUSD,
        uint256 _maxPositionSizeETH
    ) internal {
        // Set state variables
        state.minProfitThreshold = uint128(_minProfitThreshold);
        state.lastExecutionTime = uint64(block.timestamp);
        state.isPaused = false;
        state.emergencyTimeout = 0;
        
        // Set public variables
        minProfitThreshold = _minProfitThreshold;
        maxSlippage = _maxSlippage;
        minLiquidityUSD = _minLiquidityUSD;
        maxPositionSizeETH = _maxPositionSizeETH;
        
        // Emit state initialization event
        emit StateInitialized(
            _minProfitThreshold,
            _maxSlippage,
            _minLiquidityUSD,
            _maxPositionSizeETH
        );
    }

    /**
     * @dev Validate and transfer ownership
     */
    function _validateAndTransferOwnership(address newOwner) internal {
        require(newOwner != address(0), "Invalid owner address");
        require(newOwner != address(this), "Contract cannot own itself");
        
        // Transfer ownership
        _transferOwnership(newOwner);
        
        // Add initial owner to whitelist
        whitelistedCallers[newOwner] = true;
        
        emit OwnershipTransferred(address(0), newOwner);
    }

    // Add new events
    event ContractUpgraded(address indexed oldImplementation, address indexed newImplementation);
    event ContractInitialized(
        address indexed owner,
        uint256 minProfitThreshold,
        uint256 maxSlippage,
        uint256 minLiquidityUSD,
        uint256 maxPositionSizeETH
    );
    event StateInitialized(
        uint256 minProfitThreshold,
        uint256 maxSlippage,
        uint256 minLiquidityUSD,
        uint256 maxPositionSizeETH
    );

    /**
     * @dev Validates the price impact of a swap
     */
    function _validatePriceImpact(
        uint256 amountIn,
        uint256 amountOut,
        uint256 expectedAmountOut
    ) internal pure returns (bool) {
        // Calculate price impact
        uint256 priceImpact = ((expectedAmountOut - amountOut) * 1e18) / expectedAmountOut;
        
        // Ensure price impact is within acceptable range
        return priceImpact <= maxSlippage;
    }

    /**
     * @dev Analyze order book depth for a DEX
     */
    function _analyzeOrderBookDepth(
        string memory dex,
        address tokenA,
        address tokenB
    ) internal returns (bool) {
        DexConfig storage config = dexConfigs[dex];
        OrderBookDepth storage depth = orderBookDepths[dex];
        
        // Skip if recently updated
        if (depth.lastUpdateTime == block.timestamp) {
            return true;
        }
        
        uint256 bidDepth;
        uint256 askDepth;
        
        if (config.dexType == DexType.UniswapV3) {
            // Analyze UniswapV3 depth across fee tiers
            for (uint i = 0; i < config.feeTiers.length; i++) {
                (uint256 bid, uint256 ask) = _getUniV3Depth(tokenA, tokenB, config.feeTiers[i]);
                bidDepth += bid;
                askDepth += ask;
            }
        } else {
            // Analyze UniswapV2-style depth
            (bidDepth, askDepth) = _getUniV2Depth(config.router, tokenA, tokenB);
        }
        
        // Update depth tracking
        depth.bidDepth = bidDepth;
        depth.askDepth = askDepth;
        depth.lastUpdateTime = block.timestamp;
        
        // Update DEX ranking based on depth
        _updateDexRanking(dex, bidDepth, askDepth);
        
        emit OrderBookDepthUpdated(dex, bidDepth, askDepth);
        
        return (bidDepth >= minLiquidityUSD && askDepth >= minLiquidityUSD);
    }

    /**
     * @dev Update DEX ranking based on performance metrics
     */
    function _updateDexRanking(
        string memory dex,
        uint256 bidDepth,
        uint256 askDepth
    ) internal {
        DexConfig storage config = dexConfigs[dex];
        
        // Skip if updated recently (prevent gaming)
        if (block.timestamp - config.lastUpdateTime < 1 hours) {
            return;
        }
        
        // Calculate success rate (0-10000)
        uint256 successRate = config.successfulTrades * 10000 / 
            (config.successfulTrades + config.failedTrades + 1);
            
        // Calculate depth score (0-10000)
        uint256 depthScore = (bidDepth + askDepth) * 10000 / (2 * minLiquidityUSD);
        if (depthScore > 10000) depthScore = 10000;
        
        // Update ranking score (70% success rate, 30% depth)
        config.rankingScore = uint128((successRate * 7 + depthScore * 3) / 10);
        config.depthScore = uint128(depthScore);
        config.lastUpdateTime = uint64(block.timestamp);
        
        emit DexRankingUpdated(dex, config.rankingScore);
    }

    /**
     * @dev Update protocol performance metrics
     */
    function _updateProtocolPerformance(
        string memory protocol,
        bool success,
        uint256 gasUsed,
        uint256 profitRatio
    ) internal {
        ProtocolPerformance storage perf = protocolPerformance[protocol];
        
        // Update execution counts
        perf.totalExecutions++;
        if (success) {
            perf.successfulExecs++;
        }
        
        // Update gas and profit metrics
        perf.avgGasUsed = uint96((uint256(perf.avgGasUsed) * (perf.totalExecutions - 1) + 
            gasUsed) / perf.totalExecutions);
        
        perf.avgProfitRatio = uint96((uint256(perf.avgProfitRatio) * (perf.totalExecutions - 1) + 
            profitRatio) / perf.totalExecutions);
            
        // Update reliability score
        perf.reliabilityScore = uint128(perf.successfulExecs * 10000 / perf.totalExecutions);
        
        emit ProtocolPerformanceUpdated(
            protocol,
            perf.reliabilityScore,
            perf.utilizationRate
        );
    }

    /**
     * @dev Enhanced validation with order book depth analysis
     */
    function _validatePairAndLiquidity(
        string memory dex,
        address tokenA,
        address tokenB,
        uint256 amountIn
    ) internal returns (bool) {
        DexConfig memory config = dexConfigs[dex];
        
        // Basic validation
        require(tokenA != address(0) && tokenB != address(0), "Invalid tokens");
        require(tokenA != tokenB, "Identical tokens");
        
        // Get token decimals
        uint8 decimalsA = IERC20(tokenA).decimals();
        uint8 decimalsB = IERC20(tokenB).decimals();
        
        // Analyze order book depth
        require(_analyzeOrderBookDepth(dex, tokenA, tokenB), "Insufficient depth");
        
        // Get pool liquidity
        uint256 liquidityUSD;
        if (config.dexType == DexType.UniswapV3) {
            for (uint i = 0; i < config.feeTiers.length; i++) {
                liquidityUSD += _getUniV3PoolLiquidity(tokenA, tokenB, config.feeTiers[i]);
            }
        } else {
            liquidityUSD = _getUniV2PoolLiquidity(config.router, tokenA, tokenB);
        }
        
        // Enhanced validation with depth analysis
        OrderBookDepth memory depth = orderBookDepths[dex];
        require(depth.bidDepth >= amountIn, "Insufficient bid depth");
        require(depth.askDepth >= amountIn, "Insufficient ask depth");
        
        // Check if amount is within limits
        uint256 amountInUSD = _getTokenAmountInUSD(tokenA, amountIn);
        require(amountInUSD <= maxPositionSizeETH * 1e18, "Position too large");
        
        return true;
    }

    /**
     * @dev Get UniswapV3 pool liquidity in USD
     */
    function _getUniV3PoolLiquidity(
        address tokenA,
        address tokenB,
        uint24 fee
    ) internal view returns (uint256) {
        // This is a simplified version. In production, you would:
        // 1. Get the pool address from the factory
        // 2. Query pool's liquidity and tick spacing
        // 3. Calculate actual liquidity in USD using oracle prices
        return 0; // Placeholder
    }

    /**
     * @dev Get UniswapV2 pool liquidity in USD
     */
    function _getUniV2PoolLiquidity(
        address router,
        address tokenA,
        address tokenB
    ) internal view returns (uint256) {
        // This is a simplified version. In production, you would:
        // 1. Get the pair address from the factory
        // 2. Query reserves
        // 3. Calculate actual liquidity in USD using oracle prices
        return 0; // Placeholder
    }

    /**
     * @dev Get token amount in USD
     */
    function _getTokenAmountInUSD(
        address token,
        uint256 amount
    ) internal view returns (uint256) {
        // This is a simplified version. In production, you would:
        // 1. Use a price oracle (like Chainlink)
        // 2. Get the current price
        // 3. Calculate USD value
        return 0; // Placeholder
    }

    /**
     * @dev Execute swap based on DEX type
     */
    function _executeSwap(
        string memory dex,
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 amountOutMin,
        bytes memory swapData
    ) internal returns (uint256) {
        DexConfig memory config = dexConfigs[dex];
        require(config.enabled, "DEX not enabled");
        require(config.router != address(0), "Invalid router");

        // Validate pair and liquidity
        require(
            _validatePairAndLiquidity(dex, tokenIn, tokenOut, amountIn),
            "Pair validation failed"
        );

        // Get expected amount out
        uint256 expectedAmountOut = _getExpectedAmountOut(dex, tokenIn, tokenOut, amountIn);
        require(expectedAmountOut >= amountOutMin, "Insufficient output amount");
        require(
            expectedAmountOut <= amountOutMin * 120 / 100,  // Max 20% positive slippage
            "Suspicious output amount"
        );

        // Record pre-swap balance
        uint256 preBalance = IERC20(tokenOut).balanceOf(address(this));

        // Execute swap based on DEX type
        uint256 amountOut;
        if (config.dexType == DexType.Curve) {
            amountOut = _executeCurveSwap(
                config.router,
                tokenIn,
                tokenOut,
                amountIn,
                amountOutMin
            );
        } else if (config.dexType == DexType.Morpho) {
            amountOut = _executeMorphoOperation(
                config.router,
                tokenIn,
                amountIn,
                swapData
            );
        } else {
            // Execute existing DEX swaps
            // ... existing swap code ...
        }

        // Verify actual output amount
        uint256 postBalance = IERC20(tokenOut).balanceOf(address(this));
        uint256 actualOutput = postBalance - preBalance;
        
        require(actualOutput >= amountOutMin, "Insufficient actual output");
        require(
            _validatePriceImpact(amountIn, actualOutput, expectedAmountOut),
            "Price impact too high"
        );
        
        // Clear approvals
        IERC20(tokenIn).safeApprove(config.router, 0);
        
        return actualOutput;
    }

    /**
     * @dev Get expected amount out for a swap
     */
    function _getExpectedAmountOut(
        string memory dex,
        address tokenIn,
        address tokenOut,
        uint256 amountIn
    ) internal view returns (uint256) {
        DexConfig memory config = dexConfigs[dex];
        
        if (config.dexType == DexType.UniswapV3) {
            // Use UniswapV3 quoter
            IQuoter quoter = IQuoter(0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6);
            bytes memory path = abi.encodePacked(
                tokenIn,
                uint24(3000), // Use default fee tier
                tokenOut
            );
            return quoter.quoteExactInput(path, amountIn);
        } else {
            // Use UniswapV2 getAmountsOut
            address[] memory path = new address[](2);
            path[0] = tokenIn;
            path[1] = tokenOut;
            
            uint256[] memory amounts = IUniswapV2Router(config.router).getAmountsOut(amountIn, path);
            return amounts[1];
        }
    }

    /**
     * @dev This function is called after your contract has received the flash loaned amount
     */
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external override nonReentrant returns (bool) {
        require(msg.sender == address(POOL), "Caller must be pool");
        require(initiator == address(this), "Initiator must be this contract");
        
        // Decode parameters
        (
            address[] memory path,
            uint256[] memory amounts,
            bytes[] memory swapData
        ) = abi.decode(params, (address[], uint256[], bytes[]));
        
        // Validate parameters
        require(path.length >= 2, "Invalid path length");
        require(amounts.length == path.length - 1, "Invalid amounts length");
        require(swapData.length == amounts.length, "Invalid swap data length");
        
        // Execute swaps with slippage protection
        uint256 initialBalance = IERC20(asset).balanceOf(address(this));
        
        for (uint256 i = 0; i < amounts.length; i++) {
            address tokenIn = path[i];
            address tokenOut = path[i + 1];
            
            // Approve spending if needed
            if (IERC20(tokenIn).allowance(address(this), path[i]) < amounts[i]) {
                IERC20(tokenIn).safeApprove(path[i], 0);
                IERC20(tokenIn).safeApprove(path[i], amounts[i]);
            }
            
            // Execute swap with safety checks
            (bool success, ) = path[i].call(swapData[i]);
            require(success, "Swap failed");
            
            // Verify received amount
            uint256 received = IERC20(tokenOut).balanceOf(address(this));
            require(received >= amounts[i], "Insufficient output amount");
        }
        
        // Verify profit
        uint256 finalBalance = IERC20(asset).balanceOf(address(this));
        require(
            finalBalance >= initialBalance + premium + state.minProfitThreshold,
            "Insufficient profit"
        );
        
        // Approve repayment
        uint256 amountToRepay = amount + premium;
        IERC20(asset).safeApprove(address(POOL), amountToRepay);
        
        return true;
    }

    /**
     * @dev Initiates a flash loan arbitrage
     */
    function executeArbitrage(
        address asset,
        uint256 amount,
        string[] calldata dexes,
        address[] calldata path,
        uint256[] calldata amountsOutMin,
        bytes[] calldata swapData
    ) external onlyOwner nonReentrant whenNotPaused {
        require(path.length > 1, "Invalid path length");
        require(path[0] == path[path.length - 1], "Invalid arbitrage path");
        require(dexes.length == path.length - 1, "Invalid dexes length");
        require(amountsOutMin.length == path.length - 1, "Invalid amountsOutMin length");
        require(swapData.length == path.length - 1, "Invalid swapData length");
        
        bytes memory params = abi.encode(dexes, path, amountsOutMin, swapData);
        POOL.flashLoanSimple(address(this), asset, amount, params, 0);
        
        emit FlashLoanExecuted(FlashLoanProvider.AAVE, asset, amount, 0);
    }

    /**
     * @dev Updates a DEX configuration
     */
    function updateDexConfig(
        string calldata dex,
        address router,
        DexType dexType,
        uint24[] calldata feeTiers,
        bool enabled,
        bool isStablePair
    ) external onlyOwner {
        require(router != address(0), "Invalid router address");
        dexConfigs[dex] = DexConfig({
            router: router,
            dexType: dexType,
            feeTiers: feeTiers,
            enabled: enabled,
            isStablePair: isStablePair,
            rankingScore: 0,
            depthScore: 0,
            lastUpdateTime: 0,
            successfulTrades: 0,
            failedTrades: 0
        });
        emit DexRouterUpdated(dex, router);
    }

    /**
     * @dev Updates the minimum profit threshold
     */
    function updateProfitThreshold(uint256 _minProfitThreshold) external onlyOwner {
        state.minProfitThreshold = uint128(_minProfitThreshold);
        emit ProfitThresholdUpdated(_minProfitThreshold);
    }

    /**
     * @dev Stores a new arbitrage strategy
     */
    function setStrategy(
        string calldata strategyType,
        bytes calldata strategyData
    ) external onlyOwner {
        bytes32 strategyHash = keccak256(abi.encodePacked(strategyType));
        strategies[strategyHash] = strategyData;
        emit StrategyUpdated(strategyType, strategyData);
    }

    /**
     * @dev Emergency withdrawal of stuck funds
     */
    function emergencyWithdraw(address token) external onlyOwner {
        uint256 balance = IERC20(token).balanceOf(address(this));
        require(balance > 0, "No balance to withdraw");
        
        IERC20(token).safeTransfer(owner(), balance);
        emit EmergencyWithdraw(token, balance);
    }

    /**
     * @dev Emergency pause function
     */
    function emergencyPause() external onlyOwner {
        state.isPaused = true;
        state.emergencyTimeout = uint64(block.timestamp + GRACE_PERIOD);
        emit EmergencyPaused(msg.sender);
    }

    /**
     * @dev Emergency unpause function
     */
    function emergencyUnpause() external onlyOwner {
        require(
            block.timestamp >= state.emergencyTimeout,
            "Emergency timeout not elapsed"
        );
        state.isPaused = false;
        emit EmergencyUnpaused(msg.sender);
    }

    /**
     * @dev Execute flash loan with optimal provider selection
     */
    function executeOptimalFlashLoan(
        address asset,
        uint256 amount,
        bytes calldata params
    ) external onlyOwner nonReentrant whenNotPaused {
        // Get optimal provider
        string memory provider = _getOptimalFlashLoanProvider(asset, amount);
        require(bytes(provider).length > 0, "No suitable provider found");
        
        // Validate security conditions
        require(_validateSecurityConditions(asset, amount, msg.sender), "Security check failed");
        
        // Execute flash loan with selected provider
        if (keccak256(bytes(provider)) == keccak256(bytes("AAVE"))) {
            _executeAaveFlashLoan(asset, amount, params);
        } else if (keccak256(bytes(provider)) == keccak256(bytes("BALANCER"))) {
            _executeBalancerFlashLoan(asset, amount, params);
        } else if (keccak256(bytes(provider)) == keccak256(bytes("RADIANT"))) {
            _executeRadiantFlashLoan(asset, amount, params);
        }
        
        // Update protocol metrics
        _updateProtocolMetrics(provider, asset, amount);
    }

    /**
     * @dev Get optimal flash loan provider based on current conditions
     */
    function _getOptimalFlashLoanProvider(
        address asset,
        uint256 amount
    ) internal view returns (string memory) {
        string[3] memory providers = ["AAVE", "BALANCER", "RADIANT"];
        uint256 bestScore = 0;
        string memory bestProvider = "";
        
        for (uint i = 0; i < providers.length; i++) {
            string memory provider = providers[i];
            FlashLoanConfig memory config = flashLoanConfigs[FlashLoanProvider(i)];
            ProtocolHealth memory health = protocolHealth[provider];
            
            // Skip if provider is not viable
            if (!config.enabled ||
                !health.isHealthy ||
                amount < config.minAmount ||
                amount > config.maxAmount ||
                block.timestamp - config.lastLoanTimestamp < config.cooldownPeriod) {
                continue;
            }
            
            // Calculate provider score
            uint256 score = _calculateProviderScore(
                config,
                health,
                amount
            );
            
            // Update best provider if score is higher
            if (score > bestScore) {
                bestScore = score;
                bestProvider = provider;
            }
        }
        
        return bestProvider;
    }

    /**
     * @dev Calculate provider score based on various metrics
     */
    function _calculateProviderScore(
        FlashLoanConfig memory config,
        ProtocolHealth memory health,
        uint256 amount
    ) internal view returns (uint256) {
        // Base score from provider stats
        uint256 score = uint256(config.stats.reliability) * uint256(config.stats.successRate);
        
        // Adjust for utilization
        if (health.utilizationRate > 8000) {  // 80%
            score = score * (10000 - health.utilizationRate) / 2000;  // Reduce score when highly utilized
        }
        
        // Adjust for gas efficiency
        score = score * (1000000 / (config.stats.avgGasUsed + 1));
        
        // Adjust for amount efficiency
        uint256 amountScore = amount * 10000 / config.maxAmount;
        if (amountScore > 8000) {  // Penalize if using >80% of max amount
            score = score * (10000 - amountScore) / 2000;
        }
        
        return score;
    }

    /**
     * @dev Validate security conditions before execution
     */
    function _validateSecurityConditions(
        address asset,
        uint256 amount,
        address sender
    ) internal returns (bool) {
        // Check gas price
        require(tx.gasprice <= securitySettings.maxGasPrice, "Gas price too high");
        
        // Check token is trusted
        require(_isTokenTrusted(asset), "Untrusted token");
        
        // Check position size
        require(amount <= securitySettings.maxPositionSize, "Position too large");
        
        // Check for blocked addresses
        require(!securitySettings.blockedAddresses[sender], "Address blocked");
        
        // Check daily volume limits
        uint256 newDailyVolume = securitySettings.dailyVolumes[sender] + amount;
        require(newDailyVolume <= securitySettings.maxDailyVolume, "Daily volume exceeded");
        
        // Check transaction count
        uint256 currentDay = block.timestamp / 1 days;
        if (currentDay > securitySettings.lastTxTime[sender] / 1 days) {
            securitySettings.txCounts[sender] = 0;
            securitySettings.dailyVolumes[sender] = 0;
        }
        require(securitySettings.txCounts[sender] < securitySettings.maxTxCount, "Max daily tx count exceeded");
        
        // Check cooldown period
        require(
            block.timestamp >= securitySettings.lastTxTime[sender] + securitySettings.cooldownPeriod,
            "Cooldown period not elapsed"
        );
        
        // Update tracking
        securitySettings.dailyVolumes[sender] += amount;
        securitySettings.txCounts[sender]++;
        securitySettings.lastTxTime[sender] = block.timestamp;
        
        return true;
    }

    /**
     * @dev Check if token is trusted
     */
    function _isTokenTrusted(address token) internal view returns (bool) {
        return securitySettings.trustedTokens[token];
    }

    /**
     * @dev Enhanced security settings initialization
     */
    function _initializeSecuritySettings() internal {
        securitySettings.maxSlippagePercent = 100;      // 1% max slippage
        securitySettings.maxGasPrice = 500 gwei;        // 500 gwei max gas price
        securitySettings.minProfitRatio = 110;          // 10% min profit ratio
        securitySettings.maxPositionSize = 1000 ether;  // 1000 ETH max position
        securitySettings.emergencyTimelock = 24 hours;  // 24 hour timelock
        securitySettings.maxDailyVolume = MAX_DAILY_VOLUME;
        securitySettings.maxTxCount = MAX_TX_COUNT;
        securitySettings.cooldownPeriod = MIN_COOLDOWN;
        
        // Initialize trusted tokens
        _initializeTrustedTokens();
        
        emit SecuritySettingsUpdated(
            securitySettings.maxSlippagePercent,
            securitySettings.maxGasPrice,
            securitySettings.maxDailyVolume,
            securitySettings.maxTxCount
        );
    }

    /**
     * @dev Initialize trusted tokens
     */
    function _initializeTrustedTokens() internal {
        // Add base tokens
        securitySettings.trustedTokens[WETH] = true;
        securitySettings.trustedTokens[0x4200000000000000000000000000000000000006] = true; // USDC
        securitySettings.trustedTokens[0x4200000000000000000000000000000000000007] = true; // DAI
        // Add other trusted tokens as needed
    }

    /**
     * @dev Emergency shutdown with timelock
     */
    function emergencyShutdown(string memory reason) external onlyOwner {
        require(
            block.timestamp >= state.emergencyTimeout,
            "Emergency timelock not elapsed"
        );
        
        // Pause all operations
        state.isPaused = true;
        
        // Set emergency timeout
        state.emergencyTimeout = uint64(block.timestamp + securitySettings.emergencyTimelock);
        
        // Emit event
        emit EmergencyShutdown(msg.sender, reason);
    }

    /**
     * @dev Block malicious address
     */
    function blockAddress(address addr, string memory reason) external onlyOwner {
        require(addr != address(0), "Invalid address");
        require(addr != owner(), "Cannot block owner");
        
        securitySettings.blockedAddresses[addr] = true;
        emit AddressBlocked(addr, reason);
    }

    /**
     * @dev Update security settings with validation
     */
    function updateSecuritySettings(
        uint256 _maxSlippage,
        uint256 _maxGasPrice,
        uint256 _maxDailyVolume,
        uint256 _maxTxCount,
        uint256 _cooldownPeriod
    ) external onlyOwner {
        require(_maxSlippage <= 1000, "Slippage too high"); // Max 10%
        require(_maxGasPrice <= 1000 gwei, "Gas price too high");
        require(_maxDailyVolume <= MAX_DAILY_VOLUME * 2, "Volume too high");
        require(_maxTxCount <= MAX_TX_COUNT * 2, "Tx count too high");
        require(_cooldownPeriod >= MIN_COOLDOWN, "Cooldown too short");
        
        securitySettings.maxSlippagePercent = _maxSlippage;
        securitySettings.maxGasPrice = _maxGasPrice;
        securitySettings.maxDailyVolume = _maxDailyVolume;
        securitySettings.maxTxCount = _maxTxCount;
        securitySettings.cooldownPeriod = _cooldownPeriod;
        
        emit SecuritySettingsUpdated(
            _maxSlippage,
            _maxGasPrice,
            _maxDailyVolume,
            _maxTxCount
        );
    }

    /**
     * @dev Update protocol metrics after execution
     */
    function _updateProtocolMetrics(
        string memory protocol,
        address asset,
        uint256 amount
    ) internal {
        ProtocolHealth storage health = protocolHealth[protocol];
        
        // Update volume
        health.dailyVolume += amount;
        
        // Update TVL if needed
        if (block.timestamp - health.lastUpdateTime >= 1 hours) {
            health.totalValueLocked = _getProtocolTVL(protocol);
            health.lastUpdateTime = block.timestamp;
        }
        
        // Update utilization rate
        health.utilizationRate = _calculateUtilizationRate(protocol);
        
        // Update health status
        health.isHealthy = _checkProtocolHealth(protocol);
        
        emit ProtocolHealthUpdated(protocol, health.isHealthy);
    }

    /**
     * @dev Execute flash loan with optimal provider
     */
    function executeFlashLoan(
        address token,
        uint256 amount,
        bytes calldata params
    ) external nonReentrant whenNotPaused {
        require(amount > 0, "Amount must be greater than 0");
        
        // Get optimal provider based on amount, gas price, and historical performance
        FlashLoanProvider provider = _getOptimalProvider(token, amount);
        
        // Execute flash loan based on provider
        if (provider == FlashLoanProvider.AAVE) {
            _executeAaveFlashLoan(token, amount, params);
        } else if (provider == FlashLoanProvider.BALANCER) {
            _executeBalancerFlashLoan(token, amount, params);
        } else if (provider == FlashLoanProvider.RADIANT) {
            _executeRadiantFlashLoan(token, amount, params);
        }
    }

    /**
     * @dev Execute Balancer flash loan
     */
    function _executeBalancerFlashLoan(
        address token,
        uint256 amount,
        bytes calldata params
    ) internal {
        IERC20[] memory tokens = new IERC20[](1);
        tokens[0] = IERC20(token);
        
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = amount;
        
        IVault(flashLoanConfigs[FlashLoanProvider.BALANCER].provider)
            .flashLoan(address(this), tokens, amounts, params);
    }

    /**
     * @dev Execute Radiant flash loan
     */
    function _executeRadiantFlashLoan(
        address token,
        uint256 amount,
        bytes calldata params
    ) internal {
        address[] memory assets = new address[](1);
        assets[0] = token;
        
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = amount;
        
        uint256[] memory modes = new uint256[](1);
        modes[0] = 0; // no debt
        
        IRadiantFlashLoan(flashLoanConfigs[FlashLoanProvider.RADIANT].provider)
            .flashLoan(
                address(this),
                assets,
                amounts,
                modes,
                address(this),
                params,
                0
            );
    }

    /**
     * @dev Balancer flash loan callback
     */
    function receiveFlashLoan(
        IERC20[] memory tokens,
        uint256[] memory amounts,
        uint256[] memory feeAmounts,
        bytes memory userData
    ) external override {
        require(
            msg.sender == flashLoanConfigs[FlashLoanProvider.BALANCER].provider,
            "Only Balancer Vault"
        );
        
        // Execute arbitrage strategy
        _executeStrategy(address(tokens[0]), amounts[0], userData);
        
        // Repay flash loan
        uint256 repayAmount = amounts[0] + feeAmounts[0];
        tokens[0].transfer(msg.sender, repayAmount);
        
        // Update provider stats
        _updateProviderStats(FlashLoanProvider.BALANCER, true, gasleft());
        
        emit FlashLoanExecuted(
            FlashLoanProvider.BALANCER,
            address(tokens[0]),
            amounts[0],
            feeAmounts[0]
        );
    }

    /**
     * @dev Get optimal flash loan provider based on ML recommendations
     */
    function _getOptimalProvider(
        address token,
        uint256 amount
    ) internal view returns (FlashLoanProvider) {
        // Get current gas price
        uint256 gasPrice = tx.gasprice;
        
        // Calculate costs and scores for each provider
        uint256 bestScore = 0;
        FlashLoanProvider bestProvider = FlashLoanProvider.AAVE;
        
        for (uint i = 0; i < 3; i++) {
            FlashLoanProvider provider = FlashLoanProvider(i);
            FlashLoanConfig memory config = flashLoanConfigs[provider];
            
            if (!config.stats.enabled || amount < config.minAmount) continue;
            
            // Calculate provider score based on multiple factors
            uint256 gasCost = config.stats.avgGasUsed * gasPrice;
            uint256 flashLoanFee = (amount * config.fee) / 10000;
            
            // Score calculation formula:
            // (reliability * 40 + successRate * 30 + (1000000/gasCost) * 15 + (1000000/flashLoanFee) * 15) / 100
            uint256 score = (
                (config.stats.reliability * 40) +
                (config.stats.successRate * 30) +
                ((1000000 / gasCost) * 15) +
                ((1000000 / flashLoanFee) * 15)
            ) / 100;
            
            if (score > bestScore) {
                bestScore = score;
                bestProvider = provider;
            }
        }
        
        return bestProvider;
    }

    /**
     * @dev Update provider statistics
     */
    function _updateProviderStats(
        FlashLoanProvider provider,
        bool success,
        uint256 gasUsed
    ) internal {
        FlashLoanConfig storage config = flashLoanConfigs[provider];
        
        if (success) {
            providerSuccesses[provider]++;
        } else {
            providerFailures[provider]++;
        }
        
        config.stats.executionCount++;
        config.stats.successRate = (providerSuccesses[provider] * 100) / config.stats.executionCount;
        
        providerTotalGas[provider] += gasUsed;
        config.stats.avgGasUsed = providerTotalGas[provider] / config.stats.executionCount;
        
        emit ProviderStatsUpdated(provider, config.stats.reliability, config.stats.successRate);
    }

    /**
     * @dev Receive function to accept ETH
     */
    receive() external payable {}

    // Add interface for generic router calls
    interface IGenericRouter {
        function swap(bytes memory data) external returns (uint256);
    }

    /**
     * @dev Initialize protocol configurations
     */
    function _initializeProtocolConfigs() internal {
        // Aave configuration
        protocolConfigs["aave"].maxLeverage = 5;
        protocolConfigs["aave"].minCollateral = 0.1 ether;
        protocolConfigs["aave"].optimalUtilization = 8000; // 80%
        protocolConfigs["aave"].maxDrawdown = 2000; // 20%
        protocolConfigs["aave"].supportsFLashMint = true;
        
        // Balancer configuration
        protocolConfigs["balancer"].maxLeverage = 3;
        protocolConfigs["balancer"].minCollateral = 0.05 ether;
        protocolConfigs["balancer"].optimalUtilization = 7000; // 70%
        protocolConfigs["balancer"].maxDrawdown = 1500; // 15%
        protocolConfigs["balancer"].supportsFLashMint = true;
        
        // Radiant configuration
        protocolConfigs["radiant"].maxLeverage = 4;
        protocolConfigs["radiant"].minCollateral = 0.1 ether;
        protocolConfigs["radiant"].optimalUtilization = 7500; // 75%
        protocolConfigs["radiant"].maxDrawdown = 1800; // 18%
        protocolConfigs["radiant"].supportsFLashMint = true;
    }

    /**
     * @dev Enhanced protocol health check
     */
    function _checkProtocolHealth(string memory protocol) internal view returns (bool) {
        ProtocolHealth storage health = protocolHealth[protocol];
        ProtocolConfig storage config = protocolConfigs[protocol];
        
        // Check TVL
        if (health.totalValueLocked < config.minCollateral * 1000) {
            return false;
        }
        
        // Check utilization
        if (health.utilizationRate > config.optimalUtilization + 1500) { // +15% buffer
            return false;
        }
        
        // Check volume
        if (health.dailyVolume < health.totalValueLocked / 100) { // Min 1% daily volume
            return false;
        }
        
        // Check last update time
        if (block.timestamp - health.lastUpdateTime > 1 hours) {
            return false;
        }
        
        return true;
    }

    /**
     * @dev Calculate actual utilization rate
     */
    function _calculateUtilizationRate(string memory protocol) internal view returns (uint256) {
        ProtocolHealth storage health = protocolHealth[protocol];
        
        if (health.totalValueLocked == 0) {
            return 0;
        }
        
        // Calculate based on borrowed amount vs total liquidity
        uint256 borrowedAmount = _getProtocolBorrowedAmount(protocol);
        return (borrowedAmount * 10000) / health.totalValueLocked;
    }

    /**
     * @dev Get protocol borrowed amount
     */
    function _getProtocolBorrowedAmount(string memory protocol) internal view returns (uint256) {
        if (keccak256(bytes(protocol)) == keccak256(bytes("aave"))) {
            return ILendingPool(flashLoanConfigs[FlashLoanProvider.AAVE].provider).totalDebt();
        } else if (keccak256(bytes(protocol)) == keccak256(bytes("balancer"))) {
            return IBalancerVault(flashLoanConfigs[FlashLoanProvider.BALANCER].provider).getPoolTokenInfo();
        } else if (keccak256(bytes(protocol)) == keccak256(bytes("radiant"))) {
            return IRadiantPool(flashLoanConfigs[FlashLoanProvider.RADIANT].provider).totalDebt();
        }
        return 0;
    }

    /**
     * @dev Get protocol TVL with enhanced accuracy
     */
    function _getProtocolTVL(string memory protocol) internal view returns (uint256) {
        address[] memory supportedTokens = _getSupportedTokens(protocol);
        uint256 totalTVL;
        
        for (uint i = 0; i < supportedTokens.length; i++) {
            address token = supportedTokens[i];
            uint256 tokenTVL = _getTokenTVL(protocol, token);
            uint256 tokenPrice = _getTokenPrice(token);
            totalTVL += (tokenTVL * tokenPrice) / 1e18;
        }
        
        return totalTVL;
    }

    /**
     * @dev Get UniswapV3 pool depth
     */
    function _getUniV3Depth(
        address tokenA,
        address tokenB,
        uint24 fee
    ) internal view returns (uint256 bidDepth, uint256 askDepth) {
        address pool = IUniswapV3Factory(UNIV3_FACTORY).getPool(tokenA, tokenB, fee);
        if (pool == address(0)) return (0, 0);
        
        // Get pool slot0 data
        (uint160 sqrtPriceX96, int24 tick, , , , , ) = IUniswapV3Pool(pool).slot0();
        
        // Get liquidity and ticks
        uint128 liquidity = IUniswapV3Pool(pool).liquidity();
        int24 tickSpacing = IUniswapV3Pool(pool).tickSpacing();
        
        // Calculate depths around current tick
        (bidDepth, askDepth) = _calculateUniV3Depths(
            sqrtPriceX96,
            tick,
            tickSpacing,
            liquidity,
            tokenA,
            tokenB
        );
    }
    
    /**
     * @dev Calculate UniswapV3 depths
     */
    function _calculateUniV3Depths(
        uint160 sqrtPriceX96,
        int24 currentTick,
        int24 tickSpacing,
        uint128 liquidity,
        address tokenA,
        address tokenB
    ) internal view returns (uint256 bidDepth, uint256 askDepth) {
        // Get decimals
        uint8 decimalsA = IERC20(tokenA).decimals();
        uint8 decimalsB = IERC20(tokenB).decimals();
        
        // Calculate price and amounts
        uint256 price = FullMath.mulDiv(
            uint256(sqrtPriceX96) * uint256(sqrtPriceX96),
            10 ** decimalsA,
            2 ** 192
        );
        
        // Calculate bid depth (amount of tokenA that can be sold)
        bidDepth = FullMath.mulDiv(
            uint256(liquidity),
            price,
            10 ** decimalsB
        );
        
        // Calculate ask depth (amount of tokenB that can be bought)
        askDepth = FullMath.mulDiv(
            uint256(liquidity),
            10 ** decimalsA,
            price
        );
    }
    
    /**
     * @dev Get UniswapV2 pool depth
     */
    function _getUniV2Depth(
        address router,
        address tokenA,
        address tokenB
    ) internal view returns (uint256 bidDepth, uint256 askDepth) {
        address pair = IUniswapV2Factory(UNIV2_FACTORY).getPair(tokenA, tokenB);
        if (pair == address(0)) return (0, 0);
        
        // Get reserves
        (uint112 reserve0, uint112 reserve1, ) = IUniswapV2Pair(pair).getReserves();
        
        // Determine token order
        (uint112 reserveA, uint112 reserveB) = tokenA < tokenB ? 
            (reserve0, reserve1) : (reserve1, reserve0);
            
        // Convert to common decimals
        uint8 decimalsA = IERC20(tokenA).decimals();
        uint8 decimalsB = IERC20(tokenB).decimals();
        
        bidDepth = uint256(reserveA) * 10 ** decimalsB / (10 ** decimalsA);
        askDepth = uint256(reserveB);
    }
    
    /**
     * @dev Get Maverick pool depth
     */
    function _getMaverickDepth(
        address tokenA,
        address tokenB,
        uint24 fee
    ) internal view returns (uint256 bidDepth, uint256 askDepth) {
        address pool = IMaverickFactory(MAVERICK_FACTORY).getPool(tokenA, tokenB, fee);
        if (pool == address(0)) return (0, 0);
        
        // Get pool state
        (uint256 reserve0, uint256 reserve1) = IMaverickPool(pool).getReserves();
        
        // Determine token order
        (uint256 reserveA, uint256 reserveB) = tokenA < tokenB ? 
            (reserve0, reserve1) : (reserve1, reserve0);
            
        // Convert to common decimals
        uint8 decimalsA = IERC20(tokenA).decimals();
        uint8 decimalsB = IERC20(tokenB).decimals();
        
        bidDepth = reserveA * 10 ** decimalsB / (10 ** decimalsA);
        askDepth = reserveB;
    }

    /**
     * @dev Get aggregated depth across DEXes
     */
    function _getAggregatedDepth(
        address tokenA,
        address tokenB
    ) internal view returns (uint256 totalBidDepth, uint256 totalAskDepth) {
        string[3] memory dexes = ["uniswap", "sushiswap", "baseswap"];
        
        for (uint i = 0; i < dexes.length; i++) {
            DexConfig memory config = dexConfigs[dexes[i]];
            if (!config.enabled) continue;
            
            uint256 bidDepth;
            uint256 askDepth;
            
            if (config.dexType == DexType.UniswapV3) {
                // Check all fee tiers
                for (uint j = 0; j < config.feeTiers.length; j++) {
                    (uint256 bid, uint256 ask) = _getUniV3Depth(
                        tokenA,
                        tokenB,
                        config.feeTiers[j]
                    );
                    bidDepth += bid;
                    askDepth += ask;
                }
            } else if (config.dexType == DexType.UniswapV2) {
                (bidDepth, askDepth) = _getUniV2Depth(
                    config.router,
                    tokenA,
                    tokenB
                );
            } else if (config.dexType == DexType.Maverick) {
                // Check all fee tiers
                for (uint j = 0; j < config.feeTiers.length; j++) {
                    (uint256 bid, uint256 ask) = _getMaverickDepth(
                        tokenA,
                        tokenB,
                        config.feeTiers[j]
                    );
                    bidDepth += bid;
                    askDepth += ask;
                }
            }
            
            totalBidDepth += bidDepth;
            totalAskDepth += askDepth;
        }
    }

    /**
     * @dev Get supported tokens for a protocol
     */
    function _getSupportedTokens(string memory protocol) internal view returns (address[] memory) {
        if (keccak256(bytes(protocol)) == keccak256(bytes("aave"))) {
            return _getAaveTokens();
        } else if (keccak256(bytes(protocol)) == keccak256(bytes("balancer"))) {
            return _getBalancerTokens();
        } else if (keccak256(bytes(protocol)) == keccak256(bytes("radiant"))) {
            return _getRadiantTokens();
        }
        return new address[](0);
    }

    /**
     * @dev Get token TVL in protocol
     */
    function _getTokenTVL(string memory protocol, address token) internal view returns (uint256) {
        if (keccak256(bytes(protocol)) == keccak256(bytes("aave"))) {
            return ILendingPool(flashLoanConfigs[FlashLoanProvider.AAVE].provider).getReserveData(token).totalStableDebt;
        } else if (keccak256(bytes(protocol)) == keccak256(bytes("balancer"))) {
            return IERC20(token).balanceOf(flashLoanConfigs[FlashLoanProvider.BALANCER].provider);
        } else if (keccak256(bytes(protocol)) == keccak256(bytes("radiant"))) {
            return IRadiantPool(flashLoanConfigs[FlashLoanProvider.RADIANT].provider).getReserveData(token).totalStableDebt;
        }
        return 0;
    }

    /**
     * @dev Get token price from oracle
     */
    function _getTokenPrice(address token) internal view returns (uint256) {
        // Use Chainlink price feeds for accurate pricing
        address oracle = _getPriceOracle(token);
        if (oracle != address(0)) {
            return IPriceOracle(oracle).getAssetPrice(token);
        }
        return 0;
    }

    /**
     * @dev Get price oracle for token
     */
    function _getPriceOracle(address token) internal pure returns (address) {
        // Implement oracle address mapping based on token
        // This should be properly initialized with actual oracle addresses
        return address(0);
    }

    /**
     * @dev Get Aave supported tokens
     */
    function _getAaveTokens() internal view returns (address[] memory) {
        // This should be implemented to return actual supported tokens
        address[] memory tokens = new address[](3);
        tokens[0] = WETH;
        tokens[1] = 0x4200000000000000000000000000000000000006; // USDC
        tokens[2] = 0x4200000000000000000000000000000000000007; // DAI
        return tokens;
    }

    /**
     * @dev Get Balancer supported tokens
     */
    function _getBalancerTokens() internal view returns (address[] memory) {
        // This should be implemented to return actual supported tokens
        address[] memory tokens = new address[](3);
        tokens[0] = WETH;
        tokens[1] = 0x4200000000000000000000000000000000000006; // USDC
        tokens[2] = 0x4200000000000000000000000000000000000007; // DAI
        return tokens;
    }

    /**
     * @dev Get Radiant supported tokens
     */
    function _getRadiantTokens() internal view returns (address[] memory) {
        // This should be implemented to return actual supported tokens
        address[] memory tokens = new address[](3);
        tokens[0] = WETH;
        tokens[1] = 0x4200000000000000000000000000000000000006; // USDC
        tokens[2] = 0x4200000000000000000000000000000000000007; // DAI
        return tokens;
    }

    /**
     * @dev Initialize DEX configurations
     */
    function _initializeDexConfigs() internal {
        // Initialize UniswapV3 fee tiers
        uint24[] memory uniV3Fees = new uint24[](4);
        uniV3Fees[0] = 100;    // 0.01%
        uniV3Fees[1] = 500;    // 0.05%
        uniV3Fees[2] = 3000;   // 0.3%
        uniV3Fees[3] = 10000;  // 1%

        // Initialize empty fee tiers for UniswapV2-style DEXes
        uint24[] memory emptyFees = new uint24[](0);

        // Initialize PancakeSwap fee tiers
        uint24[] memory pancakeV3Fees = new uint24[](4);
        pancakeV3Fees[0] = 100;   // 0.01%
        pancakeV3Fees[1] = 500;   // 0.05%
        pancakeV3Fees[2] = 2500;  // 0.25%
        pancakeV3Fees[3] = 10000; // 1%

        // Initialize Maverick fee tiers
        uint24[] memory maverickFees = new uint24[](3);
        maverickFees[0] = 100;   // 0.01%
        maverickFees[1] = 500;   // 0.05%
        maverickFees[2] = 3000;  // 0.3%

        // Configure UniswapV3-style DEXes
        _configureDex("uniswap", 0x2626664c2603336E57B271c5C0b26F421741e481, DexType.UniswapV3, uniV3Fees, false);
        _configureDex("sushiswap", 0x6BDED42c6DA8FD5E8B11852d05692eE20717c7fE, DexType.UniswapV3, uniV3Fees, false);
        _configureDex("pancakeswap", 0x678Aa4bF4E210cf2166753e054d5b7c31cc7fa86, DexType.UniswapV3, pancakeV3Fees, false);
        _configureDex("maverick", 0x32AED3Bce901DA12ca8489788F3A99fCe1056e14, DexType.Maverick, maverickFees, false);

        // Configure UniswapV2-style DEXes
        _configureDex("baseswap", 0x327Df1E6de05895d2ab08513aaDD9313Fe505d86, DexType.UniswapV2, emptyFees, false);
        _configureDex("alienbase", 0x8c1A3cF8f83074169FE5D7aD50B978e1cD6b37c7, DexType.UniswapV2, emptyFees, false);
        _configureDex("swapbased", 0xaaa3b1F1bd7BCc97fD1917c18ADE665C5D31F066, DexType.UniswapV2, emptyFees, false);
        _configureDex("aerodrome", 0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43, DexType.UniswapV2, emptyFees, false);
        _configureDex("synthswap", 0x7C2c7E7bA4Df5101931bF49C5c2Eb4d9c59D9F0b, DexType.UniswapV2, emptyFees, false);
        _configureDex("velocore", 0x85E8262849Cd8479A3Cd0D1DaB5886D51E585E28, DexType.UniswapV2, emptyFees, false);

        // Emit DEX initialization event
        emit DexConfigsInitialized();
    }

    /**
     * @dev Configure individual DEX with validation
     */
    function _configureDex(
        string memory dexName,
        address router,
        DexType dexType,
        uint24[] memory feeTiers,
        bool isStablePair
    ) internal {
        require(router != address(0), "Invalid router address");
        require(bytes(dexName).length > 0, "Invalid DEX name");

        // Validate fee tiers based on DEX type
        if (dexType == DexType.UniswapV2) {
            require(feeTiers.length == 0, "UniswapV2 should not have fee tiers");
        } else {
            require(feeTiers.length > 0, "Fee tiers required for AMM");
        }

        // Create and store DEX configuration
        dexConfigs[dexName] = DexConfig({
            router: router,
            dexType: dexType,
            feeTiers: feeTiers,
            enabled: true,
            isStablePair: isStablePair,
            rankingScore: 0,
            depthScore: 0,
            lastUpdateTime: 0,
            successfulTrades: 0,
            failedTrades: 0
        });

        // Emit DEX configuration event
        emit DexConfigured(
            dexName,
            router,
            uint8(dexType),
            feeTiers,
            isStablePair
        );
    }

    /**
     * @dev Initialize flash loan configurations
     */
    function _initializeFlashLoanConfigs() internal {
        // Configure Aave
        flashLoanConfigs[FlashLoanProvider.AAVE] = FlashLoanConfig({
            provider: 0xA238Dd80C259a72e81d7e4664a9801593F98d1c5,
            router: 0x9F63Db0fD893403b2bf3B0655f2DA5582fB52dA8,
            minAmount: 0.1 ether,
            fee: 9,  // 0.09%
            maxAmount: 10 ether,
            utilizationRate: 0,
            cooldownPeriod: 0,
            lastLoanTimestamp: 0,
            enabled: true,
            stats: ProviderStats({
                reliability: 95,
                successRate: 98,
                avgGasUsed: 250000,
                executionCount: 0,
                enabled: true
            })
        });

        // Configure Balancer
        flashLoanConfigs[FlashLoanProvider.BALANCER] = FlashLoanConfig({
            provider: 0xBA12222222228d8Ba445958a75a0704d566BF2C8,
            router: address(0),  // Balancer doesn't need a router
            minAmount: 0.05 ether,
            fee: 1,  // 0.01%
            maxAmount: 10 ether,
            utilizationRate: 0,
            cooldownPeriod: 0,
            lastLoanTimestamp: 0,
            enabled: true,
            stats: ProviderStats({
                reliability: 90,
                successRate: 95,
                avgGasUsed: 200000,
                executionCount: 0,
                enabled: true
            })
        });

        // Configure Radiant
        flashLoanConfigs[FlashLoanProvider.RADIANT] = FlashLoanConfig({
            provider: 0x2032b9A8e9F7e76768CA9271003d3e43E1616B1F,
            router: 0x5E7aD666D83dF1E7F0F7189E86a802Be61D11877,
            minAmount: 0.1 ether,
            fee: 9,  // 0.09%
            maxAmount: 10 ether,
            utilizationRate: 0,
            cooldownPeriod: 0,
            lastLoanTimestamp: 0,
            enabled: true,
            stats: ProviderStats({
                reliability: 85,
                successRate: 92,
                avgGasUsed: 275000,
                executionCount: 0,
                enabled: true
            })
        });

        // Emit flash loan configuration event
        emit FlashLoanConfigsInitialized();
    }

    // Add new events
    event DexConfigsInitialized();
    event FlashLoanConfigsInitialized();
    event DexConfigured(
        string indexed dexName,
        address router,
        uint8 dexType,
        uint24[] feeTiers,
        bool isStablePair
    );

    // Add Curve and Morpho specific swap functions
    function _executeCurveSwap(
        address pool,
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 amountOutMin
    ) internal returns (uint256) {
        // Approve spending if needed
        if (IERC20(tokenIn).allowance(address(this), pool) < amountIn) {
            IERC20(tokenIn).safeApprove(pool, 0);
            IERC20(tokenIn).safeApprove(pool, amountIn);
        }
        
        // Execute swap
        return ICurveRouter(pool).exchange(
            pool,
            tokenIn,
            tokenOut,
            amountIn,
            amountOutMin,
            address(this)
        );
    }

    function _executeMorphoOperation(
        address market,
        address tokenIn,
        uint256 amountIn,
        bytes memory params
    ) internal returns (uint256) {
        // Approve spending if needed
        if (IERC20(tokenIn).allowance(address(this), market) < amountIn) {
            IERC20(tokenIn).safeApprove(market, 0);
            IERC20(tokenIn).safeApprove(market, amountIn);
        }
        
        // Execute operation based on params
        (bytes32 operationType, uint256 amount, address onBehalfOf) = 
            abi.decode(params, (bytes32, uint256, address));
            
        if (operationType == "SUPPLY") {
            IMorphoRouter(market).supply(tokenIn, amount, onBehalfOf, 0);
            return amount;
        } else if (operationType == "WITHDRAW") {
            return IMorphoRouter(market).withdraw(tokenIn, amount, address(this));
        }
        
        revert("Invalid Morpho operation");
    }

    // Add new functions for token management
    function addTrustedToken(address token) external onlyOwner {
        require(token != address(0), "Invalid token address");
        securitySettings.trustedTokens[token] = true;
        emit TrustedTokenAdded(token);
    }

    function removeTrustedToken(address token) external onlyOwner {
        require(token != address(0), "Invalid token address");
        securitySettings.trustedTokens[token] = false;
        emit TrustedTokenRemoved(token);
    }

    function isTokenTrusted(address token) public view returns (bool) {
        return securitySettings.trustedTokens[token];
    }

    // Add events for token management
    event TrustedTokenAdded(address indexed token);
    event TrustedTokenRemoved(address indexed token);

    // Add function to update health check thresholds
    function updateHealthThresholds(
        uint256 _minTvlThreshold,
        uint256 _maxUtilization,
        uint256 _minDailyVolume,
        uint256 _healthCheckInterval
    ) external onlyOwner {
        require(_maxUtilization <= MAX_BPS, "Invalid utilization rate");
        require(_healthCheckInterval >= 1 hours && _healthCheckInterval <= 24 hours, "Invalid interval");
        
        protocolHealth.minTvlThreshold = _minTvlThreshold;
        protocolHealth.maxUtilization = _maxUtilization;
        protocolHealth.minDailyVolume = _minDailyVolume;
        protocolHealth.healthCheckInterval = _healthCheckInterval;
        
        emit HealthThresholdsUpdated(
            _minTvlThreshold,
            _maxUtilization,
            _minDailyVolume,
            _healthCheckInterval
        );
    }

    // Enhanced protocol health check function
    function checkProtocolHealth() public returns (bool) {
        require(
            block.timestamp >= protocolHealth.lastUpdateTime + protocolHealth.healthCheckInterval,
            "Health check too frequent"
        );

        // Update TVL and other metrics
        uint256 newTvl = calculateTotalValueLocked();
        uint256 newUtilization = calculateUtilizationRate();
        uint256 newDailyVolume = calculateDailyVolume();

        bool isHealthy = 
            newTvl >= protocolHealth.minTvlThreshold &&
            newUtilization <= protocolHealth.maxUtilization &&
            newDailyVolume >= protocolHealth.minDailyVolume;

        // Update state
        protocolHealth.totalValueLocked = newTvl;
        protocolHealth.utilizationRate = newUtilization;
        protocolHealth.dailyVolume = newDailyVolume;
        protocolHealth.lastUpdateTime = block.timestamp;
        protocolHealth.isHealthy = isHealthy;

        emit ProtocolHealthUpdated(
            block.timestamp,
            newTvl,
            newDailyVolume,
            newUtilization,
            isHealthy
        );

        return isHealthy;
    }

    // Helper functions for health check calculations
    function calculateTotalValueLocked() internal view returns (uint256) {
        // Implementation for TVL calculation
        uint256 tvl = 0;
        // Add TVL calculation logic here
        return tvl;
    }

    function calculateUtilizationRate() internal view returns (uint256) {
        // Implementation for utilization rate calculation
        uint256 utilization = 0;
        // Add utilization calculation logic here
        return utilization;
    }

    function calculateDailyVolume() internal view returns (uint256) {
        // Implementation for daily volume calculation
        uint256 volume = 0;
        // Add volume calculation logic here
        return volume;
    }

    // Optimized order book depth analysis
    function analyzeOrderBookDepth(
        address dex,
        address tokenA,
        address tokenB,
        uint24[] calldata feeTiers
    ) external returns (bool) {
        require(dex != address(0), "Invalid DEX address");
        require(tokenA != address(0) && tokenB != address(0), "Invalid token address");
        
        OrderBookDepth storage depth = orderBookDepths[dex];
        require(
            block.timestamp >= depth.lastUpdateTime + 5 minutes,
            "Update too frequent"
        );

        // Reset previous values
        depth.bidDepth = 0;
        depth.askDepth = 0;
        
        uint256[] memory liquidityPerTier = new uint256[](feeTiers.length);
        
        // Analyze each fee tier
        for (uint256 i = 0; i < feeTiers.length;) {
            uint24 feeTier = feeTiers[i];
            
            // Get liquidity for this fee tier
            (uint256 bidLiquidity, uint256 askLiquidity) = _getFeeTierLiquidity(
                dex,
                tokenA,
                tokenB,
                feeTier
            );
            
            // Update depth tracking
            if (bidLiquidity > 0) {
                depth.bidDepth += uint128(bidLiquidity);
            }
            if (askLiquidity > 0) {
                depth.askDepth += uint128(askLiquidity);
            }
            
            // Store fee tier liquidity
            depth.feeTierLiquidity[feeTier] = bidLiquidity + askLiquidity;
            liquidityPerTier[i] = bidLiquidity + askLiquidity;
            
            unchecked { ++i; } // Gas optimization for loop counter
        }
        
        // Update metadata
        depth.lastUpdateTime = uint64(block.timestamp);
        depth.updateCount++;
        
        emit OrderBookDepthUpdated(
            dex,
            depth.bidDepth,
            depth.askDepth,
            liquidityPerTier
        );
        
        // Return true if sufficient liquidity found
        return depth.bidDepth > 0 && depth.askDepth > 0;
    }
    
    // Internal helper to get liquidity for a specific fee tier
    function _getFeeTierLiquidity(
        address dex,
        address tokenA,
        address tokenB,
        uint24 feeTier
    ) internal view returns (uint256 bidLiquidity, uint256 askLiquidity) {
        // Implementation will vary based on DEX type
        // This is a placeholder - actual implementation needed based on DEX interfaces
        return (0, 0);
    }

    // View function to get current depth
    function getOrderBookDepth(
        address dex
    ) external view returns (
        uint256 bidDepth,
        uint256 askDepth,
        uint256 lastUpdateTime,
        uint256 updateCount
    ) {
        OrderBookDepth storage depth = orderBookDepths[dex];
        return (
            depth.bidDepth,
            depth.askDepth,
            depth.lastUpdateTime,
            depth.updateCount
        );
    }
} 