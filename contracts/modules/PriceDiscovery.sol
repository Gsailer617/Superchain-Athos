// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "../interfaces/uniswap/IUniswapV3Combined.sol";
import "../interfaces/dex/IStableSwapRouter.sol";
import "../interfaces/dex/IRouter.sol";
import "../libraries/TokenValidationLib.sol";

contract PriceDiscovery is Ownable {
    using TokenValidationLib for address;

    // Constants for DEX routers
    address public constant BASESWAP_ROUTER = 0x327Df1E6de05895d2ab08513aaDD9313Fe505d86;
    address public constant AERODROME_ROUTER = 0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43;
    address public constant SUSHISWAP_ROUTER = 0x8a742c5590b4164EAc43866C45D85C1cF7Fc8155;
    address public constant PANCAKESWAP_ROUTER = 0x678Aa4bF4E210cf2166753e054d5b7c31cc7fa86;
    address public constant UNISWAP_V3_ROUTER = 0x2626664c2603336E57B271c5C0b26F421741e481;
    
    // Price monitoring
    struct PricePoint {
        uint256 price;
        uint256 timestamp;
    }

    struct LiquiditySnapshot {
        uint256 liquidity;
        uint256 utilization;
        uint256 timestamp;
    }

    // State variables for price monitoring
    mapping(address => mapping(address => PricePoint[])) private _priceHistory;
    mapping(address => mapping(uint256 => LiquiditySnapshot[])) private _liquidityHistory;
    uint256 private constant MAX_PRICE_POINTS = 24; // Keep 24 hours of hourly prices
    
    // Events
    event PriceChecked(
        address indexed token0,
        address indexed token1,
        string dex,
        uint256 price,
        uint256 liquidity,
        uint256 utilization,
        uint256 timestamp
    );

    event LiquidityUpdated(
        address indexed pool,
        uint256 liquidity,
        uint256 utilization,
        uint256 timestamp
    );

    event PriceVolatilityAlert(
        address indexed token0,
        address indexed token1,
        uint256 volatility,
        uint256 timestamp
    );

    struct PoolInfo {
        address pool;
        uint24 fee;
        bool isStable;
        uint256 price;
        uint256 liquidity;
        uint256 utilization;
    }

    // Price discovery functions
    function getPrice(
        address token0,
        address token1,
        address router
    ) public virtual returns (uint256) {
        uint256 price;
        if (router == BASESWAP_ROUTER || router == AERODROME_ROUTER) {
            // For stable swap routers
            address[] memory path = _getPath(token0, token1);
            address[] memory pools = new address[](1);
            pools[0] = IStableSwapRouter(router).poolFor(token0, token1, true);
            if (pools[0] == address(0)) {
                pools[0] = IStableSwapRouter(router).poolFor(token0, token1, false);
            }
            price = IStableSwapRouter(router).getAmountsOut(1e18, path, pools)[1];
        } else if (router == SUSHISWAP_ROUTER || router == PANCAKESWAP_ROUTER) {
            // For standard routers
            address[] memory path = _getPath(token0, token1);
            price = IRouter(router).getAmountsOut(1e18, path)[1];
        } else if (router == UNISWAP_V3_ROUTER) {
            // For Uniswap V3
            price = IUniswapV3Combined(router).quoteExactInputSingle(
                token0,
                token1,
                3000, // Default fee tier
                1e18,
                0
            );
        }
        
        _recordPricePoint(token0, token1, price);
        return price;
    }

    function _recordPricePoint(
        address token0,
        address token1,
        uint256 price
    ) private {
        PricePoint[] storage history = _priceHistory[token0][token1];
        
        // Remove oldest price point if at capacity
        if (history.length >= MAX_PRICE_POINTS) {
            // Shift array left, removing oldest element
            for (uint i = 0; i < history.length - 1; i++) {
                history[i] = history[i + 1];
            }
            history.pop();
        }
        
        // Add new price point
        history.push(PricePoint({
            price: price,
            timestamp: block.timestamp
        }));

        // Calculate and check volatility
        uint256 volatility = calculateVolatility(token0, token1);
        if (volatility > 50) { // 50% threshold
            emit PriceVolatilityAlert(token0, token1, volatility, block.timestamp);
        }
    }

    function calculateVolatility(
        address token0,
        address token1
    ) public view returns (uint256) {
        PricePoint[] storage history = _priceHistory[token0][token1];
        if (history.length < 2) return 0;

        uint256 maxPrice = 0;
        uint256 minPrice = type(uint256).max;
        
        for (uint i = 0; i < history.length; i++) {
            if (history[i].price > maxPrice) maxPrice = history[i].price;
            if (history[i].price < minPrice) minPrice = history[i].price;
        }
        
        if (minPrice == 0) return 100;
        return ((maxPrice - minPrice) * 100) / minPrice;
    }

    function getPoolInfo(
        address token0,
        address token1,
        address router
    ) public virtual returns (PoolInfo memory info) {
        if (router == UNISWAP_V3_ROUTER) {
            info.pool = IUniswapV3Combined(router).getPool(token0, token1, 3000);
            info.fee = 3000;
            info.isStable = false;
        } else if (router == AERODROME_ROUTER) {
            // Try stable pool first
            info.pool = IStableSwapRouter(router).poolFor(token0, token1, true);
            if (info.pool != address(0)) {
                info.isStable = true;
                info.fee = 100; // 0.01% for stable pools
            } else {
                info.pool = IStableSwapRouter(router).poolFor(token0, token1, false);
                info.isStable = false;
                info.fee = 300; // 0.3% for volatile pools
            }
        } else {
            info.pool = IRouter(router).pairFor(token0, token1);
            info.fee = 300; // 0.3% standard fee
            info.isStable = false;
        }

        if (info.pool != address(0)) {
            info.price = getPrice(token0, token1, router);
            (info.liquidity, info.utilization) = getLiquidityAndUtilization(info.pool);
            
            emit PriceChecked(
                token0,
                token1,
                _getDexName(router),
                info.price,
                info.liquidity,
                info.utilization,
                block.timestamp
            );

            // Record liquidity snapshot
            _recordLiquiditySnapshot(info.pool, info.liquidity, info.utilization);
        }
    }

    function getLiquidityAndUtilization(
        address pool
    ) public view returns (uint256 liquidity, uint256 utilization) {
        try IUniswapV3Combined(pool).liquidity() returns (uint256 liq) {
            liquidity = liq;
            // For V3, utilization is based on tick range
            try IUniswapV3Combined(pool).tickSpacing() returns (int24 spacing) {
                utilization = uint256(uint24(spacing)) * 100 / 887272; // Max tick range
            } catch {
                utilization = 50; // Default to 50%
            }
        } catch {
            // For traditional AMMs
            IERC20 lpToken = IERC20(pool);
            liquidity = lpToken.totalSupply();
            
            // Calculate utilization based on reserves ratio
            try IRouter(pool).getReserves() returns (uint112 reserve0, uint112 reserve1) {
                if (reserve0 > 0 && reserve1 > 0) {
                    utilization = (reserve0 * reserve1 * 100) / (reserve0 + reserve1) ** 2;
                } else {
                    utilization = 50; // Default to 50%
                }
            } catch {
                utilization = 50; // Default to 50%
            }
        }
    }

    function _recordLiquiditySnapshot(
        address pool,
        uint256 liquidity,
        uint256 utilization
    ) private {
        uint256 hourKey = block.timestamp / 1 hours;
        LiquiditySnapshot[] storage history = _liquidityHistory[pool][hourKey];
        
        // Keep only recent snapshots
        if (history.length >= MAX_PRICE_POINTS) {
            // Shift array left, removing oldest element
            for (uint i = 0; i < history.length - 1; i++) {
                history[i] = history[i + 1];
            }
            history.pop();
        }
        
        // Add new snapshot
        history.push(LiquiditySnapshot({
            liquidity: liquidity,
            utilization: utilization,
            timestamp: block.timestamp
        }));

        emit LiquidityUpdated(pool, liquidity, utilization, block.timestamp);
    }

    function _getPath(
        address token0,
        address token1
    ) internal pure virtual returns (address[] memory) {
        address[] memory path = new address[](2);
        path[0] = token0;
        path[1] = token1;
        return path;
    }

    function _getDexName(address router) internal pure virtual returns (string memory) {
        if (router == UNISWAP_V3_ROUTER) return "UniswapV3";
        if (router == BASESWAP_ROUTER) return "BaseSwap";
        if (router == AERODROME_ROUTER) return "Aerodrome";
        if (router == SUSHISWAP_ROUTER) return "SushiSwap";
        if (router == PANCAKESWAP_ROUTER) return "PancakeSwap";
        return "Unknown";
    }

    function calculatePriceImpactScore(
        uint256 amount,
        address[] memory path,
        address router
    ) public view virtual returns (uint256) {
        if (path.length < 2) return 0;
        
        address token0 = path[0];
        address token1 = path[1];
        
        // Get pool info
        address pool;
        if (router == UNISWAP_V3_ROUTER) {
            pool = IUniswapV3Combined(router).getPool(token0, token1, 3000);
        } else {
            pool = IRouter(router).pairFor(token0, token1);
        }
        
        if (pool == address(0)) return 100; // Maximum impact if pool doesn't exist
        
        // Get current liquidity
        (uint256 liquidity, uint256 utilization) = getLiquidityAndUtilization(pool);
        if (liquidity == 0) return 100;
        
        // Calculate impact score (0-100)
        uint256 impactScore = (amount * 100) / liquidity;
        
        // Adjust for utilization
        impactScore = (impactScore * (100 + utilization)) / 100;
        
        return impactScore > 100 ? 100 : impactScore;
    }
} 