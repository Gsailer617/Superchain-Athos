// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "../interfaces/uniswap/IUniswapV3Combined.sol";
import "../interfaces/dex/IStableSwapRouter.sol";
import "../interfaces/dex/IRouter.sol";

library PriceDiscoveryLib {
    // Price discovery functions
    function getPrice(
        address token0,
        address token1,
        address router,
        address quoter,
        uint24 defaultFeeTier
    ) internal returns (uint256) {
        if (router == address(0)) revert("Invalid router");
        
        if (_isUniswapV3(router)) {
            return IUniswapV3Combined(quoter).quoteExactInputSingle(
                token0,
                token1,
                defaultFeeTier,
                1e18, // 1 token0
                0
            );
        } else {
            address[] memory path = _getPath(token0, token1);
            uint256[] memory amounts = IRouter(router).getAmountsOut(1e18, path);
            return amounts[1];
        }
    }

    function getLiquidity(address pool, address router) internal view returns (uint256) {
        if (pool == address(0)) return 0;
        
        // Different implementations for different DEXes
        if (_isUniswapV3(router)) {
            try IUniswapV3Combined(pool).liquidity() returns (uint256 liq) {
                return liq;
            } catch {
                return 0;
            }
        } else {
            return IERC20(pool).totalSupply(); // For traditional AMMs
        }
    }

    function getPoolInfo(
        address token0,
        address token1,
        address router,
        address quoter,
        uint24 defaultFeeTier
    ) internal returns (
        address pool,
        uint24 fee,
        bool isStable,
        uint256 price,
        uint256 liquidity
    ) {
        if (_isUniswapV3(router)) {
            pool = IUniswapV3Combined(router).getPool(token0, token1, defaultFeeTier);
            fee = defaultFeeTier;
            isStable = false;
        } else if (_isAerodrome(router)) {
            // Try stable pool first
            pool = IStableSwapRouter(router).poolFor(token0, token1, true);
            if (pool != address(0)) {
                isStable = true;
                fee = 100; // 0.01% for stable pools
            } else {
                pool = IStableSwapRouter(router).poolFor(token0, token1, false);
                isStable = false;
                fee = 300; // 0.3% for volatile pools
            }
        } else {
            pool = IRouter(router).pairFor(token0, token1);
            fee = 300; // 0.3% standard fee
            isStable = false;
        }

        if (pool != address(0)) {
            price = getPrice(token0, token1, router, quoter, defaultFeeTier);
            liquidity = getLiquidity(pool, router);
        }
    }

    function calculatePriceImpact(
        uint256,  // amount
        address[] memory,  // path
        address  // router
    ) internal pure returns (uint256) {
        // Implementation will be added
        return 0;
    }

    // Internal helper functions
    function _getPath(
        address token0,
        address token1
    ) private pure returns (address[] memory) {
        address[] memory path = new address[](2);
        path[0] = token0;
        path[1] = token1;
        return path;
    }

    function _isUniswapV3(address router) private pure returns (bool) {
        // Add your Uniswap V3 router address check here
        return router == 0x2626664c2603336E57B271c5C0b26F421741e481;
    }

    function _isAerodrome(address router) private pure returns (bool) {
        // Add your Aerodrome router address check here
        return router == 0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43;
    }
} 