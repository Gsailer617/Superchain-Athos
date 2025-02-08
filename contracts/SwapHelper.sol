// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

interface IQuoter {
    function quoteExactInput(
        bytes memory path,
        uint256 amountIn
    ) external returns (uint256 amountOut);
}

library SwapHelper {
    // DEX-specific constants
    uint24 constant FEE_LOW = 100;      // 0.01%
    uint24 constant FEE_MEDIUM = 500;   // 0.05%
    uint24 constant FEE_HIGH = 3000;    // 0.3%
    uint24 constant FEE_VERY_HIGH = 10000; // 1%

    /**
     * @dev Encodes path for UniswapV3-style swaps
     */
    function encodeV3Path(
        address[] memory tokens,
        uint24[] memory fees
    ) internal pure returns (bytes memory) {
        require(tokens.length >= 2, "Invalid path length");
        require(fees.length == tokens.length - 1, "Invalid fees length");

        bytes memory path = new bytes(0);
        for (uint i = 0; i < tokens.length - 1; i++) {
            if (i == tokens.length - 2) {
                path = abi.encodePacked(path, tokens[i], fees[i], tokens[i + 1]);
            } else {
                path = abi.encodePacked(path, tokens[i], fees[i]);
            }
        }
        return path;
    }

    /**
     * @dev Encodes swap data for UniswapV2-style DEXes
     */
    function encodeV2Swap(
        address[] memory path,
        uint256 amountIn,
        uint256 amountOutMin,
        address recipient,
        uint256 deadline
    ) internal pure returns (bytes memory) {
        return abi.encodeWithSelector(
            bytes4(keccak256("swapExactTokensForTokens(uint256,uint256,address[],address,uint256)")),
            amountIn,
            amountOutMin,
            path,
            recipient,
            deadline
        );
    }

    /**
     * @dev Encodes swap data for UniswapV3
     */
    function encodeUniswapV3Swap(
        address[] memory path,
        uint24[] memory fees,
        uint256 amountIn,
        uint256 amountOutMin,
        address recipient,
        uint256 deadline
    ) internal pure returns (bytes memory) {
        bytes memory encodedPath = encodeV3Path(path, fees);
        return abi.encodeWithSelector(
            bytes4(keccak256("exactInput((bytes,address,uint256,uint256,uint256))")),
            encodedPath,
            recipient,
            deadline,
            amountIn,
            amountOutMin
        );
    }

    /**
     * @dev Encodes swap data for PancakeSwap (supports both V2 and V3)
     */
    function encodePancakeSwap(
        address[] memory path,
        uint24[] memory fees,
        uint256 amountIn,
        uint256 amountOutMin,
        address recipient,
        uint256 deadline,
        bool isV3
    ) internal pure returns (bytes memory) {
        if (isV3) {
            return encodeUniswapV3Swap(path, fees, amountIn, amountOutMin, recipient, deadline);
        } else {
            return encodeV2Swap(path, amountIn, amountOutMin, recipient, deadline);
        }
    }

    /**
     * @dev Encodes swap data for Maverick
     */
    function encodeMaverickSwap(
        address[] memory path,
        uint24[] memory fees,
        uint256 amountIn,
        uint256 amountOutMin,
        address recipient,
        uint256 deadline
    ) internal pure returns (bytes memory) {
        bytes memory encodedPath = encodeV3Path(path, fees);
        return abi.encodeWithSelector(
            bytes4(keccak256("exactInput(bytes,address,uint256,uint256,uint256)")),
            encodedPath,
            recipient,
            deadline,
            amountIn,
            amountOutMin
        );
    }

    /**
     * @dev Encodes swap data for SushiSwap
     */
    function encodeSushiSwap(
        address[] memory path,
        uint256 amountIn,
        uint256 amountOutMin,
        address recipient,
        uint256 deadline
    ) internal pure returns (bytes memory) {
        return encodeV2Swap(path, amountIn, amountOutMin, recipient, deadline);
    }

    /**
     * @dev Encodes swap data for BaseSwap
     */
    function encodeBaseSwap(
        address[] memory path,
        uint256 amountIn,
        uint256 amountOutMin,
        address recipient,
        uint256 deadline
    ) internal pure returns (bytes memory) {
        return encodeV2Swap(path, amountIn, amountOutMin, recipient, deadline);
    }

    /**
     * @dev Encodes swap data for AlienBase
     */
    function encodeAlienBaseSwap(
        address[] memory path,
        uint256 amountIn,
        uint256 amountOutMin,
        address recipient,
        uint256 deadline
    ) internal pure returns (bytes memory) {
        return encodeV2Swap(path, amountIn, amountOutMin, recipient, deadline);
    }

    /**
     * @dev Encodes swap data for SwapBased
     */
    function encodeSwapBasedSwap(
        address[] memory path,
        uint256 amountIn,
        uint256 amountOutMin,
        address recipient,
        uint256 deadline
    ) internal pure returns (bytes memory) {
        return encodeV2Swap(path, amountIn, amountOutMin, recipient, deadline);
    }

    /**
     * @dev Encodes swap data for Aerodrome
     */
    function encodeAerodromeSwap(
        address[] memory path,
        uint256 amountIn,
        uint256 amountOutMin,
        address recipient,
        uint256 deadline
    ) internal pure returns (bytes memory) {
        return encodeV2Swap(path, amountIn, amountOutMin, recipient, deadline);
    }

    /**
     * @dev Encodes swap data for SynthSwap
     */
    function encodeSynthSwap(
        address[] memory path,
        uint256 amountIn,
        uint256 amountOutMin,
        address recipient,
        uint256 deadline
    ) internal pure returns (bytes memory) {
        return encodeV2Swap(path, amountIn, amountOutMin, recipient, deadline);
    }

    /**
     * @dev Encodes swap data for HorizonDEX
     */
    function encodeHorizonDEXSwap(
        address[] memory path,
        uint256 amountIn,
        uint256 amountOutMin,
        address recipient,
        uint256 deadline
    ) internal pure returns (bytes memory) {
        return encodeV2Swap(path, amountIn, amountOutMin, recipient, deadline);
    }

    /**
     * @dev Get optimal fee tier based on liquidity and volatility
     */
    function getOptimalFeeTier(
        address,  // tokenA
        address,  // tokenB
        bool isStablePair
    ) internal pure returns (uint24) {
        if (isStablePair) {
            return FEE_LOW;  // 0.01% for stable pairs
        }
        
        // Default to 0.3% for most pairs
        return FEE_HIGH;
    }
} 