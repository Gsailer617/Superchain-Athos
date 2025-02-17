// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IFlashSwapCallback {
    /**
     * @notice Called to `msg.sender` after executing a swap via IUniswapV3Pool#swap.
     * @param amount0Delta The amount of token0 that was sent (negative) or must be received (positive) by the pool
     * @param amount1Delta The amount of token1 that was sent (negative) or must be received (positive) by the pool
     * @param data Any data passed through by the caller via the swap call
     */
    function uniswapV3SwapCallback(
        int256 amount0Delta,
        int256 amount1Delta,
        bytes calldata data
    ) external;

    /**
     * @notice Called to `msg.sender` after executing a flash swap via Curve pool
     * @param sender The address initiating the flash swap
     * @param tokensBorrowed Array of token addresses borrowed
     * @param amountsBorrowed Array of token amounts borrowed
     * @param feeAmounts Array of fee amounts to be paid
     * @param data Arbitrary data passed through the flash swap
     */
    function curveFlashSwapCallback(
        address sender,
        address[] calldata tokensBorrowed,
        uint256[] calldata amountsBorrowed,
        uint256[] calldata feeAmounts,
        bytes calldata data
    ) external;
} 