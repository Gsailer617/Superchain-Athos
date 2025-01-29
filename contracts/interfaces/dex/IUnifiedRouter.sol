// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

interface IUnifiedRouter {
    // Common functions across all DEXes
    function getAmountsOut(
        uint256 amountIn,
        address[] calldata path
    ) external view returns (uint256[] memory amounts);

    function pairFor(
        address tokenA,
        address tokenB
    ) external view returns (address pair);

    // Stable swap specific functions
    function getStableAmountsOut(
        uint256 amountIn,
        address[] calldata path,
        address[] calldata pools
    ) external view returns (uint256[] memory amounts);

    function poolFor(
        address tokenA,
        address tokenB,
        bool stable
    ) external view returns (address pool);

    function isStable(address pool) external view returns (bool);

    // Swap functions
    function swapExactTokensForTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external returns (uint256[] memory amounts);

    function swapExactTokensForTokensStable(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address[] calldata pools,
        address to,
        uint256 deadline
    ) external returns (uint256[] memory amounts);
} 