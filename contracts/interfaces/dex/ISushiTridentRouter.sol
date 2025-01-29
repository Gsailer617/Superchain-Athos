// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

interface ISushiTridentRouter {
    struct ExactInputParams {
        address tokenIn;
        address tokenOut;
        uint256 amountIn;
        uint256 amountOutMin;
        address to;
        bytes route;
    }
    
    function exactInput(ExactInputParams calldata params) external returns (uint256 amountOut);
} 