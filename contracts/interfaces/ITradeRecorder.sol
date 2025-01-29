// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

interface ITradeRecorder {
    struct DetailedTradeRecord {
        address tokenIn;
        address tokenOut;
        uint256 amountIn;
        uint256 amountOut;
        uint256 profit;
        uint256 timestamp;
        string sourceDex;
        string targetDex;
        bytes32 transactionHash;
        uint256 blockNumber;
        uint256 gasUsed;
        uint256 gasPrice;
        address executor;
        uint256 slippage;
        uint256 priceImpact;
        uint256 marketPrice0;
        uint256 marketPrice1;
        uint256 liquidityAtExecution;
    }

    event DetailedTradeExecuted(
        uint256 indexed tradeId,
        address indexed tokenIn,
        address indexed tokenOut,
        uint256 amountIn,
        uint256 amountOut,
        uint256 profit,
        uint256 timestamp,
        string sourceDex,
        string targetDex,
        bytes32 transactionHash,
        uint256 gasUsed,
        uint256 priceImpact
    );

    event RiskMetricsLogged(
        uint256 indexed tradeId,
        uint256 slippage,
        uint256 priceImpact,
        uint256 marketPrice0,
        uint256 marketPrice1,
        uint256 liquidityAtExecution
    );

    function recordTrade(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 amountOut,
        uint256 profit,
        string calldata sourceDex,
        string calldata targetDex,
        uint256 slippage,
        uint256 priceImpact,
        uint256 marketPrice0,
        uint256 marketPrice1,
        uint256 liquidityAtExecution
    ) external returns (uint256);

    function getTrade(uint256 tradeId) external view returns (DetailedTradeRecord memory);
    function getTotalTrades() external view returns (uint256);
} 