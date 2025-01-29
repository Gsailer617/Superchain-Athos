// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

library MLMonitoringLib {
    struct BotStatus {
        bool isScanning;
        bool isExecutingTrade;
        uint256 lastScanTime;
        uint256 totalScansCompleted;
        uint256 totalOpportunitiesFound;
        uint256 totalTradesExecuted;
        uint256 totalProfitRealized;
        uint256 totalGasUsed;
    }

    struct TradeHistory {
        address tokenIn;
        address tokenOut;
        uint256 amountIn;
        uint256 amountOut;
        uint256 profit;
        uint256 timestamp;
        string sourceDex;
        string targetDex;
    }

    function updateBotStatus(
        BotStatus storage status,
        uint256 profitRealized,
        uint256 gasUsed
    ) internal {
        status.totalTradesExecuted++;
        status.totalProfitRealized += profitRealized;
        status.totalGasUsed += gasUsed;
    }

    function recordNewToken(
        mapping(address => bool) storage discoveredTokens,
        address[] storage allDiscoveredTokens,
        address token
    ) internal returns (bool isNew) {
        if (!discoveredTokens[token]) {
            discoveredTokens[token] = true;
            allDiscoveredTokens.push(token);
            isNew = true;
        }
    }

    function recordNewPair(
        mapping(address => mapping(address => mapping(address => bool))) storage discoveredPairs,
        address dex,
        address token0,
        address token1
    ) internal returns (bool isNew) {
        if (!discoveredPairs[dex][token0][token1]) {
            discoveredPairs[dex][token0][token1] = true;
            discoveredPairs[dex][token1][token0] = true;
            isNew = true;
        }
    }

    function addTradeHistory(
        TradeHistory[] storage history,
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 amountOut,
        uint256 profit,
        string memory sourceDex,
        string memory targetDex
    ) internal {
        history.push(TradeHistory({
            tokenIn: tokenIn,
            tokenOut: tokenOut,
            amountIn: amountIn,
            amountOut: amountOut,
            profit: profit,
            timestamp: block.timestamp,
            sourceDex: sourceDex,
            targetDex: targetDex
        }));
    }

    function calculateHistoricalScore(
        TradeHistory[] storage history,
        string memory dex1,
        string memory dex2
    ) internal view returns (uint256) {
        uint256 totalTrades = 0;
        uint256 profitableTrades = 0;

        for (uint i = 0; i < history.length; i++) {
            TradeHistory storage trade = history[i];
            if (
                keccak256(abi.encodePacked(trade.sourceDex)) == keccak256(abi.encodePacked(dex1)) &&
                keccak256(abi.encodePacked(trade.targetDex)) == keccak256(abi.encodePacked(dex2))
            ) {
                totalTrades++;
                if (trade.profit > 0) {
                    profitableTrades++;
                }
            }
        }

        if (totalTrades == 0) return 80; // Base score if no history
        return (profitableTrades * 100) / totalTrades;
    }
} 