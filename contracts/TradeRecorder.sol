// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "./interfaces/ITradeRecorder.sol";

contract TradeRecorder is ITradeRecorder {
    using SafeERC20 for IERC20;

    // Private storage
    mapping(uint256 => DetailedTradeRecord) private _tradeRecords;
    uint256 private _totalTrades;

    // Public view functions
    function getTrade(uint256 tradeId) external view override returns (DetailedTradeRecord memory) {
        return _tradeRecords[tradeId];
    }

    function getTotalTrades() external view override returns (uint256) {
        return _totalTrades;
    }

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
    ) external override returns (uint256) {
        uint256 tradeId = _totalTrades++;
        DetailedTradeRecord storage record = _tradeRecords[tradeId];
        
        // Record basic trade info
        record.tokenIn = tokenIn;
        record.tokenOut = tokenOut;
        record.amountIn = amountIn;
        record.amountOut = amountOut;
        record.profit = profit;
        record.timestamp = block.timestamp;
        record.sourceDex = sourceDex;
        record.targetDex = targetDex;
        
        // Record transaction info
        record.transactionHash = blockhash(block.number - 1);
        record.blockNumber = block.number;
        record.gasUsed = gasleft();
        record.gasPrice = tx.gasprice;
        record.executor = msg.sender;
        
        // Record metrics
        record.slippage = slippage;
        record.priceImpact = priceImpact;
        record.marketPrice0 = marketPrice0;
        record.marketPrice1 = marketPrice1;
        record.liquidityAtExecution = liquidityAtExecution;

        // Emit events using interface events
        emit ITradeRecorder.DetailedTradeExecuted(
            tradeId,
            tokenIn,
            tokenOut,
            amountIn,
            amountOut,
            profit,
            block.timestamp,
            sourceDex,
            targetDex,
            record.transactionHash,
            record.gasUsed,
            priceImpact
        );

        emit ITradeRecorder.RiskMetricsLogged(
            tradeId,
            slippage,
            priceImpact,
            marketPrice0,
            marketPrice1,
            liquidityAtExecution
        );

        return tradeId;
    }
} 