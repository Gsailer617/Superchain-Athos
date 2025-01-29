// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

interface IArbitrageTypes {
    enum FlashLoanProvider { MOONWELL, MORPHO, AAVE, BALANCER }
    enum DEX { BASESWAP, AERODROME, SUSHISWAP, PANCAKESWAP, UNISWAP_V3 }
    enum DexRoute {
        NONE,
        UNISWAP_TO_SUSHI,
        UNISWAP_TO_PANCAKE
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

    struct ArbitragePath {
        address[] tokens;
        address[] pools;
        uint256[] amounts;
        DEX[] dexes;
        uint24[] fees;
        bool[] isStablePools;
    }

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
} 