import { BigNumberish } from 'ethers';

export enum DEX {
    BASESWAP,
    AERODROME,
    SUSHISWAP,
    PANCAKESWAP,
    UNISWAP_V3,
    DACKIESWAP
}

export interface Trade {
    timestamp: number;
    amount: BigNumberish;
    price: BigNumberish;
    hash: string;
    blockNumber: number;
}

export interface PoolData {
    reserves: {
        reserve0: BigNumberish;
        reserve1: BigNumberish;
    };
    totalLiquidity: BigNumberish;
    pair: string;
}

export interface MempoolTx {
    hash: string;
    to: string;
    from: string;
    data: string;
    value: BigNumberish;
    gasPrice: BigNumberish;
    maxFeePerGas?: BigNumberish;
    maxPriorityFeePerGas?: BigNumberish;
}

export interface LiquidityAnalysis {
    volumeScore: number;
    depthScore: number;
    volatilityScore: number;
    healthScore: number;
    recentTrades: {
        count: number;
        averageSize: number;
        largestTrade: number;
        timeWeightedVolume: number;
    };
}

export interface ArbitrageOpportunity {
    sourceDex: DEX;
    targetDex: DEX;
    tokenIn: string;
    tokenOut: string;
    amountIn: BigNumberish;
    expectedProfit: BigNumberish;
    confidence: number;
    healthScore: number;
    route: string[];
}

export interface PricePoint {
    timestamp: number;
    price: BigNumberish;
    volume: BigNumberish;
    liquidity: BigNumberish;
}

export interface DEXConfig {
    router: string;
    factory: string;
    initCodeHash: string;
    fees: number[];
} 