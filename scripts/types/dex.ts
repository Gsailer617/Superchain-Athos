import { BigNumberish } from 'ethers';

export enum DEXType {
    UNISWAP_V2 = 'UNISWAP_V2',
    SUSHISWAP = 'SUSHISWAP',
    AERODROME = 'AERODROME',
    BASESWAP = 'BASESWAP'
}

export interface DexInfo {
    type: DEXType;
    name: string;
    address: `0x${string}`;
    version: string;
    factoryAddress: `0x${string}`;
    routerAddress: `0x${string}`;
    initCodeHash: `0x${string}`;
    supportedTokens: `0x${string}`[];
}

export interface Pool {
    address: `0x${string}`;
    token0: `0x${string}`;
    token1: `0x${string}`;
    reserve0: bigint;
    reserve1: bigint;
    fee: number;
    isStable?: boolean;  // Optional property for Aerodrome pools
}

export enum FlashLoanProvider {
    AAVE = 'AAVE',
    BALANCER = 'BALANCER',
    MOONWELL = 'MOONWELL',  // Base lending protocol
    BASESWAP = 'BASESWAP',   // BaseSwap flash loans
    MORPHO = 'MORPHO'     // Morpho lending protocol
}

export interface FlashLoanParams {
    tokenIn: `0x${string}`;
    amount: bigint;
    fee: number;
    provider: FlashLoanProvider;
    assets: `0x${string}`[];
    amounts: bigint[];
    modes: number[];
    params: `0x${string}`;
    maxFeePerGas: bigint;
    maxPriorityFeePerGas: bigint;
    deadline: bigint;
    maxBorrowRates: bigint[];
    pairs: `0x${string}`[];
    slippage: number;
}

export interface VolumeMetrics {
    volumeScore: number;
    tradeFrequency: number;
    tradeSizeDistribution: {
        small: number;
        medium: number;
        large: number;
    };
    volatility: number;
    trend: 'increasing' | 'decreasing' | 'stable';
    confidence: number;
}

export interface LiquidityAnalysis {
    totalLiquidity: bigint;
    concentrationScore: number;
    healthScore: number;
}

export interface PriceImpactAnalysis {
    expectedImpact: number;
    worstCase: number;
    confidence: number;
    recommendedMaxSize: bigint;
}

export interface ArbitrageOpportunity {
    path: DEXType[];
    tokens: `0x${string}`[];
    expectedProfit: bigint;
    confidence: number;
    amountIn: bigint;
    gasEstimate: bigint;
    route: {
        dex: DEXType;
        path: `0x${string}`[];
    }[];
    amounts: bigint[];
    mevRisk: {
        riskScore: number;
        type: string;
        probability: number;
    };
    optimizedGas: bigint;
    expectedSlippage: number;
    recommendedPath: string[];
}

export interface VolumeData {
    volume1h: number;
    volume24h: number;
    volume7d: number;
    tradeCount1h: number;
    tradeCount24h: number;
    tradeCount7d: number;
    averageTradeSize1h: number;
    averageTradeSize24h: number;
    averageTradeSize7d: number;
    largestTrade1h: number;
    largestTrade24h: number;
    largestTrade7d: number;
    lastUpdate: number;
}

export interface TrendAnalysis {
    timeframe: string;
    direction: 'up' | 'down' | 'sideways';
    magnitude: number;
    confidence: number;
    volatility: number;
    support: number;
    resistance: number;
    breakoutProbability: number;
}

export interface PricePoint {
    timestamp: number;
    price: number;
    volume: number;
    liquidity: number;
}

export interface TransactionWithTo {
    to: string;
    hash: string;
    blockNumber: number;
    blockTimestamp?: number;
    data: string;
    value: bigint;
    gasPrice: bigint;
    gasLimit: bigint;
}

export interface StableSwapPool {
    address: string;
    tokens: string[];
    amplifier: bigint;
    fees: bigint;
    virtualPrice: bigint;
    totalLiquidity: bigint;
    poolBalance: bigint[];
    provider?: string;  // Added to track which protocol provides the pool
}

export interface StableSwapQuote {
    expectedOutput: bigint;
    priceImpact: number;
    fee: bigint;
    route: string[];
    pool: StableSwapPool;
}

export interface PoolStats {
    address: string;
    token0Volume: VolumeData;
    token1Volume: VolumeData;
    metrics: VolumeMetrics;
    lastUpdate: number;
}

export interface AerodromePool extends StableSwapPool {
    isStable: boolean;
    factory: string;
    token0: string;
    token1: string;
    reserve0: bigint;
    reserve1: bigint;
    gaugeAddress?: string;
    bribeAddress?: string;
    feePercent: number;
    stableFee: number;
    volatileFee: number;
}

export interface PoolConfig {
    address: string;
    dex: DEXType;
    isStable?: boolean;
    fee?: number;
    minLiquidity?: bigint;
    maxSlippage?: number;
}

// Add lending protocol specific interfaces
export interface LendingProtocolConfig {
    name: string;
    address: `0x${string}`;
    provider: FlashLoanProvider;
    supportedTokens: `0x${string}`[];
    flashLoanFee: number;
    maxLTV: number;
} 