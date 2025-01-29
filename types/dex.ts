export interface DEXConfig {
    name?: string;
    router: string;
    factory: string;
    initCodeHash: string;
    fees: number[];
    quoter?: string;
    features?: {
        bentoBox?: boolean;
        concentratedLiquidity?: boolean;
        trident?: boolean;
        stableSwap?: boolean;
    };
    stableSwapRouter?: string;
    stableSwapFactory?: string;
    bentoBox?: string;
    tridentRouter?: string;
    v2Router?: string;
    v2Factory?: string;
    v3Factory?: string;
    v3Router?: string;
    positionManager?: string;
    poolDeployer?: string;
}

export enum DEX {
    BASESWAP = 'BASESWAP',
    AERODROME = 'AERODROME',
    SUSHISWAP = 'SUSHISWAP',
    PANCAKESWAP = 'PANCAKESWAP',
    UNISWAP_V3 = 'UNISWAP_V3'
}

export interface SwapSimulation {
    expectedOutput: bigint;
    priceImpact: number;
    fees: bigint;
    minOutput: bigint;
    maxOutput: bigint;
    hasLiquidity: boolean;
    routeType?: 'v2' | 'trident' | 'concentrated';
    bentoBoxOptimized?: boolean;
}

export interface MultiHopPath {
    hops: HopInfo[];
    totalPriceImpact: number;
    expectedProfit: bigint;
    isValid: boolean;
    routeType?: 'v2' | 'trident' | 'concentrated';
    bentoBoxOptimized?: boolean;
}

export interface HopInfo {
    dex: DEX;
    tokenIn: string;
    tokenOut: string;
    amountIn: bigint;
    amountOut: bigint;
    priceImpact: number;
    fee?: number;
    routeType?: 'v2' | 'trident' | 'concentrated';
}

export interface StableSwapPool {
    address: string;
    tokens: string[];
    amplifier: number;
    fees: {
        swapFee: number;
        adminFee: number;
    };
    virtualPrice: bigint;
    totalLiquidity: bigint;
}

export interface StableSwapQuote {
    expectedOutput: bigint;
    priceImpact: number;
    fee: bigint;
    route: string[];
    pool: StableSwapPool;
} 