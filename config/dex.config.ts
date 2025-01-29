import { DEX, DEXConfig } from '../types/dex';

export const DEX_CONFIGS: Record<DEX, DEXConfig> = {
    [DEX.BASESWAP]: {
        router: '0x327Df1E6de05895d2ab08513aaDD9313Fe505d86',
        factory: '0x0Df4b4dE8c4A76F112d4c17f3E8Eb5A5467C99B1',
        initCodeHash: '0x9a100ded5f254205b40702e6992c05dac519e5c4cc8fda0f208269d6b6d62b72',
        fees: [30] // 0.3%
    },
    [DEX.AERODROME]: {
        router: '0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43',
        factory: '0x420DD381b31aEf6683db6B902084cB0FFEe076d7',
        initCodeHash: '0x8edd3d361f32469c7fd4747f6e3d8d914d10c986b0a33c88a7529ee5b7938a2d',
        fees: [2, 5, 30] // 0.02%, 0.05%, 0.3%
    },
    [DEX.SUSHISWAP]: {
        router: '0x8a742C5590B4164EAc43866c45D85c1cF7fc8155',
        factory: '0xc35DADB65012eC5796536bD9864eD8773aBc74C4',
        initCodeHash: '0xe18a34eb0e04b04f7a0ac29a6e80748dca96319b42c54d679cb821dca90c6303',
        fees: [1, 5, 30, 100], // 0.01%, 0.05%, 0.3%, 1%
        features: {
            bentoBox: true,
            concentratedLiquidity: true,
            trident: true
        },
        bentoBox: '0xc381a85ed7C7448Da073b7d6C9d4cBf1Cbf576f0',
        tridentRouter: '0xE52180815c81D7711B83c692C4Fd3417F37cF9b1'
    },
    [DEX.PANCAKESWAP]: {
        router: '0x678Aa4bF4E210cf2166753e054d5b7c31cc7fa86',
        factory: '0x0BFbCF9fa4f9C56B0F40a671Ad40E0805A091865',
        initCodeHash: '0x00fb7f630766e6a796048ea87d01acd3068e8ff67d078148a3fa3f4a84f69bd5',
        fees: [25, 100, 300, 1000], // 0.25%, 1%, 3%, 10%
        features: {
            stableSwap: true,
            concentratedLiquidity: true
        },
        stableSwapRouter: '0x678Aa4bF4E210cf2166753e054d5b7c31cc7fa86',
        stableSwapFactory: '0x0BFbCF9fa4f9C56B0F40a671Ad40E0805A091865'
    },
    [DEX.UNISWAP_V3]: {
        name: 'Uniswap V3',
        router: '0x2626664c2603336E57B271c5C0b26F421741e481',
        factory: '0x33128a8fC17869897dcE68Ed026d694621f6FDfD',
        initCodeHash: '0xe34f199b19b2b4f47f68442619d555527d244f78a3297ea89325f843f87b8b54',
        fees: [100, 500, 3000, 10000],
        quoter: '0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a',
        v3Factory: '0x33128a8fC17869897dcE68Ed026d694621f6FDfD',
        v3Router: '0x2626664c2603336E57B271c5C0b26F421741e481',
        poolDeployer: '0x33128a8fC17869897dcE68Ed026d694621f6FDfD'
    }
};

export const COMMON_TOKENS = {
    WETH: '0x4200000000000000000000000000000000000006',
    USDC: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
    DAI: '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb',
    USDT: '0x4A3A6Dd60A34bB2Aba60D73B4C88315E9CeB6A3D',
    cbETH: '0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22',
    USDbC: '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA'
};

export const QUALITY_THRESHOLDS = {
    MIN_LIQUIDITY_USD: 100000, // Minimum pool liquidity in USD
    MIN_VOLUME_24H: 50000,     // Minimum 24h volume in USD
    MAX_PRICE_IMPACT: 100,     // Maximum price impact in basis points (1%)
    MIN_TRADES_COUNT: 10,      // Minimum number of trades to consider pool active
    GAS_PRICE_THRESHOLD: 100,  // Maximum gas price in gwei
    MIN_PROFIT_USD: 50,        // Minimum profit in USD
    HEALTH_SCORE_THRESHOLD: 70, // Minimum health score (0-100)
    STABLE_SWAP: {
        MIN_LIQUIDITY_USD: 500000,    // Higher liquidity requirement for stable swaps
        MAX_PRICE_DEVIATION: 50,      // 0.5% max price deviation
        MIN_AMPLIFIER: 50,            // Minimum amplification coefficient
        MAX_SLIPPAGE: 50,             // 0.5% max slippage for stable swaps
        VIRTUAL_PRICE_DEVIATION: 10    // 0.1% max virtual price deviation
    }
};

export const STABLE_PAIRS = [
    {
        tokens: [COMMON_TOKENS.USDC, COMMON_TOKENS.USDT, COMMON_TOKENS.DAI],
        amplifier: 100,
        expectedVolatility: 0.01, // 1% max expected volatility
        maxPriceDeviation: 50 // 0.5% max price deviation
    }
]; 