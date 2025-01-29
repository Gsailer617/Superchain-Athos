export const QUALITY_THRESHOLDS = {
    MIN_LIQUIDITY_USD: 5000, // Lowered from 10k to 5k USD
    HEALTH_SCORE_THRESHOLD: 0.6, // Lowered from 0.7 to 0.6
    VOLUME: {
        MIN_DAILY_VOLUME: BigInt(3500), // Lowered from 5k to 3.5k
        MIN_TRADES: 8, // Lowered from 10 to 8
        MIN_LIQUIDITY: BigInt(500) // Lowered from 10k to 1k
    },
    PRICE: {
        MAX_SLIPPAGE: 0.05, // Increased from 0.03 to 0.05
        MIN_PRICE_IMPACT: 0.0006, // Lowered from 0.001
        MAX_PRICE_IMPACT: 0.1 // Increased from 0.05 to 0.1
    },
    GAS: {
        MAX_GAS_PRICE: BigInt(100000000000), // Increased from 100 to 300 gwei
        MIN_PRIORITY_FEE: BigInt(100000000), // Lowered from 1 gwei to 0.1 gwei
        MAX_GAS_LIMIT: BigInt(1000000) // Increased from 500k to 1M
    },
    TIME: {
        MAX_EXECUTION_TIME: 30000, // Increased from 30s to 60s
        MAX_PENDING_TIME: 300000, // Increased from 5m to 10m
        MIN_BLOCK_CONFIRMATIONS: 1
    },
    PROFIT: {
        MIN_PROFIT_USD: 2n, // Lowered from 10 to 2 USD
        MIN_ROI: 20, // Lowered from 50 to 20 (0.2%)
        TARGET_ROI: 50 // Lowered from 100 to 50 (0.5%)
    },
    MIN_TRADES_COUNT: 8, // Lowered from 10 to 8
    MIN_VOLUME_SCORE: 0.5, // Lowered from 0.7 to 0.5
    MAX_POOL_SHARE_BPS: 700, // Increased from 500 to 1000 (7%)
    EXTREME_CONDITIONS: {
        MAX_PRICE_CHANGE_PCT: 10,
        MIN_LIQUIDITY_RATIO: 0.6, // Lowered from 0.7 to 0.6
        MAX_MANIPULATION_RISK: 0.3, // Increased from 0.3 to 0.5
        MAX_FLASH_LOAN_RISK: 0.5 // Increased from 0.4 to 0.6
    }
}; 