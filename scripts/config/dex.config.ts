export const QUALITY_THRESHOLDS = {
    MIN_LIQUIDITY_USD: 100000,      // Minimum pool liquidity in USD
    MIN_VOLUME_24H: 50000,          // Minimum 24h volume in USD
    MAX_PRICE_IMPACT: 100,          // Maximum price impact in basis points (1%)
    MIN_TRADES_COUNT: 10,           // Minimum number of trades in recent history
    GAS_PRICE_THRESHOLD: 100,       // Maximum gas price in gwei
    MIN_PROFIT_USD: 50,             // Minimum profit in USD
    HEALTH_SCORE_THRESHOLD: 70,     // Minimum health score (0-100)
    MAX_POOL_SHARE_BPS: 500,        // Maximum trade size as percentage of pool (5%)
    EXTREME_CONDITIONS: {
        MAX_PRICE_CHANGE_BPS: 1000, // 10% sudden price change threshold
        MIN_LIQUIDITY_RATIO: 0.5,   // Minimum liquidity ratio between pools
        MANIPULATION_RISK_THRESHOLD: 0.7,  // Risk threshold for manipulation
        FLASH_LOAN_RISK_THRESHOLD: 0.3,    // Risk threshold for flash loan activity
        MAX_VOLATILITY: 0.2,        // Maximum acceptable volatility (20%)
        MIN_CONFIDENCE: 0.8,        // Minimum confidence score for trends
        VOLUME_CHANGE_THRESHOLD: 0.3 // Maximum acceptable volume change (30%)
    },
    TREND_ANALYSIS: {
        SHORT_WINDOW: 3600,         // 1 hour in seconds
        MEDIUM_WINDOW: 86400,       // 24 hours in seconds
        LONG_WINDOW: 604800,        // 7 days in seconds
        MIN_DATA_POINTS: 10,        // Minimum number of data points for analysis
        TREND_THRESHOLD: 0.01,      // Minimum slope for trend detection
        BREAKOUT_THRESHOLD: 0.05    // 5% threshold for breakout detection
    }
} as const; 