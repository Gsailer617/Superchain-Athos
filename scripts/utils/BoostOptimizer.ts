import { BigNumberish } from 'ethers';

interface BoostMetrics {
    successRate: number;
    averageProfit: number;
    volatilityScore: number;
    liquidityScore: number;
    gasEfficiency: number;
}

interface BoostConfig {
    maxBoostMultiplier: number;    // Maximum leverage multiplier (e.g., 3x)
    minSuccessRate: number;        // Minimum required success rate to maintain boost
    targetProfitMargin: number;    // Target profit margin in percentage
    gasMultiplierThreshold: number;// Minimum ratio of profit to gas cost
    volatilityWeight: number;      // Weight of volatility in boost calculation
}

export class BoostOptimizer {
    private readonly DEFAULT_CONFIG: BoostConfig = {
        maxBoostMultiplier: 3,     // 3x maximum leverage
        minSuccessRate: 0.95,      // 95% success rate required
        targetProfitMargin: 0.5,   // 0.5% minimum profit margin
        gasMultiplierThreshold: 10, // Profit should be 10x gas cost
        volatilityWeight: 0.3      // 30% weight to volatility
    };

    private tradeHistory: Map<string, {
        timestamp: number;
        success: boolean;
        profit: number;
        gasUsed: number;
        boostUsed: number;
    }[]> = new Map();

    constructor(private config: BoostConfig = {} as BoostConfig) {
        this.config = { ...this.DEFAULT_CONFIG, ...config };
    }

    calculateOptimalBoost(
        pair: string,
        currentPrice: number,
        volatility24h: number,
        liquidityUSD: number,
        estimatedGasCost: number,
        baseTradeSize: number
    ): {
        boostMultiplier: number;
        reason: string;
        metrics: BoostMetrics;
    } {
        // Calculate metrics
        const metrics = this.calculateMetrics(pair, volatility24h, liquidityUSD, estimatedGasCost);
        
        // Base multiplier starts at 1
        let multiplier = 1;
        let reason = '';

        // Adjust based on success rate
        if (metrics.successRate >= this.config.minSuccessRate) {
            multiplier *= 1 + (metrics.successRate - this.config.minSuccessRate);
            reason = 'High success rate allows increased boost. ';
        }

        // Adjust for volatility
        const volatilityFactor = 1 - (metrics.volatilityScore * this.config.volatilityWeight);
        multiplier *= volatilityFactor;
        reason += metrics.volatilityScore > 0.5 ? 'Reduced boost due to high volatility. ' : 'Stable market conditions. ';

        // Adjust for liquidity
        const liquidityFactor = Math.min(1, liquidityUSD / (baseTradeSize * multiplier * 20));
        multiplier *= liquidityFactor;
        if (liquidityFactor < 1) {
            reason += 'Boost limited by available liquidity. ';
        }

        // Adjust for gas efficiency
        if (metrics.gasEfficiency < this.config.gasMultiplierThreshold) {
            const gasAdjustment = metrics.gasEfficiency / this.config.gasMultiplierThreshold;
            multiplier *= gasAdjustment;
            reason += 'Boost reduced due to high gas costs. ';
        }

        // Cap at maximum boost
        multiplier = Math.min(multiplier, this.config.maxBoostMultiplier);
        
        return {
            boostMultiplier: multiplier,
            reason: reason.trim(),
            metrics
        };
    }

    private calculateMetrics(
        pair: string,
        volatility24h: number,
        liquidityUSD: number,
        estimatedGasCost: number
    ): BoostMetrics {
        const history = this.tradeHistory.get(pair) || [];
        const recent = history.slice(-50); // Look at last 50 trades

        // Calculate success rate
        const successRate = recent.length > 0 
            ? recent.filter(t => t.success).length / recent.length
            : 0.5; // Start conservative

        // Calculate average profit
        const averageProfit = recent.length > 0
            ? recent.reduce((sum, t) => sum + t.profit, 0) / recent.length
            : 0;

        // Normalize volatility to 0-1 score
        const volatilityScore = Math.min(volatility24h / 0.1, 1); // 10% daily volatility = score of 1

        // Calculate liquidity score (0-1)
        const liquidityScore = Math.min(liquidityUSD / 1000000, 1); // $1M liquidity = score of 1

        // Calculate gas efficiency
        const recentGasEfficiency = recent.length > 0
            ? recent.reduce((sum, t) => sum + (t.profit / t.gasUsed), 0) / recent.length
            : this.config.gasMultiplierThreshold;

        return {
            successRate,
            averageProfit,
            volatilityScore,
            liquidityScore,
            gasEfficiency: recentGasEfficiency
        };
    }

    recordTrade(
        pair: string,
        success: boolean,
        profit: number,
        gasUsed: number,
        boostUsed: number
    ): void {
        const history = this.tradeHistory.get(pair) || [];
        history.push({
            timestamp: Date.now(),
            success,
            profit,
            gasUsed,
            boostUsed
        });

        // Keep last 100 trades
        if (history.length > 100) {
            history.shift();
        }

        this.tradeHistory.set(pair, history);
    }

    getTradeStats(pair: string): {
        totalTrades: number;
        successRate: number;
        averageProfit: number;
        averageBoost: number;
        profitPerGas: number;
    } {
        const history = this.tradeHistory.get(pair) || [];
        if (history.length === 0) {
            return {
                totalTrades: 0,
                successRate: 0,
                averageProfit: 0,
                averageBoost: 1,
                profitPerGas: 0
            };
        }

        const successful = history.filter(t => t.success);
        return {
            totalTrades: history.length,
            successRate: successful.length / history.length,
            averageProfit: history.reduce((sum, t) => sum + t.profit, 0) / history.length,
            averageBoost: history.reduce((sum, t) => sum + t.boostUsed, 0) / history.length,
            profitPerGas: history.reduce((sum, t) => sum + (t.profit / t.gasUsed), 0) / history.length
        };
    }

    // New public methods for accessing trade history
    getAllPairs(): string[] {
        return Array.from(this.tradeHistory.keys());
    }

    getTradesForPair(pair: string): {
        timestamp: number;
        success: boolean;
        profit: number;
        gasUsed: number;
        boostUsed: number;
    }[] {
        return this.tradeHistory.get(pair) || [];
    }

    getAllTrades(): [string, {
        timestamp: number;
        success: boolean;
        profit: number;
        gasUsed: number;
        boostUsed: number;
    }[]][] {
        return Array.from(this.tradeHistory.entries());
    }
} 