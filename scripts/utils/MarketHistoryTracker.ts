import { DEX } from '../types/dex';
import { NumericUtils } from './NumericUtils';

interface MarketSnapshot {
    timestamp: number;
    priceUSD: bigint;
    liquidityUSD: bigint;
    volume24h: bigint;
    volatility: number;
    manipulationRisk: number;
    flashLoanActivity: number;
    blockNumber: number;
}

interface PoolSnapshot {
    timestamp: number;
    reserves: {
        reserve0: bigint;
        reserve1: bigint;
    };
    totalLiquidity: bigint;
    trades: {
        count: number;
        volume: bigint;
    };
}

export class MarketHistoryTracker {
    private static readonly MAX_HISTORY_DAYS = 30; // Keep 30 days of history
    private static readonly SNAPSHOT_INTERVAL = 5 * 60; // 5 minutes in seconds
    private static readonly CLEANUP_INTERVAL = 24 * 60 * 60; // 24 hours in seconds

    private marketHistory: Map<string, MarketSnapshot[]> = new Map();
    private poolHistory: Map<string, PoolSnapshot[]> = new Map();
    private lastCleanup: number = 0;

    constructor() {
        this.cleanupOldData();
    }

    async addMarketSnapshot(
        tokenIn: string,
        tokenOut: string,
        dex: DEX,
        snapshot: MarketSnapshot
    ): Promise<void> {
        const key = this.getMarketKey(tokenIn, tokenOut, dex);
        if (!this.marketHistory.has(key)) {
            this.marketHistory.set(key, []);
        }

        const history = this.marketHistory.get(key)!;
        history.push(snapshot);

        // Cleanup old data if needed
        await this.cleanupOldData();
    }

    async addPoolSnapshot(
        tokenIn: string,
        tokenOut: string,
        dex: DEX,
        snapshot: PoolSnapshot
    ): Promise<void> {
        const key = this.getMarketKey(tokenIn, tokenOut, dex);
        if (!this.poolHistory.has(key)) {
            this.poolHistory.set(key, []);
        }

        const history = this.poolHistory.get(key)!;
        history.push(snapshot);
    }

    getMarketHistory(
        tokenIn: string,
        tokenOut: string,
        dex: DEX,
        timeframeSeconds: number
    ): MarketSnapshot[] {
        const key = this.getMarketKey(tokenIn, tokenOut, dex);
        const history = this.marketHistory.get(key) || [];
        const cutoffTime = Date.now() / 1000 - timeframeSeconds;
        
        return history.filter(snapshot => snapshot.timestamp >= cutoffTime);
    }

    getPoolHistory(
        tokenIn: string,
        tokenOut: string,
        dex: DEX,
        timeframeSeconds: number
    ): PoolSnapshot[] {
        const key = this.getMarketKey(tokenIn, tokenOut, dex);
        const history = this.poolHistory.get(key) || [];
        const cutoffTime = Date.now() / 1000 - timeframeSeconds;
        
        return history.filter(snapshot => snapshot.timestamp >= cutoffTime);
    }

    analyzeVolatilityTrend(
        tokenIn: string,
        tokenOut: string,
        dex: DEX,
        timeframeSeconds: number
    ): {
        trend: 'increasing' | 'decreasing' | 'stable';
        averageVolatility: number;
        maxVolatility: number;
        volatilityScore: number;
    } {
        const history = this.getMarketHistory(tokenIn, tokenOut, dex, timeframeSeconds);
        if (history.length < 2) {
            return {
                trend: 'stable',
                averageVolatility: 0,
                maxVolatility: 0,
                volatilityScore: 0
            };
        }

        const volatilities = history.map(h => h.volatility);
        const avgVolatility = volatilities.reduce((a, b) => a + b) / volatilities.length;
        const maxVolatility = Math.max(...volatilities);

        // Calculate trend using linear regression
        const xValues = history.map((_, i) => i);
        const yValues = volatilities;
        const slope = this.calculateTrendSlope(xValues, yValues);

        return {
            trend: slope > 0.01 ? 'increasing' : slope < -0.01 ? 'decreasing' : 'stable',
            averageVolatility: avgVolatility,
            maxVolatility,
            volatilityScore: this.calculateVolatilityScore(volatilities)
        };
    }

    analyzeLiquidityTrend(
        tokenIn: string,
        tokenOut: string,
        dex: DEX,
        timeframeSeconds: number
    ): {
        trend: 'increasing' | 'decreasing' | 'stable';
        averageLiquidity: bigint;
        liquidityChange: number;
        healthScore: number;
    } {
        const history = this.getPoolHistory(tokenIn, tokenOut, dex, timeframeSeconds);
        if (history.length < 2) {
            return {
                trend: 'stable',
                averageLiquidity: 0n,
                liquidityChange: 0,
                healthScore: 0
            };
        }

        const liquidities = history.map(h => Number(h.totalLiquidity));
        const avgLiquidity = NumericUtils.average(history.map(h => h.totalLiquidity));
        
        // Calculate percentage change
        const firstLiquidity = liquidities[0];
        const lastLiquidity = liquidities[liquidities.length - 1];
        const change = (lastLiquidity - firstLiquidity) / firstLiquidity;

        // Calculate trend
        const slope = this.calculateTrendSlope(
            history.map((_, i) => i),
            liquidities
        );

        return {
            trend: slope > 0.01 ? 'increasing' : slope < -0.01 ? 'decreasing' : 'stable',
            averageLiquidity: avgLiquidity,
            liquidityChange: change,
            healthScore: this.calculateLiquidityHealthScore(change, liquidities)
        };
    }

    private calculateTrendSlope(x: number[], y: number[]): number {
        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);

        return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    }

    private calculateVolatilityScore(volatilities: number[]): number {
        if (volatilities.length === 0) return 0;

        const recentWeight = 0.6; // More weight to recent volatility
        const historicalWeight = 0.4;

        const recentVolatility = volatilities[volatilities.length - 1];
        const historicalAvg = volatilities.slice(0, -1).reduce((a, b) => a + b, 0) / 
            (volatilities.length - 1);

        return (recentVolatility * recentWeight + historicalAvg * historicalWeight);
    }

    private calculateLiquidityHealthScore(change: number, liquidities: number[]): number {
        // Penalize negative changes more heavily
        const changeScore = change >= 0 ? 
            Math.min(change * 100, 100) : 
            Math.max(100 + change * 200, 0);

        // Calculate stability score
        const avg = liquidities.reduce((a, b) => a + b) / liquidities.length;
        const variance = liquidities.reduce((sum, val) => {
            const diff = val - avg;
            return sum + (diff * diff);
        }, 0) / liquidities.length;
        const stabilityScore = Math.max(0, 100 - (Math.sqrt(variance) / avg * 100));

        // Combine scores
        return (changeScore * 0.7 + stabilityScore * 0.3);
    }

    private getMarketKey(tokenIn: string, tokenOut: string, dex: DEX): string {
        return `${dex}-${tokenIn}-${tokenOut}`;
    }

    private async cleanupOldData(): Promise<void> {
        const now = Date.now() / 1000;
        if (now - this.lastCleanup < MarketHistoryTracker.CLEANUP_INTERVAL) {
            return;
        }

        const cutoffTime = now - MarketHistoryTracker.MAX_HISTORY_DAYS * 24 * 60 * 60;

        // Cleanup market history
        for (const [key, history] of this.marketHistory.entries()) {
            this.marketHistory.set(
                key,
                history.filter(snapshot => snapshot.timestamp >= cutoffTime)
            );
        }

        // Cleanup pool history
        for (const [key, history] of this.poolHistory.entries()) {
            this.poolHistory.set(
                key,
                history.filter(snapshot => snapshot.timestamp >= cutoffTime)
            );
        }

        this.lastCleanup = now;
    }
} 