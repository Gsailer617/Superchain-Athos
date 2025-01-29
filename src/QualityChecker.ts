import { ethers } from 'ethers';
import { LiquidityAnalysis, PoolData, Trade, VolumeData, VolumeMetrics, PricePoint, TrendAnalysis } from '../scripts/types/dex';
import { QUALITY_THRESHOLDS } from '../scripts/config/constants';

export class QualityChecker {
    private priceHistory: PricePoint[] = [];
    private volumeHistory: VolumeData[] = [];
    private readonly MIN_DATA_POINTS = 10;
    private readonly TREND_WINDOW = 24; // hours

    constructor() {}

    public async checkTradeQuality(
        poolData: PoolData,
        recentTrades: Trade[],
        volumeData: VolumeData,
        amountIn: bigint
    ): Promise<{
        isValid: boolean;
        reason?: string;
        adjustedAmount?: bigint;
    }> {
        // Check liquidity depth
        const liquidityAnalysis = await this.analyzeLiquidity(poolData, recentTrades);
        if (liquidityAnalysis.overallHealth < QUALITY_THRESHOLDS.HEALTH_SCORE_THRESHOLD) {
            return {
                isValid: false,
                reason: 'Insufficient liquidity health score'
            };
        }

        // Check volume metrics
        const volumeMetrics = this.analyzeVolume(volumeData);
        if (volumeMetrics.overallScore < QUALITY_THRESHOLDS.MIN_VOLUME_SCORE) {
            return {
                isValid: false,
                reason: 'Insufficient volume score'
            };
        }

        // Calculate optimal trade size
        const adjustedAmount = this.calculateOptimalTradeSize(
            amountIn,
            poolData,
            liquidityAnalysis,
            volumeMetrics
        );

        return {
            isValid: true,
            adjustedAmount
        };
    }

    private async analyzeLiquidity(
        poolData: PoolData,
        recentTrades: Trade[]
    ): Promise<LiquidityAnalysis> {
        const liquidityDepth = Number(poolData.totalLiquidity) / 1e18;
        const volumeScore = this.calculateVolumeScore(recentTrades);
        const priceStability = await this.calculatePriceStability(recentTrades);

        const liquidityScore = this.calculateLiquidityScore(liquidityDepth);
        const overallHealth = (liquidityScore + volumeScore + priceStability) / 3;

        return {
            liquidityDepth,
            liquidityScore,
            volumeScore,
            priceStability,
            overallHealth
        };
    }

    private analyzeVolume(volumeData: VolumeData): VolumeMetrics {
        const trendScore = this.calculateVolumeTrendScore(volumeData);
        const expansionScore = this.calculateVolumeExpansionScore(volumeData);
        const distributionScore = this.calculateVolumeDistributionScore(volumeData);
        
        const overallScore = (trendScore + expansionScore + distributionScore) / 3;

        return {
            trendScore,
            expansionScore,
            distributionScore,
            overallScore
        };
    }

    private calculateOptimalTradeSize(
        amountIn: bigint,
        poolData: PoolData,
        liquidityAnalysis: LiquidityAnalysis,
        volumeMetrics: VolumeMetrics
    ): bigint {
        const maxTradeSize = (poolData.totalLiquidity * BigInt(QUALITY_THRESHOLDS.MAX_POOL_SHARE_BPS)) / BigInt(10000);
        const healthAdjustment = BigInt(Math.floor(liquidityAnalysis.overallHealth * 100));
        const volumeAdjustment = BigInt(Math.floor(volumeMetrics.overallScore * 100));
        
        let adjustedAmount = amountIn;
        
        // Adjust based on health and volume scores
        adjustedAmount = (adjustedAmount * healthAdjustment) / BigInt(100);
        adjustedAmount = (adjustedAmount * volumeAdjustment) / BigInt(100);
        
        // Cap at max trade size
        return adjustedAmount > maxTradeSize ? maxTradeSize : adjustedAmount;
    }

    private calculateVolumeScore(trades: Trade[]): number {
        if (trades.length === 0) return 0;
        
        const totalVolume = trades.reduce((sum, trade) => sum + Number(trade.amount), 0);
        const averageVolume = totalVolume / trades.length;
        
        return Math.min(averageVolume / QUALITY_THRESHOLDS.MIN_VOLUME_24H, 1);
    }

    private async calculatePriceStability(trades: Trade[]): Promise<number> {
        if (trades.length < 2) return 0;
        
        const prices = trades.map(t => Number(t.price));
        const volatility = this.calculateVolatility(prices);
        
        return Math.max(0, 1 - volatility);
    }

    private calculateVolatility(prices: number[]): number {
        if (prices.length < 2) return 0;
        
        const returns: number[] = [];
        for (let i = 1; i < prices.length; i++) {
            returns.push(Math.log(prices[i] / prices[i - 1]));
        }
        
        const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
        
        return Math.sqrt(variance);
    }

    private calculateLiquidityScore(liquidityDepth: number): number {
        return Math.min(liquidityDepth / QUALITY_THRESHOLDS.MIN_LIQUIDITY_USD, 1);
    }

    private calculateVolumeTrendScore(volumeData: VolumeData): number {
        const hourlyChange = volumeData.volumeChange1h;
        const dailyChange = volumeData.volumeChange24h;
        
        // Weight recent changes more heavily
        return Math.min(1, (hourlyChange * 0.7 + dailyChange * 0.3));
    }

    private calculateVolumeExpansionScore(volumeData: VolumeData): number {
        const hourlyVolume = volumeData.volume1h;
        const dailyAvgHourlyVolume = volumeData.volume24h / 24;
        
        return Math.min(1, hourlyVolume / dailyAvgHourlyVolume);
    }

    private calculateVolumeDistributionScore(volumeData: VolumeData): number {
        const { largeTradesPercent, mediumTradesPercent, smallTradesPercent } = volumeData.volumeDistribution;
        
        // Prefer a balanced distribution with some large trades
        const idealLarge = 0.3;
        const idealMedium = 0.4;
        const idealSmall = 0.3;
        
        const largeScore = 1 - Math.abs(largeTradesPercent - idealLarge);
        const mediumScore = 1 - Math.abs(mediumTradesPercent - idealMedium);
        const smallScore = 1 - Math.abs(smallTradesPercent - idealSmall);
        
        return (largeScore + mediumScore + smallScore) / 3;
    }
} 