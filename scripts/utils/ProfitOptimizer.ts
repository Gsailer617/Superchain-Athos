import { BigNumberish } from 'ethers';
import { ArbitrageOpportunity, DEX } from '../types/dex';
import { QUALITY_THRESHOLDS } from '../config/constants';
import { NumericUtils } from './NumericUtils';
import { COMMON_TOKENS } from '../config/tokens';
import { StableSwapPool } from '../types/dex';

interface ProfitMetrics {
    expectedProfit: bigint;
    confidence: number;
    riskScore: number;
    gasEfficiency: number;
    liquidityScore: number;
    volatilityScore: number;
}

interface OptimizationResult {
    adjustedAmount: bigint;
    expectedProfit: bigint;
    confidence: number;
    reason: string;
}

interface StableSwapMetrics extends ProfitMetrics {
    virtualPriceDeviation: number;
    amplifierUtilization: number;
    poolBalance: bigint[];
}

export class ProfitOptimizer {
    private readonly MIN_CONFIDENCE = 0.5;
    private readonly MIN_LIQUIDITY_SCORE = 0.5;
    private readonly MAX_RISK_SCORE = 0.5;
    private readonly MIN_GAS_EFFICIENCY = 2;
    private readonly STABLE_SWAP_THRESHOLDS = {
        MAX_VIRTUAL_PRICE_DEVIATION: 0.002,
        MIN_AMPLIFIER_UTILIZATION: 0.3,
        MAX_POOL_IMBALANCE: 0.3,
        MIN_POOL_LIQUIDITY_RATIO: 0.05
    };

    private profitHistory: Map<string, {
        timestamp: number;
        profit: bigint;
        gasUsed: bigint;
        success: boolean;
    }[]> = new Map();

    constructor() {}

    async optimizeArbitrageOpportunity(
        opportunity: ArbitrageOpportunity,
        gasPrice: bigint,
        currentMarketConditions: {
            volatility: number;
            liquidity: bigint;
            volume24h: bigint;
        }
    ): Promise<OptimizationResult> {
        // Check if this is a stable swap opportunity
        const isStableSwap = this.isStableSwapOpportunity(opportunity);
        
        const metrics = await this.calculateProfitMetrics(
            opportunity,
            gasPrice,
            currentMarketConditions,
            isStableSwap
        );
        
        if (isStableSwap) {
            const stableMetrics = metrics as StableSwapMetrics;
            if (!this.meetsStableSwapThresholds(stableMetrics)) {
                return {
                    adjustedAmount: 0n,
                    expectedProfit: 0n,
                    confidence: 0,
                    reason: "Stable swap thresholds not met"
                };
            }
        }

        // Check if opportunity meets minimum thresholds
        if (!this.meetsMinimumThresholds(metrics)) {
            return {
                adjustedAmount: 0n,
                expectedProfit: 0n,
                confidence: 0,
                reason: 'Does not meet minimum thresholds'
            };
        }

        // Optimize trade size based on metrics
        const adjustedAmount = this.optimizeTradeSize(
            opportunity.amountIn,
            metrics,
            currentMarketConditions
        );

        // Recalculate expected profit with adjusted amount
        const adjustedProfit = this.calculateAdjustedProfit(
            adjustedAmount,
            opportunity,
            metrics
        );

        return {
            adjustedAmount,
            expectedProfit: adjustedProfit,
            confidence: metrics.confidence,
            reason: 'Optimized for maximum profit'
        };
    }

    private async calculateProfitMetrics(
        opportunity: ArbitrageOpportunity,
        gasPrice: bigint,
        marketConditions: {
            volatility: number;
            liquidity: bigint;
            volume24h: bigint;
        },
        isStableSwap: boolean = false
    ): Promise<ProfitMetrics> {
        const gasCost = this.estimateGasCost(opportunity, gasPrice);
        const netProfit = opportunity.expectedProfit - gasCost;

        const gasEfficiency = Number(netProfit) / Number(gasCost);
        const liquidityScore = this.calculateLiquidityScore(marketConditions.liquidity);
        const volatilityScore = 1 - marketConditions.volatility; // Lower volatility is better
        const confidence = this.calculateConfidence(opportunity, marketConditions);
        const riskScore = this.calculateRiskScore(opportunity, marketConditions);

        return {
            expectedProfit: netProfit,
            confidence,
            riskScore,
            gasEfficiency,
            liquidityScore,
            volatilityScore
        };
    }

    private meetsMinimumThresholds(metrics: ProfitMetrics): boolean {
        return (
            metrics.confidence >= this.MIN_CONFIDENCE &&
            metrics.liquidityScore >= this.MIN_LIQUIDITY_SCORE &&
            metrics.riskScore <= this.MAX_RISK_SCORE &&
            metrics.gasEfficiency >= this.MIN_GAS_EFFICIENCY &&
            metrics.expectedProfit > 0n
        );
    }

    private optimizeTradeSize(
        currentAmount: bigint,
        metrics: ProfitMetrics,
        marketConditions: {
            volatility: number;
            liquidity: bigint;
            volume24h: bigint;
        }
    ): bigint {
        // Start with current amount
        let optimizedAmount = currentAmount;

        // Adjust based on liquidity
        const maxLiquidityAmount = (marketConditions.liquidity * BigInt(QUALITY_THRESHOLDS.MAX_POOL_SHARE_BPS)) / BigInt(10000);
        optimizedAmount = optimizedAmount > maxLiquidityAmount ? maxLiquidityAmount : optimizedAmount;

        // Adjust based on volatility
        const volatilityFactor = BigInt(Math.floor((1 - marketConditions.volatility) * 100));
        optimizedAmount = (optimizedAmount * volatilityFactor) / BigInt(100);

        // Adjust based on gas efficiency
        if (metrics.gasEfficiency < this.MIN_GAS_EFFICIENCY) {
            const adjustmentFactor = BigInt(Math.floor((metrics.gasEfficiency / this.MIN_GAS_EFFICIENCY) * 100));
            optimizedAmount = (optimizedAmount * adjustmentFactor) / BigInt(100);
        }

        // Ensure minimum profitable amount
        const minProfitableAmount = this.calculateMinProfitableAmount(metrics);
        if (optimizedAmount < minProfitableAmount) {
            return 0n; // Not profitable at any size
        }

        return optimizedAmount;
    }

    private calculateAdjustedProfit(
        adjustedAmount: bigint,
        opportunity: ArbitrageOpportunity,
        metrics: ProfitMetrics
    ): bigint {
        if (adjustedAmount === 0n) return 0n;

        const profitRatio = opportunity.expectedProfit / opportunity.amountIn;
        let adjustedProfit = (adjustedAmount * profitRatio);

        // Apply confidence factor
        adjustedProfit = (adjustedProfit * BigInt(Math.floor(metrics.confidence * 100))) / BigInt(100);

        return adjustedProfit;
    }

    private estimateGasCost(
        opportunity: ArbitrageOpportunity,
        gasPrice: bigint
    ): bigint {
        // Estimate gas usage based on route complexity
        const baseGas = 300000n; // Base gas for flash loan
        const routeGas = BigInt(opportunity.route.length * 100000); // Additional gas per hop
        const totalGas = baseGas + routeGas;

        return totalGas * gasPrice;
    }

    private calculateLiquidityScore(liquidity: bigint): number {
        const minLiquidity = BigInt(QUALITY_THRESHOLDS.MIN_LIQUIDITY_USD) * BigInt(1e18);
        return Math.min(Number(liquidity) / Number(minLiquidity), 1);
    }

    private calculateConfidence(
        opportunity: ArbitrageOpportunity,
        marketConditions: {
            volatility: number;
            liquidity: bigint;
            volume24h: bigint;
        }
    ): number {
        // Base confidence on historical success
        const history = this.profitHistory.get(this.getOpportunityKey(opportunity)) || [];
        const recentHistory = history.slice(-50); // Look at last 50 trades
        
        if (recentHistory.length === 0) return 0.5; // Start conservative

        // Calculate success rate
        const successRate = recentHistory.filter(h => h.success).length / recentHistory.length;

        // Adjust for market conditions
        const volatilityFactor = 1 - marketConditions.volatility;
        const liquidityFactor = this.calculateLiquidityScore(marketConditions.liquidity);
        const volumeFactor = Math.min(Number(marketConditions.volume24h) / QUALITY_THRESHOLDS.MIN_VOLUME_24H, 1);

        return (successRate * 0.4 + volatilityFactor * 0.3 + liquidityFactor * 0.2 + volumeFactor * 0.1);
    }

    private calculateRiskScore(
        opportunity: ArbitrageOpportunity,
        marketConditions: {
            volatility: number;
            liquidity: bigint;
            volume24h: bigint;
        }
    ): number {
        // Higher score means higher risk
        const volatilityRisk = marketConditions.volatility;
        const liquidityRisk = 1 - this.calculateLiquidityScore(marketConditions.liquidity);
        const complexityRisk = (opportunity.route.length - 1) * 0.1; // More hops = more risk
        
        return (volatilityRisk * 0.4 + liquidityRisk * 0.4 + complexityRisk * 0.2);
    }

    private calculateMinProfitableAmount(metrics: ProfitMetrics): bigint {
        // Calculate minimum amount where gas costs don't eat too much into profits
        const minProfit = BigInt(QUALITY_THRESHOLDS.MIN_PROFIT_USD) * BigInt(1e18);
        return (minProfit * BigInt(100)) / BigInt(Math.floor(metrics.gasEfficiency));
    }

    private getOpportunityKey(opportunity: ArbitrageOpportunity): string {
        return `${opportunity.tokenIn}-${opportunity.tokenOut}-${opportunity.sourceDex}-${opportunity.targetDex}`;
    }

    recordTradeResult(
        opportunity: ArbitrageOpportunity,
        profit: bigint,
        gasUsed: bigint,
        success: boolean
    ): void {
        const key = this.getOpportunityKey(opportunity);
        const history = this.profitHistory.get(key) || [];
        
        history.push({
            timestamp: Date.now(),
            profit,
            gasUsed,
            success
        });

        // Keep last 100 trades
        if (history.length > 100) {
            history.shift();
        }

        this.profitHistory.set(key, history);
    }

    private isStableSwapOpportunity(opportunity: ArbitrageOpportunity): boolean {
        const stableTokens = new Set([
            COMMON_TOKENS.USDC,
            COMMON_TOKENS.USDT,
            COMMON_TOKENS.DAI,
            COMMON_TOKENS.USDbC
        ]);
        return stableTokens.has(opportunity.tokenIn) && stableTokens.has(opportunity.tokenOut);
    }

    private async calculateStableSwapMetrics(
        opportunity: ArbitrageOpportunity,
        pool: StableSwapPool,
        gasPrice: bigint,
        currentMarketConditions: {
            volatility: number;
            liquidity: bigint;
            volume24h: bigint;
        }
    ): Promise<StableSwapMetrics> {
        const baseMetrics = await this.calculateProfitMetrics(
            opportunity,
            gasPrice,
            currentMarketConditions
        );

        // Calculate virtual price deviation
        const virtualPriceDeviation = Math.abs(1 - Number(pool.virtualPrice) / 1e18);

        // Calculate amplifier utilization
        const totalBalance = pool.poolBalance.reduce((a: bigint, b: bigint) => a + b, 0n);
        const idealBalance = totalBalance / BigInt(pool.poolBalance.length);
        const maxDeviation = pool.poolBalance.reduce((max: number, balance: bigint) => {
            const deviation = Math.abs(Number(balance - idealBalance) / Number(idealBalance));
            return Math.max(max, deviation);
        }, 0);
        const amplifierUtilization = 1 - maxDeviation;

        return {
            ...baseMetrics,
            virtualPriceDeviation,
            amplifierUtilization,
            poolBalance: pool.poolBalance
        };
    }

    private meetsStableSwapThresholds(metrics: StableSwapMetrics): boolean {
        if (metrics.virtualPriceDeviation > this.STABLE_SWAP_THRESHOLDS.MAX_VIRTUAL_PRICE_DEVIATION) {
            return false;
        }
        
        if (metrics.amplifierUtilization < this.STABLE_SWAP_THRESHOLDS.MIN_AMPLIFIER_UTILIZATION) {
            return false;
        }
        
        // Check pool balance
        const totalBalance = metrics.poolBalance.reduce((a, b) => a + b, 0n);
        const idealBalance = totalBalance / BigInt(metrics.poolBalance.length);
        
        for (const balance of metrics.poolBalance) {
            const deviation = Math.abs(Number(balance - idealBalance) / Number(idealBalance));
            if (deviation > this.STABLE_SWAP_THRESHOLDS.MAX_POOL_IMBALANCE) {
                return false;
            }
        }
        
        return true;
    }
} 