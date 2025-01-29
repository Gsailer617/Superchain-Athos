import { BigNumberish } from 'ethers';
import { QUALITY_THRESHOLDS } from '../config/constants';

export enum UrgencyLevel {
    LOW = 'LOW',         // Can wait, prioritize cost savings
    MEDIUM = 'MEDIUM',   // Balance between speed and cost
    HIGH = 'HIGH',       // Need fast inclusion, willing to pay premium
    CRITICAL = 'CRITICAL' // Must be included in next block
}

interface PairGasStats {
    averageGasUsed: bigint;
    successRate: number;
    failureCount: number;
    optimalGasMultiplier: number;
    historicalGasPrices: { timestamp: number; gasPrice: bigint; success: boolean }[];
    lastOptimization: number;
    marketConditions: {
        congestionLevel: number;  // 0-100
        volatilityLevel: number;  // 0-100
        lastBlockBaseFee: bigint;
        recentFailurePattern: boolean;
    };
    timeBasedStats: {
        hourOfDay: { [hour: number]: { successRate: number; avgGasPrice: bigint; samples: number } };
        dayOfWeek: { [day: number]: { successRate: number; avgGasPrice: bigint; samples: number } };
    };
    predictions: {
        shortTermEMA: bigint;  // 5-block EMA
        longTermEMA: bigint;   // 20-block EMA
        lastPrediction: {
            predictedPrice: bigint;
            confidence: number;
            timestamp: number;
        };
        predictionAccuracy: number[];  // Store last 20 prediction accuracies
    };
    urgencyMultipliers: {
        [UrgencyLevel.LOW]: number;
        [UrgencyLevel.MEDIUM]: number;
        [UrgencyLevel.HIGH]: number;
        [UrgencyLevel.CRITICAL]: number;
    };
    mevStats: {
        recentMEVActivity: {
            frontrunAttempts: number;
            sandwichAttempts: number;
            backrunAttempts: number;
            timestamp: number;
        }[];
        mevRiskLevel: number;  // 0-100
        lastMEVCheck: number;
        protectedTransactions: number;
        mevLosses: bigint;
    };
}

interface GasStats {
    averageGasUsed: bigint;
    successRate: number;
    failureCount: number;
    optimalGasMultiplier: number;
    historicalGasPrices: {
        timestamp: number;
        gasPrice: bigint;
        success: boolean;
    }[];
    lastOptimization: number;
}

export class PairGasOptimizer {
    private pairGasStats: Map<string, PairGasStats> = new Map();
    private readonly OPTIMIZATION_INTERVAL = 3600000; // 1 hour
    private readonly MIN_DATA_POINTS = 10;
    private readonly MAX_HISTORY_POINTS = 100;
    private readonly BASE_GAS_MULTIPLIER = 1.1;
    private readonly MAX_GAS_MULTIPLIER = 2.0;
    
    private readonly CONGESTION_THRESHOLDS = {
        LOW: 30,
        MEDIUM: 60,
        HIGH: 90
    };

    private readonly GAS_ADJUSTMENTS = {
        CONGESTION: {
            LOW: 1.0,
            MEDIUM: 1.2,
            HIGH: 1.5
        },
        VOLATILITY: {
            LOW: 1.0,
            MEDIUM: 1.15,
            HIGH: 1.3
        },
        FAILURE_PATTERN: 1.25
    };

    private readonly EMA_ALPHA_SHORT = 0.2;  // 5-block EMA
    private readonly EMA_ALPHA_LONG = 0.05;  // 20-block EMA
    private readonly PREDICTION_WINDOW = 3;  // blocks ahead to predict
    
    private readonly DEFAULT_URGENCY_MULTIPLIERS = {
        [UrgencyLevel.LOW]: 0.9,      // 90% of base price
        [UrgencyLevel.MEDIUM]: 1.0,    // Base price
        [UrgencyLevel.HIGH]: 1.5,      // 150% of base price
        [UrgencyLevel.CRITICAL]: 2.0    // 200% of base price
    };

    private readonly CONGESTION_URGENCY_MULTIPLIERS = {
        [UrgencyLevel.LOW]: {
            LOW: 0.9,
            MEDIUM: 0.95,
            HIGH: 1.0
        },
        [UrgencyLevel.MEDIUM]: {
            LOW: 1.0,
            MEDIUM: 1.1,
            HIGH: 1.2
        },
        [UrgencyLevel.HIGH]: {
            LOW: 1.3,
            MEDIUM: 1.5,
            HIGH: 1.8
        },
        [UrgencyLevel.CRITICAL]: {
            LOW: 1.5,
            MEDIUM: 1.8,
            HIGH: 2.5
        }
    };

    private readonly MEV_THRESHOLDS = {
        LOW_RISK: 30,
        MEDIUM_RISK: 60,
        HIGH_RISK: 80
    };

    private readonly MEV_GAS_ADJUSTMENTS = {
        LOW_RISK: 1.1,
        MEDIUM_RISK: 1.3,
        HIGH_RISK: 1.5,
        FRONTRUN_PROTECTION: 1.4,
        SANDWICH_PROTECTION: 1.6,
        BACKRUN_PROTECTION: 1.2
    };

    private readonly MEV_MONITORING_WINDOW = 3600000; // 1 hour

    updatePairGasStats(
        pair: string,
        gasUsed: bigint,
        gasPrice: bigint,
        success: boolean,
        baseFee: bigint
    ): void {
        const stats = this.pairGasStats.get(pair) || this.getDefaultStats();
        const now = new Date();
        
        stats.averageGasUsed = (stats.averageGasUsed * BigInt(stats.historicalGasPrices.length) + gasUsed) / 
            BigInt(stats.historicalGasPrices.length + 1);
        
        if (!success) stats.failureCount++;
        
        stats.successRate = (stats.successRate * stats.historicalGasPrices.length + (success ? 100 : 0)) / 
            (stats.historicalGasPrices.length + 1);

        this.updateTimeBasedStats(stats, now, gasPrice, success);

        this.updateMarketConditions(stats, baseFee, success);

        stats.historicalGasPrices.push({
            timestamp: Date.now(),
            gasPrice,
            success
        });

        if (stats.historicalGasPrices.length > this.MAX_HISTORY_POINTS) {
            stats.historicalGasPrices.shift();
        }

        this.pairGasStats.set(pair, stats);
        this.optimizeGasMultiplier(pair);
    }

    updateMEVStats(
        pair: string, 
        mevActivity: { 
            type: 'frontrun' | 'sandwich' | 'backrun',
            detected: boolean,
            loss?: bigint
        }
    ): void {
        const stats = this.pairGasStats.get(pair) || this.getDefaultStats();
        const now = Date.now();

        // Clean up old MEV activity
        stats.mevStats.recentMEVActivity = stats.mevStats.recentMEVActivity.filter(
            activity => now - activity.timestamp < this.MEV_MONITORING_WINDOW
        );

        // Update MEV activity counts
        if (mevActivity.detected) {
            const newActivity = {
                frontrunAttempts: mevActivity.type === 'frontrun' ? 1 : 0,
                sandwichAttempts: mevActivity.type === 'sandwich' ? 1 : 0,
                backrunAttempts: mevActivity.type === 'backrun' ? 1 : 0,
                timestamp: now
            };
            stats.mevStats.recentMEVActivity.push(newActivity);

            if (mevActivity.loss) {
                stats.mevStats.mevLosses += mevActivity.loss;
            }
        }

        // Calculate MEV risk level
        this.updateMEVRiskLevel(stats);
        
        this.pairGasStats.set(pair, stats);
    }

    private updateTimeBasedStats(
        stats: PairGasStats,
        now: Date,
        gasPrice: bigint,
        success: boolean
    ): void {
        const hour = now.getHours();
        const day = now.getDay();

        if (!stats.timeBasedStats.hourOfDay[hour]) {
            stats.timeBasedStats.hourOfDay[hour] = { successRate: 0, avgGasPrice: 0n, samples: 0 };
        }
        const hourStats = stats.timeBasedStats.hourOfDay[hour];
        hourStats.successRate = (hourStats.successRate * hourStats.samples + (success ? 100 : 0)) / (hourStats.samples + 1);
        hourStats.avgGasPrice = (hourStats.avgGasPrice * BigInt(hourStats.samples) + gasPrice) / BigInt(hourStats.samples + 1);
        hourStats.samples++;

        if (!stats.timeBasedStats.dayOfWeek[day]) {
            stats.timeBasedStats.dayOfWeek[day] = { successRate: 0, avgGasPrice: 0n, samples: 0 };
        }
        const dayStats = stats.timeBasedStats.dayOfWeek[day];
        dayStats.successRate = (dayStats.successRate * dayStats.samples + (success ? 100 : 0)) / (dayStats.samples + 1);
        dayStats.avgGasPrice = (dayStats.avgGasPrice * BigInt(dayStats.samples) + gasPrice) / BigInt(dayStats.samples + 1);
        dayStats.samples++;
    }

    private updateMarketConditions(
        stats: PairGasStats,
        baseFee: bigint,
        success: boolean
    ): void {
        const prevBaseFee = stats.marketConditions.lastBlockBaseFee;
        const baseFeeDelta = prevBaseFee ? Number((baseFee - prevBaseFee) * 100n / prevBaseFee) : 0;
        stats.marketConditions.congestionLevel = Math.min(100, Math.max(0, 
            stats.marketConditions.congestionLevel + (baseFeeDelta > 10 ? 10 : baseFeeDelta)
        ));

        const recentPrices = stats.historicalGasPrices.slice(-5);
        if (recentPrices.length >= 2) {
            const priceChanges = recentPrices.slice(1).map((p, i) => 
                Number((p.gasPrice - recentPrices[i].gasPrice) * 100n / recentPrices[i].gasPrice)
            );
            const avgChange = priceChanges.reduce((a, b) => a + Math.abs(b), 0) / priceChanges.length;
            stats.marketConditions.volatilityLevel = Math.min(100, avgChange);
        }

        const recentTxs = stats.historicalGasPrices.slice(-3);
        stats.marketConditions.recentFailurePattern = recentTxs.length === 3 && 
            recentTxs.every(tx => !tx.success);

        stats.marketConditions.lastBlockBaseFee = baseFee;
    }

    private getTimeBasedMultiplier(stats: PairGasStats): number {
        const now = new Date();
        const hour = now.getHours();
        const day = now.getDay();

        const hourStats = stats.timeBasedStats.hourOfDay[hour];
        const dayStats = stats.timeBasedStats.dayOfWeek[day];

        if (!hourStats || !dayStats) return 1.0;

        const hourMultiplier = hourStats.successRate < 90 ? 1.1 : 1.0;
        const dayMultiplier = dayStats.successRate < 90 ? 1.05 : 1.0;

        return hourMultiplier * dayMultiplier;
    }

    private getMarketConditionMultiplier(stats: PairGasStats): number {
        let multiplier = 1.0;

        if (stats.marketConditions.congestionLevel >= this.CONGESTION_THRESHOLDS.HIGH) {
            multiplier *= this.GAS_ADJUSTMENTS.CONGESTION.HIGH;
        } else if (stats.marketConditions.congestionLevel >= this.CONGESTION_THRESHOLDS.MEDIUM) {
            multiplier *= this.GAS_ADJUSTMENTS.CONGESTION.MEDIUM;
        }

        if (stats.marketConditions.volatilityLevel >= this.CONGESTION_THRESHOLDS.HIGH) {
            multiplier *= this.GAS_ADJUSTMENTS.VOLATILITY.HIGH;
        } else if (stats.marketConditions.volatilityLevel >= this.CONGESTION_THRESHOLDS.MEDIUM) {
            multiplier *= this.GAS_ADJUSTMENTS.VOLATILITY.MEDIUM;
        }

        if (stats.marketConditions.recentFailurePattern) {
            multiplier *= this.GAS_ADJUSTMENTS.FAILURE_PATTERN;
        }

        return multiplier;
    }

    private optimizeGasMultiplier(pair: string): void {
        const stats = this.pairGasStats.get(pair)!;
        const now = Date.now();

        if (now - stats.lastOptimization < this.OPTIMIZATION_INTERVAL || 
            stats.historicalGasPrices.length < this.MIN_DATA_POINTS) {
            return;
        }

        const recentHistory = stats.historicalGasPrices.slice(-this.MIN_DATA_POINTS);
        const currentSuccessRate = recentHistory.filter(h => h.success).length / recentHistory.length * 100;

        const successfulPrices = recentHistory
            .filter(h => h.success)
            .map(h => Number(h.gasPrice));
        
        const avgSuccessfulGas = successfulPrices.length > 0 
            ? successfulPrices.reduce((a, b) => a + b) / successfulPrices.length 
            : 0;

        let newMultiplier = stats.optimalGasMultiplier;
        
        if (currentSuccessRate < 90 || stats.marketConditions.congestionLevel >= this.CONGESTION_THRESHOLDS.HIGH) {
            newMultiplier = Math.min(
                stats.optimalGasMultiplier * 1.1,
                this.MAX_GAS_MULTIPLIER
            );
        } else if (currentSuccessRate > 98 && 
                  stats.marketConditions.congestionLevel <= this.CONGESTION_THRESHOLDS.LOW &&
                  stats.optimalGasMultiplier > this.BASE_GAS_MULTIPLIER) {
            newMultiplier = Math.max(
                stats.optimalGasMultiplier * 0.95,
                this.BASE_GAS_MULTIPLIER
            );
        }

        stats.optimalGasMultiplier = newMultiplier;
        stats.lastOptimization = now;
        this.pairGasStats.set(pair, stats);
    }

    predictGasPrice(
        pair: string, 
        currentBaseFee: bigint,
        urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    ): { 
        predictedPrice: bigint;
        confidence: number;
    } {
        const prediction = this.calculateBasePrediction(pair, currentBaseFee);
        const urgencyMultiplier = this.getUrgencyMultiplier(
            this.pairGasStats.get(pair) || this.getDefaultStats(),
            urgency
        );
        
        return {
            predictedPrice: prediction.predictedPrice * BigInt(Math.floor(urgencyMultiplier * 100)) / 100n,
            confidence: prediction.confidence * (urgency === UrgencyLevel.CRITICAL ? 0.9 : 1.0)
        };
    }

    private calculateEMA(prevEMA: bigint, newValue: bigint, alpha: number): bigint {
        return prevEMA + ((newValue - prevEMA) * BigInt(Math.floor(alpha * 100))) / 100n;
    }

    private calculateTrendStrength(stats: PairGasStats): number {
        if (stats.historicalGasPrices.length < 5) return 0.5;

        const recentPrices = stats.historicalGasPrices.slice(-5);
        let trendCount = 0;

        for (let i = 1; i < recentPrices.length; i++) {
            const prev = recentPrices[i - 1].gasPrice;
            const curr = recentPrices[i].gasPrice;
            if (curr > prev) trendCount++;
            else if (curr < prev) trendCount--;
        }

        return Math.abs(trendCount) / 4; // Normalize to 0-1
    }

    private calculateMarketAdjustment(stats: PairGasStats): number {
        let adjustment = 1.0;

        // Adjust for congestion
        if (stats.marketConditions.congestionLevel > this.CONGESTION_THRESHOLDS.HIGH) {
            adjustment *= 1.2;
        }

        // Adjust for volatility
        if (stats.marketConditions.volatilityLevel > this.CONGESTION_THRESHOLDS.HIGH) {
            adjustment *= 1.15;
        }

        // Adjust for time-based patterns
        const timeMultiplier = this.getTimeBasedMultiplier(stats);
        adjustment *= timeMultiplier;

        return adjustment;
    }

    private calculatePredictionConfidence(stats: PairGasStats, trendStrength: number): number {
        let confidence = 0.5; // Base confidence

        // Adjust based on historical accuracy
        if (stats.predictions.predictionAccuracy.length > 0) {
            const avgAccuracy = stats.predictions.predictionAccuracy.reduce((a, b) => a + b) 
                / stats.predictions.predictionAccuracy.length;
            confidence *= (0.5 + avgAccuracy / 2);
        }

        // Adjust based on market conditions
        if (stats.marketConditions.volatilityLevel > this.CONGESTION_THRESHOLDS.HIGH) {
            confidence *= 0.7;
        }

        // Adjust based on trend strength
        confidence *= (0.7 + trendStrength * 0.3);

        return Math.min(1, Math.max(0, confidence));
    }

    private getUrgencyMultiplier(stats: PairGasStats, urgency: UrgencyLevel): number {
        const congestionLevel = this.getCongestionLevel(stats.marketConditions.congestionLevel);
        const baseMultiplier = this.CONGESTION_URGENCY_MULTIPLIERS[urgency][congestionLevel];
        
        // Adjust for recent failures if urgency is high
        if ((urgency === UrgencyLevel.HIGH || urgency === UrgencyLevel.CRITICAL) && 
            stats.marketConditions.recentFailurePattern) {
            return baseMultiplier * this.GAS_ADJUSTMENTS.FAILURE_PATTERN;
        }

        // Adjust for high volatility in critical situations
        if (urgency === UrgencyLevel.CRITICAL && 
            stats.marketConditions.volatilityLevel >= this.CONGESTION_THRESHOLDS.HIGH) {
            return baseMultiplier * this.GAS_ADJUSTMENTS.VOLATILITY.HIGH;
        }

        return baseMultiplier;
    }

    private getCongestionLevel(level: number): 'LOW' | 'MEDIUM' | 'HIGH' {
        if (level >= this.CONGESTION_THRESHOLDS.HIGH) return 'HIGH';
        if (level >= this.CONGESTION_THRESHOLDS.MEDIUM) return 'MEDIUM';
        return 'LOW';
    }

    private calculateBasePrediction(pair: string, currentBaseFee: bigint): { 
        predictedPrice: bigint;
        confidence: number;
    } {
        const stats = this.pairGasStats.get(pair) || this.getDefaultStats();
        const now = Date.now();

        // Update EMAs
        if (stats.historicalGasPrices.length > 0) {
            const latestPrice = stats.historicalGasPrices[stats.historicalGasPrices.length - 1].gasPrice;
            stats.predictions.shortTermEMA = this.calculateEMA(
                stats.predictions.shortTermEMA,
                latestPrice,
                this.EMA_ALPHA_SHORT
            );
            stats.predictions.longTermEMA = this.calculateEMA(
                stats.predictions.longTermEMA,
                latestPrice,
                this.EMA_ALPHA_LONG
            );
        }

        const trendStrength = this.calculateTrendStrength(stats);
        const predictedDelta = stats.predictions.shortTermEMA > stats.predictions.longTermEMA
            ? (stats.predictions.shortTermEMA - stats.predictions.longTermEMA) * BigInt(trendStrength)
            : (stats.predictions.longTermEMA - stats.predictions.shortTermEMA) * BigInt(-trendStrength);
        
        const baselinePrediction = stats.predictions.shortTermEMA + 
            (predictedDelta * BigInt(this.PREDICTION_WINDOW)) / 100n;

        const marketAdjustment = this.calculateMarketAdjustment(stats);
        const predictedPrice = baselinePrediction * BigInt(Math.floor(marketAdjustment * 100)) / 100n;
        const confidence = this.calculatePredictionConfidence(stats, trendStrength);

        stats.predictions.lastPrediction = {
            predictedPrice,
            confidence,
            timestamp: now
        };

        this.pairGasStats.set(pair, stats);

        return { predictedPrice, confidence };
    }

    private updateMEVRiskLevel(stats: PairGasStats): void {
        const totalAttempts = stats.mevStats.recentMEVActivity.reduce((sum, activity) => 
            sum + activity.frontrunAttempts + activity.sandwichAttempts + activity.backrunAttempts, 0
        );

        const weightedRisk = stats.mevStats.recentMEVActivity.reduce((risk, activity) => {
            const age = Date.now() - activity.timestamp;
            const recencyWeight = 1 - (age / this.MEV_MONITORING_WINDOW);
            return risk + (
                (activity.frontrunAttempts * 3 + 
                activity.sandwichAttempts * 4 + 
                activity.backrunAttempts * 2) * recencyWeight
            );
        }, 0);

        stats.mevStats.mevRiskLevel = Math.min(100, 
            (weightedRisk / Math.max(1, totalAttempts)) * 25
        );
    }

    private getMEVAwareGasMultiplier(stats: PairGasStats, urgency: UrgencyLevel): number {
        let multiplier = 1.0;

        // Base MEV risk adjustment
        if (stats.mevStats.mevRiskLevel >= this.MEV_THRESHOLDS.HIGH_RISK) {
            multiplier *= this.MEV_GAS_ADJUSTMENTS.HIGH_RISK;
        } else if (stats.mevStats.mevRiskLevel >= this.MEV_THRESHOLDS.MEDIUM_RISK) {
            multiplier *= this.MEV_GAS_ADJUSTMENTS.MEDIUM_RISK;
        } else if (stats.mevStats.mevRiskLevel >= this.MEV_THRESHOLDS.LOW_RISK) {
            multiplier *= this.MEV_GAS_ADJUSTMENTS.LOW_RISK;
        }

        // Specific MEV pattern protection
        const recentActivity = stats.mevStats.recentMEVActivity;
        if (recentActivity.length > 0) {
            const hasFrontrun = recentActivity.some(a => a.frontrunAttempts > 0);
            const hasSandwich = recentActivity.some(a => a.sandwichAttempts > 0);
            const hasBackrun = recentActivity.some(a => a.backrunAttempts > 0);

            if (hasSandwich) multiplier *= this.MEV_GAS_ADJUSTMENTS.SANDWICH_PROTECTION;
            if (hasFrontrun) multiplier *= this.MEV_GAS_ADJUSTMENTS.FRONTRUN_PROTECTION;
            if (hasBackrun) multiplier *= this.MEV_GAS_ADJUSTMENTS.BACKRUN_PROTECTION;
        }

        // Additional protection for high urgency transactions
        if (urgency === UrgencyLevel.HIGH || urgency === UrgencyLevel.CRITICAL) {
            multiplier *= 1.2;
        }

        return multiplier;
    }

    getOptimalGasPrice(
        pair: string, 
        baseGasPrice: bigint, 
        urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    ): bigint {
        const stats = this.pairGasStats.get(pair) || this.getDefaultStats();
        const timeBasedMultiplier = this.getTimeBasedMultiplier(stats);
        const marketConditionMultiplier = this.getMarketConditionMultiplier(stats);
        const urgencyMultiplier = this.getUrgencyMultiplier(stats, urgency);
        const mevAwareMultiplier = this.getMEVAwareGasMultiplier(stats, urgency);
        
        const finalMultiplier = stats.optimalGasMultiplier * 
            timeBasedMultiplier * 
            marketConditionMultiplier * 
            urgencyMultiplier *
            mevAwareMultiplier;
            
        return baseGasPrice * BigInt(Math.floor(finalMultiplier * 100)) / 100n;
    }

    getPairStats(pair: string): PairGasStats {
        return this.pairGasStats.get(pair) || this.getDefaultStats();
    }

    private getDefaultStats(): PairGasStats {
        return {
            averageGasUsed: 0n,
            successRate: 100,
            failureCount: 0,
            optimalGasMultiplier: this.BASE_GAS_MULTIPLIER,
            historicalGasPrices: [],
            lastOptimization: 0,
            marketConditions: {
                congestionLevel: 0,
                volatilityLevel: 0,
                lastBlockBaseFee: 0n,
                recentFailurePattern: false
            },
            timeBasedStats: {
                hourOfDay: {},
                dayOfWeek: {}
            },
            predictions: {
                shortTermEMA: 0n,
                longTermEMA: 0n,
                lastPrediction: {
                    predictedPrice: 0n,
                    confidence: 0,
                    timestamp: 0
                },
                predictionAccuracy: []
            },
            urgencyMultipliers: { ...this.DEFAULT_URGENCY_MULTIPLIERS },
            mevStats: {
                recentMEVActivity: [],
                mevRiskLevel: 0,
                lastMEVCheck: 0,
                protectedTransactions: 0,
                mevLosses: 0n
            }
        };
    }

    async optimizeGas(profit: bigint, urgency: UrgencyLevel = UrgencyLevel.MEDIUM): Promise<bigint> {
        return 200000n;
    }
} 