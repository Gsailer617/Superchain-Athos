import { VolumeData, VolumeMetrics } from '../types/dex';
import { NumericUtils } from './NumericUtils';

export class VolumeAnalyzer {
    private readonly LARGE_TRADE_THRESHOLD = 100000n; // $100k
    private readonly MEDIUM_TRADE_THRESHOLD = 10000n; // $10k

    analyzeVolume(data: VolumeData): VolumeMetrics {
        const volumeScore = this.calculateVolumeScore(data);
        const tradeFrequency = this.calculateTradeFrequency(data);
        const tradeSizeDistribution = this.calculateTradeSizeDistribution(data);
        const volatility = this.calculateVolatility(data);
        const trend = this.determineTrend(data);
        const confidence = this.calculateConfidence(data);

        return {
            volumeScore,
            tradeFrequency,
            tradeSizeDistribution,
            volatility,
            trend,
            confidence
        };
    }

    private calculateVolumeScore(data: VolumeData): number {
        // Calculate a score based on volume metrics
        const hourlyWeight = 0.4;
        const dailyWeight = 0.4;
        const weeklyWeight = 0.2;

        const hourlyVol = Number(data.volume1h);
        const dailyVol = Number(data.volume24h) / 24;
        const weeklyVol = Number(data.volume7d) / (24 * 7);

        const hourlyScore = this.normalizeVolume(hourlyVol);
        const dailyScore = this.normalizeVolume(dailyVol);
        const weeklyScore = this.normalizeVolume(weeklyVol);

        return (
            hourlyScore * hourlyWeight +
            dailyScore * dailyWeight +
            weeklyScore * weeklyWeight
        );
    }

    private calculateTradeFrequency(data: VolumeData): number {
        // Calculate trade frequency score
        const hourlyFreq = data.tradeCount1h;
        const dailyAvgFreq = data.tradeCount24h / 24;
        const weeklyAvgFreq = data.tradeCount7d / (24 * 7);

        const weightedFreq = (
            hourlyFreq * 0.4 +
            dailyAvgFreq * 0.4 +
            weeklyAvgFreq * 0.2
        );

        return Math.min(weightedFreq / 100, 1); // Normalize to 0-1
    }

    private calculateTradeSizeDistribution(data: VolumeData): {
        small: number;
        medium: number;
        large: number;
    } {
        const avgSizeNum = Number(data.averageTradeSize1h);
        const mediumThreshold = Number(this.MEDIUM_TRADE_THRESHOLD);
        const largeThreshold = Number(this.LARGE_TRADE_THRESHOLD);
        
        let small = 0;
        let medium = 0;
        let large = 0;

        if (avgSizeNum < mediumThreshold) {
            small = 1;
        } else if (avgSizeNum < largeThreshold) {
            medium = 1;
        } else {
            large = 1;
        }

        return { small, medium, large };
    }

    private calculateVolatility(data: VolumeData): number {
        // Calculate volume volatility using Number conversions for calculations
        const hourlyVol = Number(data.volume1h);
        const dailyAvgVol = Number(data.volume24h) / 24;
        const weeklyAvgVol = Number(data.volume7d) / (24 * 7);

        const volatility = Math.abs(hourlyVol - dailyAvgVol) / dailyAvgVol;
        return Math.min(volatility, 1);
    }

    private determineTrend(data: VolumeData): 'increasing' | 'decreasing' | 'stable' {
        const hourlyVol = Number(data.volume1h);
        const dailyAvgVol = Number(data.volume24h) / 24;
        const weeklyAvgVol = Number(data.volume7d) / (24 * 7);

        const recentTrend = hourlyVol / dailyAvgVol;
        const longTermTrend = dailyAvgVol / weeklyAvgVol;

        if (recentTrend > 1.1 && longTermTrend > 1.05) {
            return 'increasing';
        } else if (recentTrend < 0.9 && longTermTrend < 0.95) {
            return 'decreasing';
        } else {
            return 'stable';
        }
    }

    private calculateConfidence(data: VolumeData): number {
        // Calculate confidence based on data freshness and consistency
        const now = Date.now();
        const dataAge = now - Number(data.lastUpdate);
        const maxAge = 3600000; // 1 hour

        const ageFactor = Math.max(0, 1 - dataAge / maxAge);
        const consistencyFactor = this.calculateConsistencyFactor(data);

        return ageFactor * consistencyFactor;
    }

    private calculateConsistencyFactor(data: VolumeData): number {
        const hourlyVol = Number(data.volume1h);
        const dailyAvgVol = Number(data.volume24h) / 24;
        const weeklyAvgVol = Number(data.volume7d) / (24 * 7);

        const shortTermDiff = Math.abs(hourlyVol - dailyAvgVol) / dailyAvgVol;
        const longTermDiff = Math.abs(dailyAvgVol - weeklyAvgVol) / weeklyAvgVol;

        return Math.max(0, 1 - (shortTermDiff + longTermDiff) / 2);
    }

    private normalizeVolume(volume: number): number {
        // Normalize volume to a 0-1 scale
        const maxVolume = 1000000; // $1M as reference
        return Math.min(volume / maxVolume, 1);
    }

    private calculateVolumeTrend(volumeData: VolumeData[]): {
        trend: 'increasing' | 'decreasing' | 'stable';
        score: number;
    } {
        const volumes = volumeData.map(d => d.volume1h);
        
        // Calculate linear regression
        const n = volumes.length;
        const xValues = Array.from({length: n}, (_, i) => i);
        const sumX = xValues.reduce((sum, x) => sum + x, 0);
        const sumY = volumes.reduce((sum, y) => sum + y, 0);
        const sumXY = xValues.reduce((sum, x, i) => sum + x * volumes[i], 0);
        const sumXX = xValues.reduce((sum, x) => sum + x * x, 0);

        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const yMean = sumY / n;
        
        // Calculate R-squared
        const totalSS = volumes.reduce((sum, y) => sum + Math.pow(y - yMean, 2), 0);
        const residualSS = volumes.reduce((sum, y, i) => {
            const yPred = slope * i + (sumY - slope * sumX) / n;
            return sum + Math.pow(y - yPred, 2);
        }, 0);
        const r2 = 1 - (residualSS / totalSS);

        const score = Math.min(1, Math.abs(slope) * 10 + r2);
        return {
            trend: slope > 0 ? 'increasing' : slope < 0 ? 'decreasing' : 'stable',
            score
        };
    }

    private checkVolumeExpansion(volumeData: VolumeData[]): {
        isExpanding: boolean;
        score: number;
    } {
        const recentVolumes = volumeData.slice(-5);
        const avgVolume = recentVolumes.reduce((sum, d) => sum + d.volume1h, 0) / recentVolumes.length;
        const stdDev = Math.sqrt(
            recentVolumes.reduce((sum, d) => sum + Math.pow(d.volume1h - avgVolume, 2), 0) / recentVolumes.length
        );

        const expansionRatio = stdDev / avgVolume;
        const isExpanding = expansionRatio > 0.2; // 20% variation threshold
        
        return {
            isExpanding,
            score: Math.min(1, expansionRatio * 2)
        };
    }

    private analyzeVolumeDistribution(volumeData: VolumeData[]): {
        distribution: 'normal' | 'skewed' | 'clustered';
        score: number;
        metrics: {
            skewness: number;
            kurtosis: number;
            clusters: Array<{
                start: number;
                end: number;
                avgVolume: number;
            }>;
        };
    } {
        const volumes = volumeData.map(d => d.volume1h);
        const avgVolume = volumes.reduce((sum, v) => sum + v, 0) / volumes.length;
        
        // Calculate skewness and kurtosis
        const variance = volumes.reduce((sum, v) => sum + Math.pow(v - avgVolume, 2), 0) / volumes.length;
        const stdDev = Math.sqrt(variance);
        const skewness = volumes.reduce((sum, v) => sum + Math.pow((v - avgVolume) / stdDev, 3), 0) / volumes.length;
        const kurtosis = volumes.reduce((sum, v) => sum + Math.pow((v - avgVolume) / stdDev, 4), 0) / volumes.length;

        // Analyze clustering
        const clusters = this.identifyVolumeClusters(volumes);
        
        let distribution: 'normal' | 'skewed' | 'clustered';
        let score: number;

        if (Math.abs(skewness) > 1) {
            distribution = 'skewed';
            score = 0.6; // Skewed distribution is less ideal
        } else if (clusters.length > 2) {
            distribution = 'clustered';
            score = 0.7; // Clustered volume can indicate manipulation
        } else {
            distribution = 'normal';
            score = 0.9; // Normal distribution is ideal
        }

        return { 
            distribution, 
            score,
            metrics: {
                skewness,
                kurtosis,
                clusters
            }
        };
    }

    private identifyVolumeClusters(volumes: number[]): Array<{
        start: number;
        end: number;
        avgVolume: number;
    }> {
        const clusters: Array<{
            start: number;
            end: number;
            avgVolume: number;
        }> = [];
        
        let currentCluster: number[] = [volumes[0]];
        let clusterStart = 0;

        for (let i = 1; i < volumes.length; i++) {
            const currentAvg = currentCluster.reduce((sum, v) => sum + v, 0) / currentCluster.length;
            const deviation = Math.abs(volumes[i] - currentAvg) / currentAvg;

            if (deviation <= 0.2) { // 20% threshold for cluster membership
                currentCluster.push(volumes[i]);
            } else {
                if (currentCluster.length >= 3) { // Minimum cluster size
                    clusters.push({
                        start: clusterStart,
                        end: i - 1,
                        avgVolume: currentAvg
                    });
                }
                currentCluster = [volumes[i]];
                clusterStart = i;
            }
        }

        // Handle last cluster
        if (currentCluster.length >= 3) {
            const currentAvg = currentCluster.reduce((sum, v) => sum + v, 0) / currentCluster.length;
            clusters.push({
                start: clusterStart,
                end: volumes.length - 1,
                avgVolume: currentAvg
            });
        }

        return clusters;
    }

    public calculatePairPriority(volumeData: VolumeData[]): {
        priority: number;
        reasons: string[];
        metrics: {
            volumeTrend: {
                trend: 'increasing' | 'decreasing' | 'stable';
                score: number;
            };
            volumeExpansion: {
                isExpanding: boolean;
                score: number;
            };
            distribution: {
                type: 'normal' | 'skewed' | 'clustered';
                score: number;
                skewness: number;
                kurtosis: number;
            };
        };
    } {
        const volumeTrend = this.calculateVolumeTrend(volumeData);
        const volumeExpansion = this.checkVolumeExpansion(volumeData);
        const distribution = this.analyzeVolumeDistribution(volumeData);

        const reasons: string[] = [];
        let priority = 0;

        // Volume trend impact
        if (volumeTrend.trend === 'increasing') {
            priority += 0.4;
            reasons.push('Increasing volume trend indicates growing market interest');
        } else if (volumeTrend.trend === 'stable') {
            priority += 0.3;
            reasons.push('Stable volume provides consistent trading opportunities');
        }

        // Volume expansion impact
        if (volumeExpansion.isExpanding) {
            priority += volumeExpansion.score * 0.3;
            reasons.push('Volume expansion suggests increased market activity');
        }

        // Distribution impact
        if (distribution.distribution === 'normal') {
            priority += 0.3;
            reasons.push('Normal volume distribution indicates healthy trading patterns');
        } else if (distribution.distribution === 'skewed') {
            priority += 0.1;
            reasons.push('Skewed volume distribution - exercise caution');
        }

        // Check for potential manipulation
        if (distribution.metrics.clusters.length > 2) {
            priority *= 0.8; // Reduce priority if multiple volume clusters detected
            reasons.push('Multiple volume clusters detected - possible manipulation');
        }

        return {
            priority: Math.min(1, priority),
            reasons,
            metrics: {
                volumeTrend,
                volumeExpansion,
                distribution: {
                    type: distribution.distribution,
                    score: distribution.score,
                    skewness: distribution.metrics.skewness,
                    kurtosis: distribution.metrics.kurtosis
                }
            }
        };
    }

    async analyzeVolumeByAddress(address: `0x${string}`): Promise<number> {
        return 1000000;
    }
} 