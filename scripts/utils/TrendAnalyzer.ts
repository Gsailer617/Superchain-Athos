import { TrendAnalysis, PricePoint, VolumeData } from '../types/dex';
import { QUALITY_THRESHOLDS } from '../config/dex.config';
import { NumericUtils } from './NumericUtils';
import { RewardTracker } from './RewardTracker';

interface DataPoint {
    timestamp: number;
    price: bigint;
    volume: bigint;
    liquidity: bigint;
}

interface TrendLine {
    slope: number;
    intercept: number;
    r2: number;  // R-squared value for trend reliability
}

interface VolumeMetrics {
    trend: 'up' | 'down' | 'sideways';
    score: number;
    distribution: 'normal' | 'skewed' | 'clustered';
    isExpanding: boolean;
}

interface VolumeDataPoint {
    volume1h: number;
    volume24h: number;
    volume7d: number;
}

export class TrendAnalyzer {
    private static readonly SUPPORT_RESISTANCE_BUCKETS = 20;
    private static readonly MIN_PATTERN_POINTS = 5;

    /**
     * Analyzes market trends using multiple timeframes and technical indicators
     */
    analyzeTrends(
        data: DataPoint[],
        timeframe: string
    ): TrendAnalysis {
        if (data.length < QUALITY_THRESHOLDS.TREND_ANALYSIS.MIN_DATA_POINTS) {
            return this.getDefaultTrendAnalysis(timeframe);
        }

        // Calculate trend line
        const trendLine = this.calculateTrendLine(data);
        
        // Calculate volatility
        const volatility = this.calculateVolatility(data);
        
        // Identify support and resistance levels
        const { support, resistance } = this.findSupportResistance(data);
        
        // Detect potential breakout
        const breakoutProbability = this.calculateBreakoutProbability(
            data,
            trendLine,
            support,
            resistance
        );

        // Calculate trend magnitude and confidence
        const magnitude = this.calculateTrendMagnitude(trendLine, data);
        const confidence = this.calculateTrendConfidence(trendLine, volatility);

        return {
            timeframe,
            direction: this.getTrendDirection(trendLine.slope),
            magnitude,
            confidence,
            volatility,
            support: Number(support),
            resistance: Number(resistance),
            breakoutProbability
        };
    }

    /**
     * Detects common chart patterns (e.g., double tops, head and shoulders)
     */
    detectPatterns(data: DataPoint[]): {
        pattern: string | null;
        confidence: number;
        potentialTarget?: bigint;
    } {
        if (data.length < TrendAnalyzer.MIN_PATTERN_POINTS) {
            return { pattern: null, confidence: 0 };
        }

        // Check for double top/bottom
        const doublePattern = this.detectDoublePattern(data);
        if (doublePattern.pattern) {
            return doublePattern;
        }

        // Check for head and shoulders
        const headAndShoulders = this.detectHeadAndShoulders(data);
        if (headAndShoulders.pattern) {
            return headAndShoulders;
        }

        // Check for triangle patterns
        const triangle = this.detectTrianglePattern(data);
        if (triangle.pattern) {
            return triangle;
        }

        return { pattern: null, confidence: 0 };
    }

    private calculateTrendLine(data: DataPoint[]): TrendLine {
        const xValues = data.map((_, i) => i);
        const yValues = data.map(d => NumericUtils.fromFixed18(d.price));
        
        const n = data.length;
        const sumX = xValues.reduce((a, b) => a + b, 0);
        const sumY = yValues.reduce((a, b) => a + b, 0);
        const sumXY = xValues.reduce((sum, x, i) => sum + x * yValues[i], 0);
        const sumXX = xValues.reduce((sum, x) => sum + x * x, 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        // Calculate R-squared
        const yMean = sumY / n;
        const totalSS = yValues.reduce((sum, y) => sum + Math.pow(y - yMean, 2), 0);
        const residualSS = yValues.reduce((sum, y, i) => {
            const yPred = slope * xValues[i] + intercept;
            return sum + Math.pow(y - yPred, 2);
        }, 0);
        const r2 = 1 - (residualSS / totalSS);

        return { slope, intercept, r2 };
    }

    private calculateVolatility(data: DataPoint[]): number {
        const prices = data.map(d => NumericUtils.fromFixed18(d.price));
        const returns = prices.slice(1).map((price, i) => 
            Math.log(price / prices[i])
        );
        
        const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance = returns.reduce((sum, ret) => 
            sum + Math.pow(ret - mean, 2), 0
        ) / returns.length;
        
        return Math.sqrt(variance);
    }

    private findSupportResistance(data: DataPoint[]): {
        support: bigint;
        resistance: bigint;
    } {
        const prices = data.map(d => NumericUtils.fromFixed18(d.price));
        const min = Math.min(...prices);
        const max = Math.max(...prices);
        const range = max - min;
        const bucketSize = range / TrendAnalyzer.SUPPORT_RESISTANCE_BUCKETS;
        
        // Create price distribution buckets
        const buckets = new Array(TrendAnalyzer.SUPPORT_RESISTANCE_BUCKETS).fill(0);
        prices.forEach(price => {
            const bucketIndex = Math.min(
                Math.floor((price - min) / bucketSize),
                TrendAnalyzer.SUPPORT_RESISTANCE_BUCKETS - 1
            );
            buckets[bucketIndex]++;
        });
        
        // Find local maxima in price distribution
        const localMaxima = buckets
            .map((count, i) => ({ count, price: min + (i + 0.5) * bucketSize }))
            .filter((bucket, i, arr) => {
                if (i === 0 || i === arr.length - 1) return false;
                return bucket.count > arr[i - 1].count && bucket.count > arr[i + 1].count;
            })
            .sort((a, b) => b.count - a.count);
        
        return {
            support: NumericUtils.toFixed18(localMaxima[0]?.price || min),
            resistance: NumericUtils.toFixed18(localMaxima[1]?.price || max)
        };
    }

    private calculateBreakoutProbability(
        data: DataPoint[],
        trendLine: TrendLine,
        support: bigint,
        resistance: bigint
    ): number {
        const lastPrice = NumericUtils.fromFixed18(data[data.length - 1].price);
        const supportPrice = NumericUtils.fromFixed18(support);
        const resistancePrice = NumericUtils.fromFixed18(resistance);
        
        // Calculate distance to support/resistance
        const distanceToSupport = Math.abs(lastPrice - supportPrice) / supportPrice;
        const distanceToResistance = Math.abs(lastPrice - resistancePrice) / resistancePrice;
        
        // Consider trend strength
        const trendStrength = Math.abs(trendLine.slope) * trendLine.r2;
        
        // Calculate volume trend
        const volumeMetrics = this.calculateVolumeTrend(data);
        
        // Combine factors
        const breakoutScore = (
            (1 - Math.min(distanceToSupport, distanceToResistance)) * 0.4 +
            trendStrength * 0.3 +
            volumeMetrics.score * 0.3
        );
        
        return Math.min(Math.max(breakoutScore, 0), 1);
    }

    private convertVolumesToNumbers(data: DataPoint[]): number[] {
        return data.map(d => Number(d.volume));
    }

    private calculateVolumeTrend(data: DataPoint[]): VolumeMetrics {
        const trend = this.calculateTrendLineForVolumes(data);
        const score = Math.min(1, Math.abs(trend.slope) * 10 + trend.r2);
        const distribution = this.analyzeVolumeDistribution(data);
        const expansion = this.checkVolumeExpansion(data);

        return {
            trend: trend.slope > 0.001 ? 'up' : trend.slope < -0.001 ? 'down' : 'sideways',
            score,
            distribution: distribution.distribution,
            isExpanding: expansion.isExpanding
        };
    }

    private calculateTrendLineForVolumes(data: DataPoint[]): TrendLine {
        const points = data.map((d, i) => ({
            x: i,
            y: Number(d.volume)
        }));

        if (points.length < 2) {
            return { slope: 0, intercept: 0, r2: 0 };
        }

        const n = points.length;
        const sumX = points.reduce((sum, p) => sum + p.x, 0);
        const sumY = points.reduce((sum, p) => sum + p.y, 0);
        const sumXY = points.reduce((sum, p) => sum + p.x * p.y, 0);
        const sumXX = points.reduce((sum, p) => sum + p.x * p.x, 0);

        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        // Calculate R-squared
        const yMean = sumY / n;
        const totalSS = points.reduce((sum, p) => sum + Math.pow(p.y - yMean, 2), 0);
        const residualSS = points.reduce((sum, p) => {
            const yPred = slope * p.x + intercept;
            return sum + Math.pow(p.y - yPred, 2);
        }, 0);
        const r2 = 1 - (residualSS / totalSS);

        return { slope, intercept, r2 };
    }

    private checkVolumeExpansion(data: DataPoint[]): {
        isExpanding: boolean;
        score: number;
    } {
        const recentData = data.slice(-5);
        const volumes = recentData.map(d => Number(d.volume));
        const avgVolume = volumes.reduce((sum, v) => sum + v, 0) / volumes.length;
        const stdDev = Math.sqrt(
            volumes.reduce((sum, v) => sum + Math.pow(v - avgVolume, 2), 0) / volumes.length
        );

        const expansionRatio = stdDev / avgVolume;
        const isExpanding = expansionRatio > 0.2; // 20% variation threshold
        
        return {
            isExpanding,
            score: Math.min(1, expansionRatio * 2)
        };
    }

    private analyzeVolumeDistribution(data: DataPoint[]): {
        distribution: 'normal' | 'skewed' | 'clustered';
        score: number;
    } {
        const volumes = data.map(d => Number(d.volume));
        const avgVolume = volumes.reduce((sum, v) => sum + v, 0) / volumes.length;
        
        // Calculate skewness
        const variance = volumes.reduce((sum, v) => sum + Math.pow(v - avgVolume, 2), 0) / volumes.length;
        const stdDev = Math.sqrt(variance);
        const skewness = volumes.reduce((sum, v) => sum + Math.pow((v - avgVolume) / stdDev, 3), 0) / volumes.length;

        // Analyze clustering
        const clusters = this.identifyVolumeClusters(data);
        
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

        return { distribution, score };
    }

    private identifyVolumeClusters(data: DataPoint[]): Array<{
        start: number;
        end: number;
        avgVolume: number;
    }> {
        const volumes = data.map(d => Number(d.volume));
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

    private calculateVolumeConfirmation(startIndex: number, endIndex: number): number {
        const volumeData = this.getVolumeData(startIndex, endIndex);
        if (!volumeData || !volumeData.length) return 0;

        // Calculate volume trend
        const volumeMetrics = this.calculateVolumeTrend(volumeData);
        
        // Combine metrics for confirmation score
        return (
            volumeMetrics.score * 0.4 +
            (volumeMetrics.isExpanding ? 0.3 : 0) +
            (volumeMetrics.distribution === 'normal' ? 0.3 : 0.1)
        );
    }

    private calculateTrendMagnitude(trendLine: TrendLine, data: DataPoint[]): number {
        const priceChange = Math.abs(trendLine.slope * data.length);
        const avgPrice = data.reduce((sum, d) => 
            sum + NumericUtils.fromFixed18(d.price), 0
        ) / data.length;
        
        return Math.min(priceChange / avgPrice, 1);
    }

    private calculateTrendConfidence(trendLine: TrendLine, volatility: number): number {
        // Combine R-squared and inverse volatility
        const volatilityScore = Math.max(0, 1 - volatility * 5);
        return (trendLine.r2 * 0.7 + volatilityScore * 0.3);
    }

    private getTrendDirection(slope: number): 'up' | 'down' | 'sideways' {
        if (Math.abs(slope) < QUALITY_THRESHOLDS.TREND_ANALYSIS.TREND_THRESHOLD) {
            return 'sideways';
        }
        return slope > 0 ? 'up' : 'down';
    }

    private getDefaultTrendAnalysis(timeframe: string): TrendAnalysis {
        return {
            timeframe,
            direction: 'sideways',
            magnitude: 0,
            confidence: 0,
            volatility: 0,
            support: 0,
            resistance: 0,
            breakoutProbability: 0
        };
    }

    private detectDoublePattern(data: DataPoint[]): {
        pattern: string | null;
        confidence: number;
        potentialTarget?: bigint;
    } {
        if (data.length < 20) return { pattern: null, confidence: 0 };

        const prices = data.map(d => NumericUtils.fromFixed18(d.price));
        const peaks = this.findPeaksAndValleys(prices);
        
        // Need at least 3 significant peaks/valleys for a double pattern
        if (peaks.length < 3) return { pattern: null, confidence: 0 };

        // Look for similar price levels
        for (let i = 0; i < peaks.length - 2; i++) {
            const first = peaks[i];
            const middle = peaks[i + 1];
            const second = peaks[i + 2];

            // Check if first and second peaks/valleys are at similar levels
            const priceDeviation = Math.abs(prices[first.index] - prices[second.index]) / prices[first.index];
            if (priceDeviation > 0.02) continue; // More than 2% difference

            // For double top, middle should be a valley lower than tops
            if (first.type === 'peak' && second.type === 'peak') {
                if (prices[middle.index] < prices[first.index] * 0.97) {
                    const confidence = this.calculateDoublePatternConfidence(
                        prices, first, middle, second
                    );
                    if (confidence > 0.7) {
                        return {
                            pattern: 'double_top',
                            confidence,
                            potentialTarget: NumericUtils.toFixed18(prices[middle.index])
                        };
                    }
                }
            }

            // For double bottom, middle should be a peak higher than bottoms
            if (first.type === 'valley' && second.type === 'valley') {
                if (prices[middle.index] > prices[first.index] * 1.03) {
                    const confidence = this.calculateDoublePatternConfidence(
                        prices, first, middle, second
                    );
                    if (confidence > 0.7) {
                        return {
                            pattern: 'double_bottom',
                            confidence,
                            potentialTarget: NumericUtils.toFixed18(prices[middle.index])
                        };
                    }
                }
            }
        }

        return { pattern: null, confidence: 0 };
    }

    private detectHeadAndShoulders(data: DataPoint[]): {
        pattern: string | null;
        confidence: number;
        potentialTarget?: bigint;
    } {
        if (data.length < 30) return { pattern: null, confidence: 0 };

        const prices = data.map(d => NumericUtils.fromFixed18(d.price));
        const peaks = this.findPeaksAndValleys(prices);

        // Need at least 5 alternating peaks and valleys
        if (peaks.length < 5) return { pattern: null, confidence: 0 };

        // Look for head and shoulders pattern
        for (let i = 0; i < peaks.length - 4; i++) {
            const leftShoulder = peaks[i];
            const leftValley = peaks[i + 1];
            const head = peaks[i + 2];
            const rightValley = peaks[i + 3];
            const rightShoulder = peaks[i + 4];

            // Verify pattern structure
            if (!this.isValidHeadAndShoulders(
                prices,
                leftShoulder,
                leftValley,
                head,
                rightValley,
                rightShoulder
            )) continue;

            const confidence = this.calculateHSPatternConfidence(
                prices,
                leftShoulder,
                leftValley,
                head,
                rightValley,
                rightShoulder
            );

            if (confidence > 0.7) {
                // Calculate neckline and target
                const neckline = (prices[leftValley.index] + prices[rightValley.index]) / 2;
                const height = prices[head.index] - neckline;
                const target = neckline - height; // Measured move target

                return {
                    pattern: 'head_and_shoulders',
                    confidence,
                    potentialTarget: NumericUtils.toFixed18(target)
                };
            }
        }

        return { pattern: null, confidence: 0 };
    }

    private detectTrianglePattern(data: DataPoint[]): {
        pattern: string | null;
        confidence: number;
        potentialTarget?: bigint;
    } {
        if (data.length < 20) return { pattern: null, confidence: 0 };

        const prices = data.map(d => NumericUtils.fromFixed18(d.price));
        const peaks = this.findPeaksAndValleys(prices);

        // Need at least 4 points to form a triangle
        if (peaks.length < 4) return { pattern: null, confidence: 0 };

        // Calculate trend lines for highs and lows
        const highTrend = this.calculateTrendLineForNumbers(prices, peaks, 'peak');
        const lowTrend = this.calculateTrendLineForNumbers(prices, peaks, 'valley');

        // Check for convergence
        const convergencePoint = this.findConvergencePoint(highTrend, lowTrend);
        if (!convergencePoint) return { pattern: null, confidence: 0 };

        // Determine triangle type and confidence
        const { pattern, confidence } = this.identifyTriangleType(
            highTrend,
            lowTrend,
            prices
        );

        if (pattern && confidence > 0.7) {
            // Calculate potential target based on triangle height
            const height = Math.abs(
                highTrend.slope * data.length + highTrend.intercept -
                (lowTrend.slope * data.length + lowTrend.intercept)
            );
            const breakoutPrice = convergencePoint.price;
            const target = pattern === 'ascending_triangle' ? 
                breakoutPrice + height :
                pattern === 'descending_triangle' ?
                    breakoutPrice - height :
                    breakoutPrice; // Symmetric triangle

            return {
                pattern,
                confidence,
                potentialTarget: NumericUtils.toFixed18(target)
            };
        }

        return { pattern: null, confidence: 0 };
    }

    private findPeaksAndValleys(prices: number[]): Array<{
        index: number;
        type: 'peak' | 'valley';
    }> {
        const result: Array<{ index: number; type: 'peak' | 'valley' }> = [];
        const minDistance = 3; // Minimum distance between peaks/valleys

        for (let i = minDistance; i < prices.length - minDistance; i++) {
            const window = prices.slice(i - minDistance, i + minDistance + 1);
            const current = prices[i];

            // Check for peak
            if (current === Math.max(...window)) {
                result.push({ index: i, type: 'peak' });
                i += minDistance; // Skip ahead to avoid detecting minor fluctuations
                continue;
            }

            // Check for valley
            if (current === Math.min(...window)) {
                result.push({ index: i, type: 'valley' });
                i += minDistance; // Skip ahead to avoid detecting minor fluctuations
            }
        }

        return result;
    }

    private calculateDoublePatternConfidence(
        prices: number[],
        first: { index: number; type: 'peak' | 'valley' },
        middle: { index: number; type: 'peak' | 'valley' },
        second: { index: number; type: 'peak' | 'valley' }
    ): number {
        // Check price similarity
        const priceDeviation = Math.abs(prices[first.index] - prices[second.index]) / prices[first.index];
        const priceScore = Math.max(0, 1 - priceDeviation * 50); // 2% deviation = 0 score

        // Check time symmetry
        const firstToMiddle = middle.index - first.index;
        const middleToSecond = second.index - middle.index;
        const timeDeviation = Math.abs(firstToMiddle - middleToSecond) / firstToMiddle;
        const timeScore = Math.max(0, 1 - timeDeviation * 2);

        // Check volume confirmation
        const volumeScore = this.calculateVolumeConfirmation(
            first.index,
            second.index
        );

        return (priceScore * 0.4 + timeScore * 0.3 + volumeScore * 0.3);
    }

    private isValidHeadAndShoulders(
        prices: number[],
        leftShoulder: { index: number; type: 'peak' | 'valley' },
        leftValley: { index: number; type: 'peak' | 'valley' },
        head: { index: number; type: 'peak' | 'valley' },
        rightValley: { index: number; type: 'peak' | 'valley' },
        rightShoulder: { index: number; type: 'peak' | 'valley' }
    ): boolean {
        // Verify correct sequence of peaks and valleys
        if (leftShoulder.type !== 'peak' || head.type !== 'peak' || rightShoulder.type !== 'peak') return false;
        if (leftValley.type !== 'valley' || rightValley.type !== 'valley') return false;

        // Head should be higher than shoulders
        if (prices[head.index] <= prices[leftShoulder.index] || 
            prices[head.index] <= prices[rightShoulder.index]) return false;

        // Shoulders should be at similar levels (within 5%)
        const shoulderDeviation = Math.abs(
            prices[leftShoulder.index] - prices[rightShoulder.index]
        ) / prices[leftShoulder.index];
        if (shoulderDeviation > 0.05) return false;

        // Valleys should be at similar levels (within 5%)
        const valleyDeviation = Math.abs(
            prices[leftValley.index] - prices[rightValley.index]
        ) / prices[leftValley.index];
        if (valleyDeviation > 0.05) return false;

        return true;
    }

    private calculateHSPatternConfidence(
        prices: number[],
        leftShoulder: { index: number; type: 'peak' | 'valley' },
        leftValley: { index: number; type: 'peak' | 'valley' },
        head: { index: number; type: 'peak' | 'valley' },
        rightValley: { index: number; type: 'peak' | 'valley' },
        rightShoulder: { index: number; type: 'peak' | 'valley' }
    ): number {
        // Check shoulder symmetry
        const shoulderDeviation = Math.abs(
            prices[leftShoulder.index] - prices[rightShoulder.index]
        ) / prices[leftShoulder.index];
        const shoulderScore = Math.max(0, 1 - shoulderDeviation * 20);

        // Check valley symmetry
        const valleyDeviation = Math.abs(
            prices[leftValley.index] - prices[rightValley.index]
        ) / prices[leftValley.index];
        const valleyScore = Math.max(0, 1 - valleyDeviation * 20);

        // Check time symmetry
        const leftWidth = head.index - leftShoulder.index;
        const rightWidth = rightShoulder.index - head.index;
        const timeDeviation = Math.abs(leftWidth - rightWidth) / leftWidth;
        const timeScore = Math.max(0, 1 - timeDeviation * 2);

        // Check volume pattern
        const volumeScore = this.calculateVolumeConfirmation(
            leftShoulder.index,
            rightShoulder.index
        );

        return (
            shoulderScore * 0.3 +
            valleyScore * 0.3 +
            timeScore * 0.2 +
            volumeScore * 0.2
        );
    }

    private calculateTrendLineForNumbers(
        data: number[],
        peaks: Array<{ index: number; type: 'peak' | 'valley' }>,
        peakType: 'peak' | 'valley'
    ): TrendLine {
        const points = peaks
            .filter(p => p.type === peakType)
            .map(p => ({
                x: p.index,
                y: data[p.index]
            }));

        if (points.length < 2) {
            return { slope: 0, intercept: 0, r2: 0 };
        }

        const n = points.length;
        const sumX = points.reduce((sum, p) => sum + p.x, 0);
        const sumY = points.reduce((sum, p) => sum + p.y, 0);
        const sumXY = points.reduce((sum, p) => sum + p.x * p.y, 0);
        const sumXX = points.reduce((sum, p) => sum + p.x * p.x, 0);

        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        // Calculate R-squared
        const yMean = sumY / n;
        const totalSS = points.reduce((sum, p) => sum + Math.pow(p.y - yMean, 2), 0);
        const residualSS = points.reduce((sum, p) => {
            const yPred = slope * p.x + intercept;
            return sum + Math.pow(p.y - yPred, 2);
        }, 0);
        const r2 = 1 - (residualSS / totalSS);

        return { slope, intercept, r2 };
    }

    private calculateTrendLineForDataPoints(
        data: DataPoint[],
        peaks: Array<{ index: number; type: 'peak' | 'valley' }>,
        peakType: 'peak' | 'valley'
    ): TrendLine {
        const volumes = data.map(d => Number(d.volume));
        return this.calculateTrendLineForNumbers(volumes, peaks, peakType);
    }

    private findConvergencePoint(
        highTrend: TrendLine,
        lowTrend: TrendLine
    ): { x: number; price: number } | null {
        if (Math.abs(highTrend.slope - lowTrend.slope) < 0.0001) {
            return null; // Lines are parallel
        }

        const x = (lowTrend.intercept - highTrend.intercept) / 
                 (highTrend.slope - lowTrend.slope);
        
        if (x <= 0) return null; // Convergence in the past

        const price = highTrend.slope * x + highTrend.intercept;
        return { x, price };
    }

    private identifyTriangleType(
        highTrend: TrendLine,
        lowTrend: TrendLine,
        prices: number[]
    ): {
        pattern: string | null;
        confidence: number;
    } {
        const highSlope = highTrend.slope;
        const lowSlope = lowTrend.slope;
        const r2Threshold = 0.7;

        if (highTrend.r2 < r2Threshold || lowTrend.r2 < r2Threshold) {
            return { pattern: null, confidence: 0 };
        }

        // Ascending triangle
        if (Math.abs(highSlope) < 0.001 && lowSlope > 0.001) {
            return {
                pattern: 'ascending_triangle',
                confidence: (highTrend.r2 + lowTrend.r2) / 2
            };
        }

        // Descending triangle
        if (highSlope < -0.001 && Math.abs(lowSlope) < 0.001) {
            return {
                pattern: 'descending_triangle',
                confidence: (highTrend.r2 + lowTrend.r2) / 2
            };
        }

        // Symmetric triangle
        if (highSlope < -0.001 && lowSlope > 0.001) {
            const symmetryScore = Math.min(
                Math.abs(highSlope) / Math.abs(lowSlope),
                Math.abs(lowSlope) / Math.abs(highSlope)
            );
            
            if (symmetryScore > 0.7) {
                return {
                    pattern: 'symmetric_triangle',
                    confidence: symmetryScore * (highTrend.r2 + lowTrend.r2) / 2
                };
            }
        }

        return { pattern: null, confidence: 0 };
    }

    private getVolumeData(startIndex: number, endIndex: number): DataPoint[] {
        // Implementation for getting volume data
        // This should return actual volume data from your data source
        return [];
    }

    public calculatePairPriority(data: DataPoint[]): {
        priority: number;
        reasons: string[];
    } {
        const volumeMetrics = this.calculateVolumeTrend(data);
        const reasons: string[] = [];
        let priority = 0;

        // Volume trend impact
        if (volumeMetrics.trend === 'up') {
            priority += 0.4;
            reasons.push('Increasing volume trend indicates growing market interest');
        } else if (volumeMetrics.trend === 'sideways') {
            priority += 0.3;
            reasons.push('Stable volume provides consistent trading opportunities');
        }

        // Volume expansion impact
        if (volumeMetrics.isExpanding) {
            priority += volumeMetrics.score * 0.3;
            reasons.push('Volume expansion suggests increased market activity');
        }

        // Distribution impact
        if (volumeMetrics.distribution === 'normal') {
            priority += 0.3;
            reasons.push('Normal volume distribution indicates healthy trading patterns');
        } else if (volumeMetrics.distribution === 'skewed') {
            priority += 0.1;
            reasons.push('Skewed volume distribution - exercise caution');
        }

        return {
            priority: Math.min(1, priority),
            reasons
        };
    }
}

const tracker = new RewardTracker();
tracker.configureNotifications({
    telegramBotToken: process.env.TELEGRAM_BOT_TOKEN || '',
    chatId: process.env.TELEGRAM_CHAT_ID || '',
    minTradeValueUSD: 50,     // Notify for trades over $50
    minStakingAPY: 15,        // Notify for APY over 15%
    minGovernanceValue: 100   // Notify for voting power over $100
}); 