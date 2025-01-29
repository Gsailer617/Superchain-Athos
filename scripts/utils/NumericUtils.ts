import { BigNumberish } from 'ethers';

export class NumericUtils {
    /**
     * Safely converts any BigNumberish value to bigint
     */
    static toBigInt(value: BigNumberish): bigint {
        if (typeof value === 'bigint') return value;
        if (typeof value === 'number') return BigInt(Math.floor(value));
        return BigInt(value.toString());
    }

    /**
     * Safely converts a bigint to number, handling potential precision loss
     */
    static toNumber(value: bigint): number {
        // Check if the value is too large for safe number conversion
        if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
            console.warn('Converting large bigint to number, potential precision loss');
        }
        return Number(value);
    }

    /**
     * Adds two values, converting them to bigint first
     */
    static add(a: BigNumberish, b: BigNumberish): bigint {
        return this.toBigInt(a) + this.toBigInt(b);
    }

    /**
     * Multiplies two values, converting them to bigint first
     */
    static multiply(a: BigNumberish, b: BigNumberish): bigint {
        return this.toBigInt(a) * this.toBigInt(b);
    }

    /**
     * Divides two values, converting them to bigint first
     */
    static divide(a: BigNumberish, b: BigNumberish): bigint {
        const denominator = this.toBigInt(b);
        if (denominator === 0n) throw new Error('Division by zero');
        return this.toBigInt(a) / denominator;
    }

    /**
     * Calculates average of bigint values
     */
    static average(values: bigint[]): bigint {
        if (values.length === 0) return 0n;
        const sum = values.reduce((acc, val) => acc + val, 0n);
        return sum / BigInt(values.length);
    }

    /**
     * Converts a decimal number to a fixed point bigint with 18 decimals
     */
    static toFixed18(value: number): bigint {
        return BigInt(Math.floor(value * 1e18));
    }

    /**
     * Converts a fixed point bigint with 18 decimals to a decimal number
     */
    static fromFixed18(value: bigint): number {
        return Number(value) / 1e18;
    }

    /**
     * Calculates percentage as a bigint (multiplied by 10000 for 4 decimal precision)
     */
    static calculatePercentage(value: bigint, total: bigint): bigint {
        if (total === 0n) return 0n;
        return (value * 10000n) / total;
    }

    /**
     * Calculates variance of fixed-point bigint values (18 decimals)
     */
    static calculateVariance(values: bigint[]): bigint {
        if (values.length < 2) return 0n;
        const mean = this.average(values);
        const sumSquaredDiff = values.reduce((acc, val) => {
            const diff = val - mean;
            return acc + ((diff * diff) / BigInt(1e18)); // Normalize to maintain precision
        }, 0n);
        return sumSquaredDiff / BigInt(values.length);
    }

    /**
     * Calculates volatility score (0-1 in fixed point 18 decimals)
     * Returns 0n for perfectly volatile and 1e18 for perfectly stable
     */
    static calculateVolatility(values: bigint[]): bigint {
        if (values.length < 2) return BigInt(1e18); // Not enough data points
        const mean = this.average(values);
        if (mean === 0n) return 0n; // Avoid division by zero

        const variance = this.calculateVariance(values);
        const volatility = this.sqrt(variance);
        
        // Convert to stability score (1 - volatility/mean), maintaining 18 decimals precision
        const volatilityRatio = (volatility * BigInt(1e18)) / mean;
        return volatilityRatio > BigInt(1e18) ? 0n : BigInt(1e18) - volatilityRatio;
    }

    /**
     * Integer square root for bigint, maintaining precision for fixed-point numbers
     */
    static sqrt(value: bigint): bigint {
        if (value < 0n) {
            throw new Error('Square root of negative number');
        }
        if (value < 2n) return value;

        // For fixed-point numbers (18 decimals), we need to adjust the precision
        const isFixedPoint = value > BigInt(1e18);
        if (isFixedPoint) {
            // Adjust for fixed-point precision before taking sqrt
            value = value * BigInt(1e18);
        }

        let x = value / 2n;
        let y = (x + value / x) / 2n;

        while (y < x) {
            x = y;
            y = (x + value / x) / 2n;
        }

        return isFixedPoint ? x / BigInt(1e9) : x; // Adjust precision back for fixed-point
    }

    /**
     * Calculates exponential moving average
     * @param current Current value
     * @param previous Previous EMA
     * @param smoothing Smoothing factor (0-1) * 1e18
     */
    static calculateEMA(current: bigint, previous: bigint, smoothing: bigint): bigint {
        const inverse = BigInt(1e18) - smoothing;
        return (
            (current * smoothing + previous * inverse) / BigInt(1e18)
        );
    }

    /**
     * Converts a regular number to basis points (1% = 100 bps)
     */
    static toBasisPoints(value: number): number {
        return Math.floor(value * 10000);
    }

    /**
     * Converts basis points to a regular number
     */
    static fromBasisPoints(bps: number): number {
        return bps / 10000;
    }
} 