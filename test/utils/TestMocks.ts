import { DEXType, DexInfo, Pool, ArbitrageOpportunity } from '../../scripts/types/dex';
import { parseEther } from 'viem';

export class MockDexDataProvider {
    async getPools(): Promise<Pool[]> {
        return [{
            address: '0x1234567890123456789012345678901234567890' as `0x${string}`,
            token0: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
            token1: '0x4200000000000000000000000000000000000006',
            reserve0: parseEther('1000000'),
            reserve1: parseEther('1000'),
            fee: 0.003
        }];
    }
}

export class MockGasOptimizer {
    async optimizeGas(profit: bigint): Promise<bigint> {
        return 200000n;
    }
}

export class MockVolumeAnalyzer {
    async analyzeVolume(): Promise<number> {
        return 1000000;
    }
}

export class MockAIAgent {
    async findAllArbitrageOpportunities(): Promise<ArbitrageOpportunity[]> {
        return [
            {
                path: ['Uniswap', 'Sushiswap'] as DEXType[],
                tokens: [
                    '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
                    '0x4200000000000000000000000000000000000006'
                ] as `0x${string}`[],
                expectedProfit: parseEther('0.05'),
                confidence: 0.95,
                amountIn: parseEther('1'),
                gasEstimate: 200000n,
                route: [
                    { 
                        dex: 'Uniswap' as DEXType, 
                        path: ['0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913'] as `0x${string}`[] 
                    },
                    { 
                        dex: 'Sushiswap' as DEXType, 
                        path: ['0x4200000000000000000000000000000000000006'] as `0x${string}`[] 
                    }
                ],
                amounts: [parseEther('1'), parseEther('1.05')],
                mevRisk: { riskScore: 0.1, type: 'low', probability: 0.05 },
                optimizedGas: 200000n,
                expectedSlippage: 0.001,
                recommendedPath: ['Uniswap', 'Sushiswap']
            }
        ];
    }

    async analyzeArbitrageOpportunity(): Promise<any> {
        return {
            isViable: true,
            confidence: 0.95,
            expectedProfit: parseEther('0.05'),
            optimizedGas: 200000n,
            mevRisk: { riskScore: 0.1, type: 'low', probability: 0.05 },
            recommendedPath: ['Uniswap', 'Sushiswap'],
            expectedSlippage: 0.001
        };
    }

    async optimizeExecutionParameters(): Promise<any> {
        return {
            amounts: [parseEther('1'), parseEther('1.05')],
            gasPrice: 50000000000n,
            deadline: BigInt(Math.floor(Date.now() / 1000) + 3600)
        };
    }

    async optimizeGasStrategy(): Promise<any> {
        return {
            gasPrice: 50000000000n,
            maxFeePerGas: 100000000000n,
            maxPriorityFeePerGas: 2000000000n
        };
    }

    async predictMEVRisk(): Promise<any> {
        return {
            riskScore: 0.1,
            type: 'low',
            probability: 0.05
        };
    }

    async getMarketVolatility() {
        return { globalVolatility: 0.02 };
    }

    async getMarketSentiment() {
        return { score: 0.7 };
    }

    async getMarketTrends() {
        return { strength: 0.8 };
    }

    async getMEVActivity() {
        return { activityLevel: 0.3 };
    }

    async getFlashbotMetrics() {
        return { activityLevel: 0.2 };
    }

    async getPrivateTransactionMetrics() {
        return { count: 5 };
    }

    async getFrontrunningRisk() {
        return { riskLevel: 0.1 };
    }

    async getSandwichAttackRisk() {
        return { riskLevel: 0.15 };
    }

    async testSandwichProtection() {
        return true;
    }

    async testFrontrunningProtection() {
        return true;
    }
} 
