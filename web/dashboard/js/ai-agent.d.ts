import { ArbitrageOpportunity } from '../../../scripts/types/dex';
import { ProcessedBlock } from '../../../scripts/types/blocks';

declare module '@ai-agent' {
    export interface MarketVolatility {
        globalVolatility: number;
        tokenVolatilities: Map<string, number>;
        timestamp: number;
    }

    export interface MarketSentiment {
        score: number;
        indicators: {
            technical: number;
            fundamental: number;
            social: number;
        };
        timestamp: number;
    }

    export interface MarketTrends {
        strength: number;
        direction: 'up' | 'down' | 'sideways';
        duration: number;
        confidence: number;
    }

    export interface MEVActivity {
        activityLevel: number;
        recentAttacks: number;
        riskLevel: number;
        timestamp: number;
    }

    export interface FlashbotMetrics {
        activityLevel: number;
        bundleCount: number;
        successRate: number;
        avgProfit: bigint;
    }

    export interface PrivateTransactionMetrics {
        count: number;
        volume: bigint;
        avgSize: bigint;
        timestamp: number;
    }

    export interface RiskMetrics {
        riskLevel: number;
        probability: number;
        impact: number;
        mitigationScore: number;
    }

    export class AIAgent {
        constructor();
        
        getMarketVolatility(): Promise<MarketVolatility>;
        getMarketSentiment(): Promise<MarketSentiment>;
        getMarketTrends(): Promise<MarketTrends>;
        getMEVActivity(): Promise<MEVActivity>;
        getFlashbotMetrics(): Promise<FlashbotMetrics>;
        getPrivateTransactionMetrics(): Promise<PrivateTransactionMetrics>;
        getFrontrunningRisk(): Promise<RiskMetrics>;
        getSandwichAttackRisk(): Promise<RiskMetrics>;
        
        findAllArbitrageOpportunities(): Promise<ArbitrageOpportunity[]>;
        analyzeArbitrageOpportunity(params: {
            opportunity: ArbitrageOpportunity;
            block: ProcessedBlock;
            marketState: any;
            gasPrice: bigint;
        }): Promise<{
            isViable: boolean;
            confidence: number;
            expectedProfit: bigint;
            optimizedGas: bigint;
            mevRisk: {
                riskScore: number;
                type: string;
                probability: number;
            };
            recommendedPath: string[];
            expectedSlippage: number;
        }>;

        testSandwichProtection(params: {
            slippageBuffer: number;
            gasAdjustmentFactor: number;
        }): Promise<boolean>;

        testFrontrunningProtection(params: {
            maxPathLength: number;
            minProfitThreshold: bigint;
        }): Promise<boolean>;
    }
} 