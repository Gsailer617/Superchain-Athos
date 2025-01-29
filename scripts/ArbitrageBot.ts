import { 
    createPublicClient, 
    http, 
    PublicClient, 
    parseEther,
    formatEther,
    formatUnits,
    Transport, 
    TransactionType, 
    Block,
    Transaction,
    GetBlockReturnType,
    Chain,
    TransactionReceipt,
    Address,
    HttpTransport,
    createTransport,
    Account,
    Hash,
    BlockTag,
    FormattedTransaction,
    TransactionEIP1559,
    TransactionLegacy,
    Client,
    WatchBlocksReturnType,
    encodeAbiParameters
} from 'viem';
import { base } from 'viem/chains';
import { QUALITY_THRESHOLDS } from './config/constants';
import { DexDataProvider } from '../services/DexDataProvider';
import { PairGasOptimizer, UrgencyLevel } from './utils/PairGasOptimizer';
import { VolumeAnalyzer } from './utils/VolumeAnalyzer';
import { 
    DEXType,
    DexInfo,
    FlashLoanParams,
    FlashLoanProvider,
    VolumeMetrics, 
    LiquidityAnalysis,
    PriceImpactAnalysis,
    Pool,
    LendingProtocolConfig
} from './types/dex';
import { ProcessedTransaction } from './types/blocks';
import * as ethers from 'ethers';
import { TokenValidationLib } from './utils/TokenValidationLib';
import { MLMonitoring } from './utils/MLMonitoring';
import { DEX_CONFIGS } from './config/dexConfigs';
import { factoryABI, pairABI } from './config/contractABIs';
import { DEXConfig } from './config/dexConfigs';
import { LENDING_CONFIGS } from './config/lendingConfigs';

type BigNumberish = bigint | number | string;

interface ExtendedProcessedBlock {
    number: bigint;
    timestamp: bigint;
    transactions: ProcessedTransaction[];
    baseFeePerGas?: bigint;
}

interface DexNode {
    type: DEXType;
    info: DexInfo;
    pools: Map<`0x${string}`, Pool>;
}

interface AIAgent {
    findAllArbitrageOpportunities(): Promise<ArbitrageOpportunity[]>;
    analyzeArbitrageOpportunity(params: {
        opportunity: ArbitrageOpportunity;
        block: ExtendedProcessedBlock;
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
    optimizeExecutionParameters(opportunity: ArbitrageOpportunity): Promise<{
        amounts: bigint[];
        gasPrice: bigint;
        deadline: bigint;
    }>;
    optimizeGasStrategy(params: any): Promise<{
        gasPrice: bigint;
        maxFeePerGas: bigint;
        maxPriorityFeePerGas: bigint;
    }>;
    predictMEVRisk(params: any): Promise<{
        riskScore: number;
        type: string;
        probability: number;
    }>;
    getMarketVolatility(): Promise<{ globalVolatility: number }>;
    getMarketSentiment(): Promise<{ score: number }>;
    getMarketTrends(): Promise<{ strength: number }>;
    getMEVActivity(): Promise<{ activityLevel: number }>;
    getFlashbotMetrics(): Promise<{ activityLevel: number }>;
    getPrivateTransactionMetrics(): Promise<{ count: number }>;
    getFrontrunningRisk(): Promise<{ riskLevel: number }>;
    getSandwichAttackRisk(): Promise<{ riskLevel: number }>;
    testSandwichProtection(params: { slippageBuffer: number; gasAdjustmentFactor: number }): Promise<boolean>;
    testFrontrunningProtection(params: { maxPathLength: number; minProfitThreshold: bigint }): Promise<boolean>;
}

type ChainTransaction<TChain extends Chain> = FormattedTransaction<TChain>;
type TransactionData<TChain extends Chain = typeof base> = (
    | (TransactionEIP1559<bigint, number, false> & { input: `0x${string}` })
    | (TransactionLegacy<bigint, number, false> & { input: `0x${string}` })
    | TransactionWithDeposit
) & {
    maxFeePerGas?: bigint;
    maxPriorityFeePerGas?: bigint;
    gasPrice?: bigint;
};

type BlockData<TChain extends Chain = typeof base> = GetBlockReturnType<TChain, true>;

interface LiquiditySnapshot {
    timeStamp: number;
    slippage: number;
}

interface EmergencyCondition {
    type: string;
    threshold: number;
    multiplier: number;
    active: boolean;
}

interface Incident {
    type: string;
    tokenAddress: `0x${string}`;
    timestamp: number;
    conditions: EmergencyCondition[];
    slippage: number;
}

interface SlippageEvent {
    type: string;
    timestamp: number;
    tokenAddress: `0x${string}`;
    slippage: number;
    reason: string;
    conditions?: EmergencyCondition[];
    recommendation?: string;
}

type TransactionWithInput = Transaction & {
    input: `0x${string}`;
    maxFeePerGas?: bigint;
    maxPriorityFeePerGas?: bigint;
};

type ExtendedBlock = Block & {
    transactions: (Transaction & { input?: `0x${string}` })[];
};

interface TransactionWithDeposit {
    type: 'deposit';
    hash: `0x${string}`;
    to: `0x${string}` | null;
    from: `0x${string}`;
    gas: bigint;
    nonce: number;
    value: bigint;
    maxFeePerBlobGas?: undefined;
    gasPrice?: undefined;
    maxFeePerGas: bigint;
    maxPriorityFeePerGas: bigint;
    transactionIndex: number | null;
    input: `0x${string}`;
}

interface ProcessedBlock {
    number: bigint;
    timestamp: bigint;
    transactions: ProcessedTransaction[];
    baseFeePerGas?: bigint;
}

interface ArbitragePath {
    path: DEXType[];
    tokens: `0x${string}`[];
    expectedProfit: bigint;
    confidence: number;
    amounts: bigint[];
    route: { dex: DEXType; path: `0x${string}`[] }[];
    profit?: bigint;
}

type ArbitrageOpportunity = {
    path: DEXType[];
    tokens: `0x${string}`[];
    expectedProfit: bigint;
    confidence: number;
    amountIn: bigint;
    gasEstimate: bigint;
    route: { dex: DEXType; path: `0x${string}`[] }[];
    amounts: bigint[];
    mevRisk: { riskScore: number; type: string; probability: number };
    optimizedGas: bigint;
    expectedSlippage: number;
    recommendedPath: string[];
};

export class ArbitrageBot<TChain extends Chain = Chain> {
    private readonly dexDataProvider: DexDataProvider;
    private readonly gasOptimizer: PairGasOptimizer;
    private readonly volumeAnalyzer: VolumeAnalyzer;
    private readonly aiAgent: AIAgent;
    private readonly client: PublicClient;
    private readonly priceImpactThreshold = 0.05;
    private readonly minProfitThreshold = parseEther("0.001");
    private readonly maxPathLength = 3;
    private readonly slippageBuffer = 0.01;
    private readonly gasAdjustmentFactor = 1.2;
    private readonly profitOptimizationWindow = 2;
    private readonly minMevRiskThreshold = 0.5;
    private readonly dexGraph: Map<DEXType, DexNode> = new Map();
    private liquiditySnapshots: Map<string, LiquiditySnapshot[]> = new Map();
    private isRunning: boolean = false;
    private slippageMonitorHandle?: NodeJS.Timeout;
    private readonly SLIPPAGE_MONITOR_INTERVAL = 60000;
    private slippageEvents: SlippageEvent[] = [];
    private gasCosts: { current: bigint; average: bigint } = { current: 0n, average: 0n };
    private readonly SLIPPAGE_CONFIG = {
        min: 0.001,
        max: 0.1,
        dynamic: true
    };
    private readonly emergencyConditions: EmergencyCondition[] = [
        {
            type: 'volatility',
            threshold: 0.1,
            multiplier: 1.5,
            active: false
        },
        {
            type: 'liquidity',
            threshold: 1000000,
            multiplier: 2.0,
            active: false
        },
        {
            type: 'gas',
            threshold: 100,
            multiplier: 1.3,
            active: false
        }
    ];

    private readonly profitOptimizationParams = {
        minProfitMargin: 0.0005,
        optimalUtilization: 0.8,
        maxCrossSpread: 0.1,
        minLiquidityRatio: 0.2
    };

    private readonly SLIPPAGE_THRESHOLDS = {
        MIN: 0.001,
        MAX: 0.05,
        OPTIMAL: 0.003,
        ADJUSTMENT_STEP: 0.001
    };

    private readonly EXECUTION_PARAMS = {
        MAX_RETRIES: 3,
        RETRY_DELAY: 500,
        MIN_PROFIT_MULTIPLIER: 1.2,
        SANDWICH_PROTECTION: true,
        MEV_PROTECTION: true
    };

    private pendingTxs = new Map<`0x${string}`, {
        hash: `0x${string}`;
        to: `0x${string}`;
        from: `0x${string}`;
        data: `0x${string}`;
        value: bigint;
        gasPrice: bigint;
        maxFeePerGas?: bigint;
        maxPriorityFeePerGas?: bigint;
    }>();

    private readonly zeroAddress = '0x0000000000000000000000000000000000000000' as `0x${string}`;
    private readonly emptyHex = '0x0' as `0x${string}`;
    private readonly provider: ethers.Provider;
    private readonly mlModels: MLMonitoring;
    private readonly _discoveredPairs: Record<string, Record<string, Record<string, boolean>>> = {};
    private readonly _tokenMetrics: Record<string, { totalLiquidity: number; holdersCount: number; priceVolatility: number }> = {};
    private readonly lendingProtocols: Map<FlashLoanProvider, LendingProtocolConfig>;

    constructor(
        client: PublicClient,
        provider: ethers.Provider,
        mlModels: MLMonitoring,
        dexDataProvider?: DexDataProvider,
        gasOptimizer?: PairGasOptimizer,
        volumeAnalyzer?: VolumeAnalyzer,
        aiAgent?: AIAgent
    ) {
        this.client = client;
        this.provider = provider;
        this.mlModels = mlModels;
        this.dexDataProvider = dexDataProvider || new DexDataProvider();
        this.gasOptimizer = gasOptimizer || new PairGasOptimizer();
        this.volumeAnalyzer = volumeAnalyzer || new VolumeAnalyzer();
        this.aiAgent = aiAgent || this.createDefaultAIAgent();
        this.lendingProtocols = new Map(Object.entries(LENDING_CONFIGS).map(([_, config]) => [config.provider, config]));
        this.initializeMonitoring();
    }

    private createDefaultAIAgent(): AIAgent {
        return {
            findAllArbitrageOpportunities: async () => [],
            analyzeArbitrageOpportunity: async () => ({
                isViable: false,
                confidence: 0,
                expectedProfit: 0n,
                optimizedGas: 0n,
                mevRisk: { riskScore: 0, type: 'unknown', probability: 0 },
                recommendedPath: [],
                expectedSlippage: 0
            }),
            optimizeExecutionParameters: async () => ({
                amounts: [],
                gasPrice: 0n,
                deadline: 0n
            }),
            optimizeGasStrategy: async () => ({
                gasPrice: 0n,
                maxFeePerGas: 0n,
                maxPriorityFeePerGas: 0n
            }),
            predictMEVRisk: async () => ({
                riskScore: 0,
                type: 'unknown',
                probability: 0
            }),
            getMarketVolatility: async () => ({ globalVolatility: 0 }),
            getMarketSentiment: async () => ({ score: 0 }),
            getMarketTrends: async () => ({ strength: 0 }),
            getMEVActivity: async () => ({ activityLevel: 0 }),
            getFlashbotMetrics: async () => ({ activityLevel: 0 }),
            getPrivateTransactionMetrics: async () => ({ count: 0 }),
            getFrontrunningRisk: async () => ({ riskLevel: 0 }),
            getSandwichAttackRisk: async () => ({ riskLevel: 0 }),
            testSandwichProtection: async () => false,
            testFrontrunningProtection: async () => false
        };
    }

    private async initializeMonitoring(): Promise<void> {
        // Initialize monitoring systems
        await this.initializeGasMonitoring();
        await this.initializeTransactionMonitoring();
        this.startSlippageMonitoring();
        await this.initializeTokenDiscovery();
    }

    private async initializeGasMonitoring(): Promise<void> {
        try {
            const feeHistory = await this.client.getFeeHistory({
                blockCount: 1,
                rewardPercentiles: [50]
            });
            this.gasCosts.current = feeHistory.baseFeePerGas[0] || 0n;
            this.gasCosts.average = this.gasCosts.current;

            setInterval(async () => {
                try {
                    const feeHistory = await this.client.getFeeHistory({
                        blockCount: 1,
                        rewardPercentiles: [50]
                    });
                    this.gasCosts.current = feeHistory.baseFeePerGas[0] || this.gasCosts.current;
                    this.gasCosts.average = (this.gasCosts.average * 9n + this.gasCosts.current) / 10n;
                } catch (error) {
                    console.error('Error updating gas prices:', error);
                }
            }, 15000);
        } catch (error) {
            console.error('Error initializing gas monitoring:', error);
        }
    }

    private async initializeTransactionMonitoring(): Promise<void> {
        try {
            this.client.watchBlocks({
                onBlock: async (block) => {
                    try {
                        const blockData = await this.dexDataProvider.getBlockForBot(block.number.toString() as BlockTag);

                        const transactions = blockData.transactions;
                        if (!transactions || !Array.isArray(transactions)) return;

                        for (const tx of transactions) {
                            if (!tx.input || tx.input.length < 10) continue;

                            if (!this.isSwapTransaction(tx)) continue;

                            const gasPriceValue = tx.maxFeePerGas ?? 0n;

                            this.pendingTxs.set(tx.hash, {
                                hash: tx.hash,
                                to: tx.to || this.zeroAddress,
                                from: tx.from,
                                data: tx.input || this.emptyHex,
                                value: tx.value,
                                gasPrice: gasPriceValue,
                                maxFeePerGas: tx.maxFeePerGas,
                                maxPriorityFeePerGas: tx.maxPriorityFeePerGas
                            });
                        }
                    } catch (error) {
                        console.error('Error processing block transactions:', error);
                    }
                }
            });
        } catch (error) {
            console.error('Error initializing transaction monitoring:', error);
        }
    }

    private isSwapTransaction(tx: ProcessedTransaction): boolean {
        return tx.input.length >= 10 && this.isDexContract(tx.to);
    }

    private isDexContract(address: `0x${string}` | null): boolean {
        if (!address) return false;
        return true;
    }

    private isValidTransaction(tx: unknown): tx is Transaction<bigint, number, false> | TransactionWithDeposit {
        if (typeof tx === 'string') return false;
        if (!tx || typeof tx !== 'object') return false;
        
        const transaction = tx as any;
        return (
            'hash' in transaction &&
            'from' in transaction &&
            ('to' in transaction || transaction.to === null) &&
            'input' in transaction &&
            'value' in transaction &&
            typeof transaction.hash === 'string' &&
            typeof transaction.from === 'string' &&
            (typeof transaction.to === 'string' || transaction.to === null) &&
            typeof transaction.input === 'string' &&
            typeof transaction.value === 'bigint'
        );
    }

    private normalizeTransaction(tx: Transaction<bigint, number, false>): TransactionData<TChain> | null {
        try {
            const baseTx = {
                ...tx,
                input: tx.input || '0x'
            };

            if ('maxFeePerGas' in tx && 'maxPriorityFeePerGas' in tx) {
                return {
                    ...baseTx,
                    maxFeePerGas: tx.maxFeePerGas,
                    maxPriorityFeePerGas: tx.maxPriorityFeePerGas,
                    gasPrice: undefined
                } as TransactionData<TChain>;
            }
            
            if ('gasPrice' in tx) {
                return {
                    ...baseTx,
                    gasPrice: tx.gasPrice,
                    maxFeePerGas: undefined,
                    maxPriorityFeePerGas: undefined
                } as TransactionData<TChain>;
            }

            return null;
        } catch (error) {
            console.error('Error normalizing transaction:', error);
            return null;
        }
    }

    async findArbitrageOpportunities(): Promise<ArbitrageOpportunity[]> {
        try {
            console.log('\n🔍 Starting arbitrage opportunity search...');
            
            const opportunities = await this.aiAgent.findAllArbitrageOpportunities();
            console.log(`✨ Found ${opportunities.length} potential opportunities`);
            
            const blockNumber = await this.client.getBlockNumber();
            const block = await this.client.getBlock({ blockNumber });
            console.log(`📦 Analyzing block #${blockNumber}`);
            
            const processedBlock: ExtendedProcessedBlock = {
                number: blockNumber,
                timestamp: BigInt(block.timestamp),
                baseFeePerGas: block.baseFeePerGas ?? undefined,
                transactions: []
            };

            if (block.transactions.length > 0) {
                const txHashes = block.transactions as `0x${string}`[];
                const txPromises = txHashes.map(hash => this.client.getTransaction({ hash }));
                const transactions = await Promise.all(txPromises);
                processedBlock.transactions = transactions
                    .filter((tx): tx is NonNullable<typeof tx> => tx !== null)
                    .map(tx => ({
                        type: tx.type,
                        hash: tx.hash,
                        blockNumber: tx.blockNumber ?? 0n,
                        from: tx.from,
                        to: tx.to,
                        value: tx.value,
                        gas: tx.gas,
                        nonce: tx.nonce,
                        maxFeePerGas: tx.maxFeePerGas ?? 0n,
                        maxPriorityFeePerGas: tx.maxPriorityFeePerGas,
                        input: tx.input
                    } as ProcessedTransaction));
            }

            console.log('\n🧠 AI analyzing opportunities...');
            const analyzedOpportunities = await Promise.all(
                opportunities.map(async (opp, index) => {
                    console.log(`\n📝 Analyzing opportunity ${index + 1}/${opportunities.length}:`);
                    console.log(`   Path: ${opp.path.join(' → ')}`);
                    console.log(`   Expected Profit: ${formatEther(opp.expectedProfit)} ETH`);

                    const analysis = await this.aiAgent.analyzeArbitrageOpportunity({
                        opportunity: opp,
                        block: processedBlock,
                        marketState: await this.getMarketState(),
                        gasPrice: await this.getCurrentGasPrice()
                    });

                    if (analysis.isViable) {
                        console.log('   ✅ Opportunity is viable:');
                        console.log(`      - Confidence: ${(analysis.confidence * 100).toFixed(2)}%`);
                        console.log(`      - Expected Profit: ${formatEther(analysis.expectedProfit)} ETH`);
                        console.log(`      - Gas Cost: ${formatEther(analysis.optimizedGas)} ETH`);
                        console.log(`      - MEV Risk: ${analysis.mevRisk.riskScore.toFixed(2)}`);
                        console.log(`      - Expected Slippage: ${(analysis.expectedSlippage * 100).toFixed(2)}%`);
                        return {
                            ...opp,
                            confidence: analysis.confidence,
                            expectedProfit: analysis.expectedProfit,
                            optimizedGas: analysis.optimizedGas,
                            mevRisk: analysis.mevRisk,
                            recommendedPath: analysis.recommendedPath,
                            expectedSlippage: analysis.expectedSlippage
                        };
                    } else {
                        console.log('   ❌ Opportunity not viable');
                        return null;
                    }
                })
            );

            const validOpportunities = analyzedOpportunities.filter((opp): opp is ArbitrageOpportunity => opp !== null);
            console.log(`\n🎯 Found ${validOpportunities.length} viable opportunities`);
            
            return validOpportunities;
        } catch (error) {
            console.error('❌ Error finding arbitrage opportunities:', error);
            return [];
        }
    }

    private async getMarketState() {
        console.log('\n📊 Gathering market state data...');
        const state = {
            globalVolatility: await this.calculateGlobalVolatility(),
            marketSentiment: await this.calculateMarketSentiment(),
            trendStrength: await this.calculateTrendStrength(),
            networkCongestion: await this.calculateNetworkCongestion(),
            pendingTransactions: await this.getPendingTransactionCount(),
            blockTime: await this.getAverageBlockTime(),
            mevActivity: await this.calculateMEVActivity(),
            flashbotActivity: await this.getFlashbotActivity(),
            privateTransactions: await this.getPrivateTransactionCount(),
            frontrunningRisk: await this.calculateFrontrunningRisk(),
            sandwichRisk: await this.calculateSandwichRisk()
        };

        console.log('Market State Summary:');
        console.log(`   - Volatility: ${(state.globalVolatility * 100).toFixed(2)}%`);
        console.log(`   - Market Sentiment: ${(state.marketSentiment * 100).toFixed(2)}%`);
        console.log(`   - Network Congestion: ${(state.networkCongestion * 100).toFixed(2)}%`);
        console.log(`   - MEV Activity Level: ${(state.mevActivity * 100).toFixed(2)}%`);
        console.log(`   - Frontrunning Risk: ${(state.frontrunningRisk * 100).toFixed(2)}%`);

        return state;
    }

    async executeArbitrageOpportunity(opportunity: ArbitrageOpportunity): Promise<boolean> {
        try {
            console.log('\n🚀 Executing arbitrage opportunity:');
            console.log(`Path: ${opportunity.path.join(' → ')}`);
            console.log(`Expected Profit: ${formatEther(opportunity.expectedProfit)} ETH`);

            console.log('🧠 Getting AI optimized execution parameters...');
            const executionParams = await this.aiAgent.optimizeExecutionParameters(opportunity);
            
            console.log('📝 Preparing flash loan parameters...');
            const flashLoanParams: FlashLoanParams = {
                tokenIn: opportunity.tokens[0],
                amount: opportunity.amountIn,
                fee: 0.0009,
                provider: this.getBestFlashLoanProvider(opportunity.tokens[0], opportunity.amountIn),
                assets: this.getAssetsFromPath(opportunity.recommendedPath),
                amounts: executionParams.amounts,
                modes: Array(executionParams.amounts.length).fill(0),
                params: this.encodeArbitrageParams(opportunity, executionParams),
                maxFeePerGas: executionParams.gasPrice,
                maxPriorityFeePerGas: executionParams.gasPrice / 2n,
                deadline: executionParams.deadline,
                maxBorrowRates: Array(executionParams.amounts.length).fill(0n),
                pairs: this.getAssetsFromPath(opportunity.recommendedPath),
                slippage: 0.01
            };

            if (opportunity.mevRisk.riskScore > this.minMevRiskThreshold) {
                console.log('🛡️ Using MEV protection for high-risk trade...');
                return await this.executeProtectedArbitrage(flashLoanParams);
            } else {
                console.log('🔄 Executing standard arbitrage trade...');
                return await this.executeStandardArbitrage(flashLoanParams);
            }
        } catch (error) {
            console.error('❌ Error executing arbitrage opportunity:', error);
            return false;
        }
    }

    private async executeProtectedArbitrage(params: FlashLoanParams): Promise<boolean> {
        console.log('🛡️ Preparing protected transaction...');
        const protectedTx = await this.prepareProtectedTransaction(params);
        
        console.log('📡 Sending protected transaction...');
        const receipt = await this.sendProtectedTransaction(protectedTx);
        
        if (receipt) {
            console.log('✅ Protected transaction successful!');
            console.log(`Transaction Hash: ${receipt.transactionHash}`);
            console.log(`Gas Used: ${receipt.gasUsed}`);
            return true;
        } else {
            console.log('❌ Protected transaction failed');
            return false;
        }
    }

    private async executeStandardArbitrage(params: FlashLoanParams): Promise<boolean> {
        console.log('🔄 Executing standard flash loan...');
        const receipt = await this.executeFlashLoan(params);
        
        if (receipt) {
            console.log('✅ Flash loan execution successful!');
            console.log(`Transaction Hash: ${receipt.transactionHash}`);
            console.log(`Gas Used: ${receipt.gasUsed}`);
            return true;
        } else {
            console.log('❌ Flash loan execution failed');
            return false;
        }
    }

    private async prepareProtectedTransaction(params: FlashLoanParams): Promise<any> {
        const gasStrategy = await this.aiAgent.optimizeGasStrategy({
            currentGasPrice: await this.getCurrentGasPrice(),
            mevActivity: await this.calculateMEVActivity(),
            networkCongestion: await this.calculateNetworkCongestion()
        });

        return {
            ...params,
            maxFeePerGas: gasStrategy.maxFeePerGas,
            maxPriorityFeePerGas: gasStrategy.maxPriorityFeePerGas
        };
    }

    private async sendProtectedTransaction(tx: any): Promise<any> {
        try {
            return null;
        } catch (error) {
            console.error('Error sending protected transaction:', error);
            return null;
        }
    }

    private getBestFlashLoanProvider(tokenIn: `0x${string}`, amount: bigint): FlashLoanProvider {
        let bestProvider = FlashLoanProvider.MOONWELL;
        let lowestFee = Infinity;

        for (const [provider, config] of this.lendingProtocols.entries()) {
            if (config.supportedTokens.includes(tokenIn) && config.flashLoanFee < lowestFee) {
                bestProvider = provider;
                lowestFee = config.flashLoanFee;
            }
        }

        return bestProvider;
    }

    private async executeFlashLoan(params: FlashLoanParams): Promise<any> {
        try {
            // Dynamically select best provider based on token and amount
            const bestProvider = this.getBestFlashLoanProvider(params.tokenIn, params.amount);
            const lendingConfig = this.lendingProtocols.get(bestProvider);

            if (!lendingConfig) {
                throw new Error(`No suitable flash loan provider found for token ${params.tokenIn}`);
            }

            // Use provider-specific fee
            const flashLoanParams: FlashLoanParams = {
                ...params,
                provider: bestProvider,
                fee: lendingConfig.flashLoanFee
            };
            
            // Implementation here
            return null;
        } catch (error) {
            console.error('Error executing flash loan:', error);
            return null;
        }
    }

    private async buildDexGraph(): Promise<void> {
        const dexTypes = Object.values(DEXType);
        
        for (const type of dexTypes) {
            const pools = await this.dexDataProvider.getAllPools(type);
            await this.initializeDexNode(type, pools);
        }
    }

    private async initializeDexNode(dex: DEXType, pools: Pool[]): Promise<void> {
        const dexInfo: DexInfo = {
            type: dex,
            name: dex.toString(),
            address: this.zeroAddress,
            version: '2.0',
            factoryAddress: this.zeroAddress,
            routerAddress: this.zeroAddress,
            initCodeHash: this.emptyHex,
            supportedTokens: []
        };

        this.dexGraph.set(dex, {
            type: dex,
            info: dexInfo,
            pools: new Map(pools.map(pool => [
                pool.address,
                {
                    address: pool.address,
                    token0: pool.token0,
                    token1: pool.token1,
                    reserve0: pool.reserve0,
                    reserve1: pool.reserve1,
                    fee: pool.fee
                }
            ]))
        });
    }

    private async findBestSwap(
        sourceDex: { type: DEXType; pools: Map<`0x${string}`, Pool> },
        targetDex: { type: DEXType; pools: Map<`0x${string}`, Pool> }
    ): Promise<ArbitrageOpportunity | null> {
        try {
            const sourcePool = await this.getPoolData(sourceDex.type);
            const targetPool = await this.getPoolData(targetDex.type);

            if (!sourcePool?.token0 || !targetPool?.token0) {
                return null;
            }

            const profitResult = await this.calculateProfit(sourcePool, targetPool);
            
            if (profitResult.profit <= 0n) {
                return null;
            }

            const gasPrice = await this.optimizeGasPrice(profitResult.profit);
            const gasEstimate = this.estimateGasCost(2);
            const gasCost = gasPrice * gasEstimate;

            const netProfit = profitResult.profit - gasCost;
            if (netProfit <= 0n) {
                return null;
            }

            return {
                path: [sourceDex.type, targetDex.type],
                tokens: [sourcePool.token0],
                expectedProfit: netProfit,
                confidence: profitResult.confidence,
                amountIn: profitResult.amountIn,
                gasEstimate,
                route: [
                    { dex: sourceDex.type, path: [sourcePool.token0] },
                    { dex: targetDex.type, path: [targetPool.token0] }
                ],
                amounts: [profitResult.amountIn, profitResult.intermediateAmount, profitResult.finalAmount],
                mevRisk: { riskScore: 0, type: 'low', probability: 0.1 },
                optimizedGas: gasPrice,
                expectedSlippage: 0.005,
                recommendedPath: [sourcePool.address, targetPool.address]
            };
        } catch (error) {
            console.error('Error finding best swap:', error);
            return null;
        }
    }

    private async analyzeArbitragePath(
        path: DEXType[],
        block: ProcessedBlock
    ): Promise<ArbitrageOpportunity | null> {
        try {
            const sourcePool = await this.getPoolData(path[0]);
            const targetPool = await this.getPoolData(path[1]);

            if (!sourcePool?.token0 || !targetPool?.token0) {
                return null;
            }

            const profitResult = await this.calculateProfit(sourcePool, targetPool);

            if (profitResult.profit <= 0n) {
                return null;
            }

            const gasPrice = await this.optimizeGasPrice(profitResult.profit);
            const gasEstimate = this.estimateGasCost(path.length);
            const gasCost = gasPrice * gasEstimate;

            const netProfit = profitResult.profit - gasCost;
            if (netProfit <= 0n) {
                return null;
            }

            return {
                path,
                tokens: [sourcePool.token0],
                expectedProfit: netProfit,
                confidence: profitResult.confidence,
                amountIn: profitResult.amountIn,
                gasEstimate,
                route: path.map(dex => ({
                    dex,
                    path: [sourcePool.token0]
                })),
                amounts: [profitResult.amountIn, profitResult.intermediateAmount, profitResult.finalAmount],
                mevRisk: { riskScore: 0, type: 'low', probability: 0.1 },
                optimizedGas: gasPrice,
                expectedSlippage: 0.005,
                recommendedPath: path.map(dex => this.zeroAddress)
            };
        } catch (error) {
            console.error('Error analyzing arbitrage path:', error);
            return null;
        }
    }

    private async getPoolData(dexType: DEXType): Promise<Pool | null> {
        try {
            const pool = await this.dexDataProvider.getPoolData(dexType, this.zeroAddress);
            return pool;
        } catch (error) {
            console.error(`Error getting pool data for ${dexType}:`, error);
            return null;
        }
    }

    private async calculateProfit(
        sourcePool: Pool,
        targetPool: Pool
    ): Promise<{ profit: bigint; confidence: number; amountIn: bigint; intermediateAmount: bigint; finalAmount: bigint }> {
        try {
            const amountIn = this.calculateOptimalTradeSize(sourcePool.reserve0, sourcePool.reserve1);
            
            const intermediateAmount = this.calculateSwapOutput(
                amountIn,
                sourcePool.reserve0,
                sourcePool.reserve1,
                sourcePool.fee
            );

            const finalAmount = this.calculateSwapOutput(
                intermediateAmount,
                targetPool.reserve0,
                targetPool.reserve1,
                targetPool.fee
            );

            const profit = finalAmount - amountIn;

            const sourceDepth = Number(sourcePool.reserve0 + sourcePool.reserve1) / 2;
            const targetDepth = Number(targetPool.reserve0 + targetPool.reserve1) / 2;
            const averageDepth = (sourceDepth + targetDepth) / 2;
            
            const confidence = Math.max(0, Math.min(1, averageDepth / Number(amountIn)));

            return { profit, confidence, amountIn, intermediateAmount, finalAmount };
        } catch (error) {
            console.error('Error calculating profit:', error);
            return { profit: 0n, confidence: 0, amountIn: 0n, intermediateAmount: 0n, finalAmount: 0n };
        }
    }

    private async optimizeGasPrice(potentialProfit: bigint): Promise<bigint> {
        try {
            const strategy = await this.aiAgent.optimizeGasStrategy({ marketState: { profit: potentialProfit } });
            return strategy.maxFeePerGas || 50000000000n;
        } catch (error) {
            console.error('Error optimizing gas price:', error);
            return 50000000000n;
        }
    }

    private estimateGasCost(pathLength: number): bigint {
        const baseCost = 100000n;
        const perHopCost = 50000n;
        return baseCost + (perHopCost * BigInt(pathLength));
    }

    private calculateSwapOutput(
        amountIn: bigint,
        reserve0: bigint,
        reserve1: bigint,
        fee: number
    ): bigint {
        const amountInWithFee = amountIn * BigInt(Math.floor((1 - fee) * 1000)) / 1000n;
        const numerator = amountInWithFee * reserve1;
        const denominator = reserve0 + amountInWithFee;
        return numerator / denominator;
    }

    private calculateOptimalTradeSize(reserve0: bigint, reserve1: bigint): bigint {
        return BigInt(Math.min(Number(reserve0), Number(reserve1))) / 100n;
    }

    private async calculateGlobalVolatility(): Promise<number> {
        try {
            const volatilityData = await this.aiAgent.getMarketVolatility();
            return volatilityData.globalVolatility;
        } catch (error) {
            console.error('Error calculating global volatility:', error);
            return 0;
        }
    }

    private async calculateMarketSentiment(): Promise<number> {
        try {
            const sentiment = await this.aiAgent.getMarketSentiment();
            return sentiment.score;
        } catch (error) {
            console.error('Error calculating market sentiment:', error);
            return 0;
        }
    }

    private async calculateTrendStrength(): Promise<number> {
        try {
            const trends = await this.aiAgent.getMarketTrends();
            return trends.strength;
        } catch (error) {
            console.error('Error calculating trend strength:', error);
            return 0;
        }
    }

    private async getPendingTransactionCount(): Promise<number> {
        try {
            const pendingBlock = await this.client.getBlock({ blockTag: 'pending' });
            return pendingBlock.transactions.length;
        } catch (error) {
            console.error('Error getting pending transaction count:', error);
            return 0;
        }
    }

    private async getAverageBlockTime(): Promise<number> {
        try {
            const latestBlock = await this.client.getBlock({ blockTag: 'latest' });
            const prevBlock = await this.client.getBlock({ 
                blockNumber: latestBlock.number - 1n 
            });
            return Number(latestBlock.timestamp - prevBlock.timestamp);
        } catch (error) {
            console.error('Error calculating average block time:', error);
            return 12;
        }
    }

    private async getFlashbotActivity(): Promise<number> {
        try {
            const flashbotData = await this.aiAgent.getFlashbotMetrics();
            return flashbotData.activityLevel;
        } catch (error) {
            console.error('Error getting flashbot activity:', error);
            return 0;
        }
    }

    private async getPrivateTransactionCount(): Promise<number> {
        try {
            const privateData = await this.aiAgent.getPrivateTransactionMetrics();
            return privateData.count;
        } catch (error) {
            console.error('Error getting private transaction count:', error);
            return 0;
        }
    }

    private async calculateFrontrunningRisk(): Promise<number> {
        try {
            const riskData = await this.aiAgent.getFrontrunningRisk();
            return riskData.riskLevel;
        } catch (error) {
            console.error('Error calculating frontrunning risk:', error);
            return 0;
        }
    }

    private async calculateSandwichRisk(): Promise<number> {
        try {
            const riskData = await this.aiAgent.getSandwichAttackRisk();
            return riskData.riskLevel;
        } catch (error) {
            console.error('Error calculating sandwich risk:', error);
            return 0;
        }
    }

    private async calculateMEVActivity(): Promise<number> {
        try {
            const mevData = await this.aiAgent.getMEVActivity();
            return mevData.activityLevel;
        } catch (error) {
            console.error('Error calculating MEV activity:', error);
            return 0;
        }
    }

    private async calculateNetworkCongestion(): Promise<number> {
        try {
            const block = await this.client.getBlock({ blockTag: 'latest' });
            const gasUsed = Number(block.gasUsed);
            const gasLimit = Number(block.gasLimit);
            return gasUsed / gasLimit;
        } catch (error) {
            console.error('Error calculating network congestion:', error);
            return 0;
        }
    }

    private async analyzeMEVRisk(block: ProcessedBlock, opportunity: ArbitrageOpportunity): Promise<{ riskScore: number; type: string; probability: number }> {
        const prediction = await this.aiAgent.predictMEVRisk({ marketState: { block, opportunity } });
        return prediction;
    }

    private async monitorSlippage(): Promise<void> {
        for (const [tokenAddress, snapshots] of this.liquiditySnapshots) {
            try {
                if (!snapshots || snapshots.length < 2) {
                    console.warn(`Insufficient snapshots for token ${tokenAddress}`);
                    continue;
                }

                const recentSnapshots = snapshots
                    .filter((s: LiquiditySnapshot) => s.timeStamp > Date.now() - 3600000)
                    .sort((a: LiquiditySnapshot, b: LiquiditySnapshot) => b.timeStamp - a.timeStamp);

                if (recentSnapshots.length < 2) {
                    console.debug(`Not enough recent snapshots for token ${tokenAddress}`);
                    continue;
                }

                const currentSlippage = recentSnapshots[0].slippage;
                const previousSlippage = recentSnapshots[1].slippage;
                
                if (previousSlippage === 0) {
                    console.warn(`Invalid previous slippage for token ${tokenAddress}`);
                    continue;
                }

                const slippageChangeRate = (currentSlippage - previousSlippage) / previousSlippage * 100;

                const emergency = await this.checkEmergencyConditions(tokenAddress as `0x${string}`, currentSlippage);
                
                if (!emergency) {
                    console.warn(`Failed to check emergency conditions for ${tokenAddress}`);
                    continue;
                }

                if (emergency.activeConditions.length > 0) {
                    await this.handleEmergencyConditions(tokenAddress, emergency);
                } else if (slippageChangeRate > 50) {
                    await this.handleHighSlippageWarning(tokenAddress, currentSlippage, slippageChangeRate);
                } else if (Math.abs(slippageChangeRate) > 20) {
                    await this.handleSignificantSlippageChange(tokenAddress, currentSlippage, slippageChangeRate);
                }
            } catch (error: unknown) {
                const errorMessage = error instanceof Error ? error.message : 'Unknown error';
                console.error(`Critical error monitoring slippage for ${tokenAddress}:`, error);
                await this.notifyAdmins(`Slippage monitoring error for ${tokenAddress}: ${errorMessage}`);
                await this.attemptRecovery(tokenAddress);
            }
        }
    }

    private startSlippageMonitoring(): void {
        this.slippageMonitorHandle = setInterval(
            () => this.monitorSlippage(),
            this.SLIPPAGE_MONITOR_INTERVAL
        );
    }

    public stop(): void {
        this.isRunning = false;
        if (this.slippageMonitorHandle) {
            clearInterval(this.slippageMonitorHandle);
        }
        console.log('ArbitrageBot stopped');
    }

    private async handleEmergencyConditions(
        tokenAddress: string, 
        emergency: { slippage: number; activeConditions: EmergencyCondition[] }
    ): Promise<void> {
        try {
            await this.recordSlippageEvent({
                type: 'emergency',
                timestamp: Date.now(),
                tokenAddress,
                slippage: emergency.slippage,
                reason: 'Emergency conditions detected',
                conditions: emergency.activeConditions,
                recommendation: this.getEmergencyRecommendation(emergency.activeConditions)
            });
            
            await this.executeEmergencyProtocol(tokenAddress, emergency);
        } catch (error: unknown) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error('Failed to handle emergency conditions:', error);
            await this.notifyAdmins(`Emergency handling failed for ${tokenAddress}: ${errorMessage}`);
        }
    }

    private async executeEmergencyProtocol(
        tokenAddress: string, 
        emergency: { slippage: number; activeConditions: EmergencyCondition[] }
    ): Promise<void> {
        await this.pauseTrading(tokenAddress);
        
        await this.notifyAdmins(`Emergency protocol activated for ${tokenAddress}`);
        
        await this.logIncident({
            type: 'EMERGENCY_PROTOCOL',
            tokenAddress,
            timestamp: Date.now(),
            conditions: emergency.activeConditions,
            slippage: emergency.slippage
        });
    }

    private async attemptRecovery(tokenAddress: string): Promise<void> {
        try {
            this.liquiditySnapshots.set(tokenAddress, []);
            
            await this.initializeTokenMonitoring(tokenAddress);
            
            console.log(`Successfully recovered monitoring for ${tokenAddress}`);
        } catch (error: unknown) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error(`Recovery failed for ${tokenAddress}:`, error);
            await this.notifyAdmins(`Recovery failed for ${tokenAddress}: ${errorMessage}`);
        }
    }

    protected async notifyAdmins(message: string): Promise<void> {
        console.log(`[ADMIN NOTIFICATION] ${message}`);
    }

    protected async pauseTrading(tokenAddress: string): Promise<void> {
        console.log(`[TRADING PAUSED] for token ${tokenAddress}`);
    }

    protected async logIncident(incident: Omit<Incident, 'tokenAddress'> & { tokenAddress: string }): Promise<void> {
        const formattedIncident: Incident = {
            ...incident,
            tokenAddress: incident.tokenAddress as `0x${string}`
        };
        console.log(`[INCIDENT LOGGED] ${JSON.stringify(formattedIncident)}`);
    }

    protected async handleHighSlippageWarning(
        tokenAddress: string, 
        currentSlippage: number, 
        slippageChangeRate: number
    ): Promise<void> {
        console.log(`[HIGH SLIPPAGE WARNING] Token: ${tokenAddress}, Rate: ${slippageChangeRate}%`);
    }

    protected async handleSignificantSlippageChange(
        tokenAddress: string, 
        currentSlippage: number, 
        slippageChangeRate: number
    ): Promise<void> {
        console.log(`[SLIPPAGE CHANGE] Token: ${tokenAddress}, Rate: ${slippageChangeRate}%`);
    }

    protected async checkEmergencyConditions(
        tokenAddress: `0x${string}`, 
        currentSlippage: number
    ): Promise<{ slippage: number; activeConditions: EmergencyCondition[] }> {
        return {
            slippage: currentSlippage,
            activeConditions: []
        };
    }

    protected getEmergencyRecommendation(conditions: EmergencyCondition[]): string {
        return conditions.map(c => `Handle ${c.type} condition`).join(', ');
    }

    protected async recordSlippageEvent(event: Omit<SlippageEvent, 'tokenAddress'> & { tokenAddress: string }): Promise<void> {
        const formattedEvent: SlippageEvent = {
            ...event,
            tokenAddress: event.tokenAddress as `0x${string}`
        };
        console.log(`[SLIPPAGE EVENT] ${JSON.stringify(formattedEvent)}`);
    }

    private getAssetsFromPath(path: string[]): `0x${string}`[] {
        return path.map(address => address as `0x${string}`);
    }

    private encodeArbitrageParams(
        opportunity: ArbitrageOpportunity, 
        executionParams: any
    ): `0x${string}` {
        const encoded = encodeAbiParameters(
            [
                { type: 'address[]' },
                { type: 'uint256[]' },
                { type: 'bytes' }
            ],
            [
                opportunity.tokens,
                executionParams.amounts,
                '0x'
            ]
        ) as `0x${string}`;
        return encoded;
    }

    private async findAllPaths(dexTypes: DEXType[], maxLength: number): Promise<ArbitrageOpportunity[]> {
        return [];
    }

    private async getCurrentGasPrice(): Promise<bigint> {
        const block = await this.client.getBlock({ blockTag: 'latest' });
        return block.baseFeePerGas || 0n;
    }

    private async initializeTokenDiscovery() {
        // Initialize factory listeners for all configured DEXes
        for (const dex of Object.values(DEX_CONFIGS)) {
            const factory = new ethers.Contract(dex.factory, factoryABI, this.provider);
            
            // Listen for new pair creation events
            factory.on("PairCreated", async (token0: `0x${string}`, token1: `0x${string}`, pair: `0x${string}`) => {
                if (!this._discoveredPairs[dex.factory]?.[token0]?.[token1]) {
                    await this.validateAndAddNewPair(token0, token1, pair, dex);
                }
            });
        }
    }

    private async validateAndAddNewPair(token0: `0x${string}`, token1: `0x${string}`, pair: `0x${string}`, dex: DEXConfig) {
        try {
            // Basic validation using existing TokenValidationLib
            const [isValid0, reason0] = await TokenValidationLib.validateTokenSecurity(token0);
            const [isValid1, reason1] = await TokenValidationLib.validateTokenSecurity(token1);

            if (!isValid0 || !isValid1) {
                console.log(`Skipping invalid pair: ${reason0 || reason1}`);
                return;
            }

            // Get liquidity and volume data
            const pairContract = new ethers.Contract(pair, pairABI, this.provider);
            const [reserves0, reserves1] = await pairContract.getReserves();
            
            // Check against quality thresholds
            const liquidityUSD = await this.calculateLiquidityUSD(reserves0, reserves1, token0, token1);
            if (liquidityUSD < QUALITY_THRESHOLDS.MIN_LIQUIDITY_USD) {
                return;
            }

            // ML-based scoring using existing MLMonitoring
            const features = await this.extractTokenFeatures(token0, token1, pair);
            const score = await this.mlModels.predictTokenScore(features);

            if (score > QUALITY_THRESHOLDS.HEALTH_SCORE_THRESHOLD) {
                // Record new pair using existing MLMonitoring functions
                await this.recordNewPair(token0, token1, dex.factory, pair, dex.fees[0]);
                
                // Update metrics
                if (!this._tokenMetrics[token0]) {
                    this._tokenMetrics[token0] = { totalLiquidity: 0, holdersCount: 0, priceVolatility: 0 };
                }
                if (!this._tokenMetrics[token1]) {
                    this._tokenMetrics[token1] = { totalLiquidity: 0, holdersCount: 0, priceVolatility: 0 };
                }
                this._tokenMetrics[token0].totalLiquidity += Number(reserves0);
                this._tokenMetrics[token1].totalLiquidity += Number(reserves1);

                console.log(`New pair discovered and validated: ${token0}-${token1} on ${dex.name}`);
                
                // Notify the dashboard
                await this.emitNewPairDiscovered({
                    token0,
                    token1,
                    pair,
                    dex: dex.name,
                    liquidity: liquidityUSD,
                    score
                });
            }
        } catch (error) {
            console.error('Error validating new pair:', error);
        }
    }

    private async extractTokenFeatures(token0: `0x${string}`, token1: `0x${string}`, pair: `0x${string}`) {
        // Use existing metrics collection
        const volume = await this.volumeAnalyzer.analyzeVolumeByAddress(pair);
        const metrics = await this.getTokenMetrics(token0, token1);
        
        return {
            volume: Number(volume),
            liquidity: metrics.totalLiquidity,
            holdersCount: metrics.holdersCount,
            priceVolatility: metrics.priceVolatility
        };
    }

    private async calculateLiquidityUSD(reserves0: number, reserves1: number, token0: string, token1: string): Promise<number> {
        // Implement the logic to calculate liquidity in USD based on reserves
        return 0; // Placeholder return, actual implementation needed
    }

    private async recordNewPair(token0: string, token1: string, factory: string, pair: string, fee: number): Promise<void> {
        // Implement the logic to record a new pair in the system
    }

    private async emitNewPairDiscovered(event: {
        token0: string;
        token1: string;
        pair: string;
        dex: string;
        liquidity: number;
        score: number;
    }): Promise<void> {
        // Implement the logic to emit a new pair discovered event
    }

    private async getTokenMetrics(token0: string, token1: string): Promise<{ totalLiquidity: number; holdersCount: number; priceVolatility: number }> {
        // Implement the logic to fetch token metrics
        return { totalLiquidity: 0, holdersCount: 0, priceVolatility: 0 }; // Placeholder return, actual implementation needed
    }

    protected async initializeTokenMonitoring(tokenAddress: string): Promise<void> {
        console.log(`[MONITORING INITIALIZED] for token ${tokenAddress}`);
    }
}

