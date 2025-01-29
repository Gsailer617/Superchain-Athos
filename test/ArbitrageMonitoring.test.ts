import { expect } from 'chai';
import { TestLogger } from './TestConfig';
import { createPublicClient, http, parseEther, formatEther } from 'viem';
import { baseSepolia } from 'viem/chains';
import { ArbitrageBot } from '../scripts/ArbitrageBot';
import { DexDataProvider } from '../services/DexDataProvider';
import { PairGasOptimizer } from '../scripts/utils/PairGasOptimizer';
import { VolumeAnalyzer } from '../scripts/utils/VolumeAnalyzer';
import fs from 'fs';
import path from 'path';

describe('Live Arbitrage Monitoring (Base Sepolia Testnet)', () => {
    let arbitrageBot: ArbitrageBot;
    let dexDataProvider: DexDataProvider;
    const monitoringInterval = 30000; // 30 seconds
    const logDir = path.join(__dirname, '../logs');
    const dashboardDataFile = path.join(logDir, 'dashboard_data.json');

    // Dashboard data structure
    const dashboardData = {
        startTime: new Date().toISOString(),
        opportunities: [] as any[],
        marketConditions: [] as any[],
        stats: {
            totalOpportunities: 0,
            totalProfit: 0n,
            averageConfidence: 0,
            successfulTrades: 0,
            failedTrades: 0
        },
        lastUpdate: new Date().toISOString()
    };

    before(async () => {
        // Create logs directory if it doesn't exist
        if (!fs.existsSync(logDir)) {
            fs.mkdirSync(logDir, { recursive: true });
        }

        TestLogger.init();
        TestLogger.log('TEST', 'Initializing live arbitrage monitoring on Base Sepolia testnet...');

        // Initialize client with Base Sepolia
        const client = createPublicClient({
            chain: baseSepolia,
            transport: http('https://sepolia.base.org'),
            batch: {
                multicall: true
            }
        });

        // Initialize real services with testnet configuration
        dexDataProvider = new DexDataProvider(client);
        const gasOptimizer = new PairGasOptimizer();
        const volumeAnalyzer = new VolumeAnalyzer();

        // Create ArbitrageBot instance for testnet
        arbitrageBot = new ArbitrageBot(
            client,
            dexDataProvider,
            gasOptimizer,
            volumeAnalyzer
        );

        TestLogger.log('TEST', 'Services initialized successfully on Base Sepolia');
        updateDashboard({ type: 'INIT', data: { status: 'initialized' }});
    });

    it('should monitor testnet arbitrage opportunities continuously', async () => {
        TestLogger.log('TEST', 'Starting continuous arbitrage monitoring on Base Sepolia');

        let monitoringCycles = 0;
        const maxCycles = 10; // Run 10 monitoring cycles

        while (monitoringCycles < maxCycles) {
            TestLogger.log('TEST', `Starting monitoring cycle ${monitoringCycles + 1}`);

            // Monitor for opportunities on testnet
            const opportunities = await arbitrageBot.findArbitrageOpportunities();
            
            TestLogger.log('TEST', `Found ${opportunities.length} potential opportunities on testnet`);

            // Process and log each opportunity
            for (const opp of opportunities) {
                const opportunityData = {
                    timestamp: new Date().toISOString(),
                    path: opp.path.join(' -> '),
                    profit: formatEther(opp.expectedProfit),
                    confidence: opp.confidence,
                    amountIn: formatEther(opp.amountIn),
                    gasEstimate: opp.gasEstimate.toString(),
                    mevRisk: opp.mevRisk
                };

                // Update dashboard data
                updateDashboard({
                    type: 'OPPORTUNITY',
                    data: opportunityData
                });

                TestLogger.log('OPPORTUNITY', 'Found arbitrage opportunity on Base Sepolia', opportunityData);

                // Verify opportunity structure
                expect(opp.path).to.be.an('array').that.is.not.empty;
                expect(opp.expectedProfit).to.be.gt(0n);
                expect(opp.confidence).to.be.within(0, 1);
                expect(opp.amountIn).to.be.gt(0n);
            }

            // Analyze market conditions
            const pools = await dexDataProvider.getPools();
            const marketData = [];

            for (const pool of pools) {
                const volume = await volumeAnalyzer.analyzeVolume(pool.address);
                const poolData = {
                    timestamp: new Date().toISOString(),
                    address: pool.address,
                    token0: pool.token0,
                    token1: pool.token1,
                    volume: volume.toString(),
                    reserve0: formatEther(pool.reserve0),
                    reserve1: formatEther(pool.reserve1)
                };

                marketData.push(poolData);
                TestLogger.log('MARKET', 'Pool Analysis (Base Sepolia)', poolData);
            }

            // Update dashboard with market data
            updateDashboard({
                type: 'MARKET',
                data: { pools: marketData }
            });

            monitoringCycles++;
            await new Promise(resolve => setTimeout(resolve, monitoringInterval));
        }
    });

    function updateDashboard(update: { type: string; data: any }) {
        try {
            // Update dashboard data based on update type
            switch (update.type) {
                case 'OPPORTUNITY':
                    dashboardData.opportunities.push(update.data);
                    dashboardData.stats.totalOpportunities++;
                    dashboardData.stats.totalProfit += BigInt(parseEther(update.data.profit));
                    dashboardData.stats.averageConfidence = 
                        (dashboardData.stats.averageConfidence * (dashboardData.stats.totalOpportunities - 1) + 
                        update.data.confidence) / dashboardData.stats.totalOpportunities;
                    break;
                case 'MARKET':
                    dashboardData.marketConditions.push(update.data);
                    break;
                case 'TRADE_RESULT':
                    if (update.data.success) {
                        dashboardData.stats.successfulTrades++;
                    } else {
                        dashboardData.stats.failedTrades++;
                    }
                    break;
            }

            dashboardData.lastUpdate = new Date().toISOString();

            // Write updated data to file
            fs.writeFileSync(
                dashboardDataFile,
                JSON.stringify({
                    ...dashboardData,
                    stats: {
                        ...dashboardData.stats,
                        totalProfit: dashboardData.stats.totalProfit.toString()
                    }
                }, null, 2)
            );
        } catch (error) {
            console.error('Error updating dashboard:', error);
        }
    }

    after(() => {
        TestLogger.log('TEST', 'Base Sepolia testnet monitoring completed');
        // Final dashboard update
        updateDashboard({
            type: 'COMPLETE',
            data: {
                endTime: new Date().toISOString(),
                summary: dashboardData.stats
            }
        });
    });
}); 