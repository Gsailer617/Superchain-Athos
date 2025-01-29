import { createPublicClient, http, PublicClient, Chain, Transport, HttpTransport } from 'viem';
import { base } from 'viem/chains';
import { ArbitrageBot } from './ArbitrageBot';

async function main() {
    console.log('🚀 Starting Arbitrage Bot...\n');

    // Initialize the client with specific configuration
    const client = createPublicClient({
        chain: base,
        transport: http(),
        batch: {
            multicall: true
        },
        pollingInterval: 1000,
        name: 'Arbitrage Bot Client',
        cacheTime: 4_000
    }) as PublicClient;

    // Create and initialize the bot
    const bot = new ArbitrageBot(client);

    // Start monitoring for opportunities
    console.log('👀 Monitoring for arbitrage opportunities...\n');
    
    while (true) {
        try {
            // Find opportunities
            const opportunities = await bot.findArbitrageOpportunities();
            
            // Execute viable opportunities
            for (const opportunity of opportunities) {
                await bot.executeArbitrageOpportunity(opportunity);
            }
            
            // Wait before next iteration
            await new Promise(resolve => setTimeout(resolve, 5000));
        } catch (error) {
            console.error('❌ Error in main loop:', error);
            // Wait before retrying
            await new Promise(resolve => setTimeout(resolve, 10000));
        }
    }
}

main().catch(console.error); 