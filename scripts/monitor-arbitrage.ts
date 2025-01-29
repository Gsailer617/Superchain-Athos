import { createPublicClient, http, formatEther, formatUnits } from 'viem';
import { base } from 'viem/chains';
import { ArbitrageBot } from './ArbitrageBot';
import chalk from 'chalk';
import Table from 'cli-table3';
import { ArbitrageOpportunity } from './types/dex';
import { PublicClient } from 'viem';
import TelegramBot from 'node-telegram-bot-api';
import dotenv from 'dotenv';

dotenv.config();

const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const TELEGRAM_CHAT_ID = Number(process.env.TELEGRAM_CHAT_ID);

if (!TELEGRAM_BOT_TOKEN || !TELEGRAM_CHAT_ID || isNaN(TELEGRAM_CHAT_ID)) {
    throw new Error('TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (must be a number) must be set in .env');
}

const bot = new TelegramBot(TELEGRAM_BOT_TOKEN, { polling: false });

async function sendTelegramMessage(message: string) {
    try {
        await bot.sendMessage(TELEGRAM_CHAT_ID, message, { parse_mode: 'HTML' });
    } catch (error) {
        console.error('Failed to send Telegram message:', error);
    }
}

async function monitorArbitrageOpportunities() {
    try {
        // Initialize the public client
        const client = createPublicClient({
            chain: base,
            transport: http(),
            batch: {
                multicall: true
            }
        }) as PublicClient;

        // Initialize the arbitrage bot
        const bot = new ArbitrageBot(client);

        // Start monitoring for opportunities
        console.log(chalk.blue('Starting arbitrage monitoring...'));
        await sendTelegramMessage('🚀 <b>Starting arbitrage monitoring</b>');
        console.log(chalk.gray('Press Ctrl+C to stop\n'));

        // Create tables for displaying information
        const opportunityTable = new Table({
            head: ['Route', 'Expected Profit', 'Confidence', 'Amount In', 'Gas Cost'].map(h => chalk.cyan(h)),
            colWidths: [30, 20, 15, 20, 15]
        });

        const statsTable = new Table({
            head: ['Metric', 'Value'].map(h => chalk.yellow(h)),
            colWidths: [20, 30]
        });

        let scanCount = 0;
        let lastBlockNumber = 0n;
        let lastNotificationTime = 0;

        // Monitor for opportunities
        while (true) {
            try {
                console.clear();
                scanCount++;
                
                // Display basic stats
                statsTable.length = 0;
                const block = await client.getBlockNumber();
                const stats = [
                    ['Scans Completed', scanCount],
                    ['Current Block', block.toString()],
                    ['Block Change', block > lastBlockNumber ? '✅ New Block' : '⏳ Same Block'],
                    ['Last Update', new Date().toLocaleTimeString()]
                ];
                
                statsTable.push(...stats);
                console.log(chalk.yellow('\nMonitoring Stats:'));
                console.log(statsTable.toString());

                // Send periodic status update to Telegram (every 5 minutes)
                const now = Date.now();
                if (now - lastNotificationTime > 300000) {
                    const statusMsg = `📊 <b>Status Update</b>\n\n` +
                        `Scans: ${scanCount}\n` +
                        `Block: ${block.toString()}\n` +
                        `Time: ${new Date().toLocaleTimeString()}`;
                    await sendTelegramMessage(statusMsg);
                    lastNotificationTime = now;
                }
                
                // Get and display opportunities
                const opportunities = await bot.findArbitrageOpportunities();
                opportunityTable.length = 0;

                if (opportunities.length > 0) {
                    console.log(chalk.green('\nFound Opportunities:'));
                    let telegramMsg = '💰 <b>Found Arbitrage Opportunities</b>\n\n';
                    
                    opportunities.forEach(opp => {
                        const route = opp.route.map(r => `${r.dex}`).join(' -> ');
                        const profit = formatEther(opp.expectedProfit);
                        const confidence = (opp.confidence * 100).toFixed(1);
                        const amountIn = formatEther(opp.amountIn);
                        const gasEstimate = formatUnits(opp.gasEstimate || 0n, 9);

                        opportunityTable.push([
                            route,
                            `$${profit}`,
                            `${confidence}%`,
                            `$${amountIn}`,
                            `${gasEstimate} GWEI`
                        ]);

                        telegramMsg += `🔄 <b>Route:</b> ${route}\n` +
                            `💵 Profit: $${profit}\n` +
                            `🎯 Confidence: ${confidence}%\n` +
                            `💰 Amount: $${amountIn}\n` +
                            `⛽ Gas: ${gasEstimate} GWEI\n\n`;
                    });

                    console.log(opportunityTable.toString());
                    await sendTelegramMessage(telegramMsg);
                } else {
                    console.log(chalk.gray('\nNo profitable opportunities found in this scan.'));
                }

                if (block > lastBlockNumber) {
                    await sendTelegramMessage(`🆕 New block detected: ${block.toString()}`);
                }

                lastBlockNumber = block;
                // Wait before checking again
                await new Promise(resolve => setTimeout(resolve, 1000));
            } catch (error) {
                console.error(chalk.red('\nError monitoring opportunities:'), error);
                await sendTelegramMessage(`❌ <b>Error:</b>\n${error instanceof Error ? error.message : 'Unknown error occurred'}`);
                await new Promise(resolve => setTimeout(resolve, 5000));
            }
        }
    } catch (error) {
        console.error(chalk.red('Fatal error:'), error);
        await sendTelegramMessage(`🚨 <b>Fatal Error:</b>\n${error instanceof Error ? error.message : 'Unknown error occurred'}`);
        process.exit(1);
    }
}

// Start monitoring
monitorArbitrageOpportunities(); 