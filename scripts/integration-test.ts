import { config } from 'dotenv';
import { HardhatRuntimeEnvironment } from 'hardhat/types';
import { HardhatEthersProvider } from '@nomicfoundation/hardhat-ethers/internal/hardhat-ethers-provider';
import { HardhatEthersSigner } from '@nomicfoundation/hardhat-ethers/signers';
import { spawn } from 'child_process';
import { join } from 'path';
import { ethers } from 'ethers';
import TelegramBot from 'node-telegram-bot-api';
import fs from 'fs/promises';
import * as hardhat from 'hardhat';
import { getContractFactory } from '@nomicfoundation/hardhat-ethers/types';
import { writeFileSync } from 'fs';
import { JsonRpcProvider, Wallet, parseEther, formatEther, ZeroAddress } from 'ethers';
import { FlashLoanArbitrage, FlashLoanArbitrage__factory } from '../typechain-types';
import { Contract, ContractTransaction } from 'ethers';
import { BoostOptimizer } from './utils/BoostOptimizer';
import { TaxDataManager } from './utils/TaxDataManager';

const hre = hardhat as unknown as HardhatRuntimeEnvironment & {
    ethers: {
        getSigners: () => Promise<any[]>;
        getContractFactory: (name: string, signer?: any) => Promise<any>;
    };
};

config();

const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const TELEGRAM_CHAT_ID = process.env.TELEGRAM_CHAT_ID;

if (!TELEGRAM_BOT_TOKEN || !TELEGRAM_CHAT_ID) {
    throw new Error('Telegram bot token or chat ID not found in environment variables');
}

const bot = new TelegramBot(TELEGRAM_BOT_TOKEN, { polling: false });

// Contract addresses
const AAVE_POOL_ADDRESSES_PROVIDER = "0x0E02EB705be325407707662C6f6d3466E939f3a0";
const UNISWAP_V3_FACTORY = "0x33128a8fC17869897dcE68Ed026d694621f6FDfD";
const UNISWAP_V3_ROUTER = "0x2626664c2603336E57B271c5C0b26F421741e481";

// Global instances
let contract: FlashLoanArbitrage;

interface SecurityCheck {
    name: string;
    check: () => Promise<boolean>;
    errorMessage: string;
}

// Add colored console logging
const colors = {
    reset: "\x1b[0m",
    bright: "\x1b[1m",
    dim: "\x1b[2m",
    green: "\x1b[32m",
    yellow: "\x1b[33m",
    blue: "\x1b[34m",
    red: "\x1b[31m",
    cyan: "\x1b[36m"
};

function log(type: string, message: string, error: boolean = false) {
    const timestamp = new Date().toISOString();
    const color = error ? colors.red : colors.blue;
    console.log(`${colors.dim}[${timestamp}]${colors.reset} ${color}${type}${colors.reset}: ${message}`);
}

async function sendTelegramMessage(message: string) {
    try {
        log("TELEGRAM", `Sending message: ${message}`);
        await bot.sendMessage(Number(TELEGRAM_CHAT_ID), message);
        log("TELEGRAM", "Message sent successfully");
    } catch (error) {
        log("TELEGRAM", `Failed to send message: ${error}`, true);
    }
}

async function fundContract(signer: Wallet, contractAddress: string) {
    log("FUNDING", `Funding contract ${contractAddress}`);
    const tx = await signer.sendTransaction({
        to: contractAddress,
        value: parseEther("0.1")
    });
    log("FUNDING", `Transaction sent: ${tx.hash}`);
    await tx.wait();
    log("FUNDING", `Funding confirmed. Amount: 0.1 ETH`);
}

async function simulateArbitrageOpportunity(
    flashLoanContract: FlashLoanArbitrage,
    token: string
) {
    // Implementation of opportunity simulation
    console.log(`Simulating arbitrage for token ${token}...`);
}

async function validateEnvironment() {
    log("SETUP", "Validating environment variables and connections");
    
    // Skip private key validation if using hardhat network
    if (process.env.HARDHAT_NETWORK === 'hardhat') {
        log("SETUP", "Using Hardhat network - skipping private key validation");
        return;
    }
    
    const requiredVars = [
        'PRIVATE_KEY',
        'BASE_RPC_URL',
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ];
    
    const missingVars = requiredVars.filter(varName => !process.env[varName]);
    if (missingVars.length > 0) {
        throw new Error(`Missing required environment variables: ${missingVars.join(', ')}`);
    }

    // Validate private key format
    const privateKey = process.env.PRIVATE_KEY;
    if (!/^[0-9a-fA-F]{64}$/.test(privateKey!)) {
        throw new Error('Invalid PRIVATE_KEY format');
    }
}

async function runSecurityChecks(): Promise<boolean> {
    log("SECURITY", "Running security checks");
    const checks: SecurityCheck[] = [
        // ... your security checks ...
    ];

    for (const check of checks) {
        log("SECURITY", `Running check: ${check.name}`);
        try {
            const passed = await check.check();
            if (!passed) {
                log("SECURITY", check.errorMessage, true);
                return false;
            }
            log("SECURITY", `${check.name}: ${colors.green}PASSED${colors.reset}`);
        } catch (error) {
            log("SECURITY", `Check failed: ${error}`, true);
            return false;
        }
    }
    
    log("SECURITY", `${colors.green}All security checks passed${colors.reset}`);
    return true;
}

async function deployContract(): Promise<FlashLoanArbitrage> {
    let signer;
    
    if (process.env.HARDHAT_NETWORK === 'hardhat') {
        const [deployer] = await hre.ethers.getSigners();
        signer = deployer;
    } else {
        const provider = new JsonRpcProvider(process.env.BASE_RPC_URL);
        signer = new Wallet(process.env.PRIVATE_KEY!, provider);
    }
    
    const factory = new FlashLoanArbitrage__factory(signer);
    const flashLoanArbitrage = await factory.deploy(
        "0x33128a8fc17869897dce68ed026d694621f6fdfd", // Uniswap v3 factory on Base
        "0x2626664c2603336e57b271c5c0b26f421741e481", // Uniswap v3 router on Base
        "0x6a6b34eac2a5045d1890ed6a4b0c1e40936d9d08", // SushiSwap router
        "0x678aa4bf4e210cf2166753e054d5b7c31cc7fa86", // PancakeSwap router
        50n,  // minProfitBps (0.5%)
        500n,  // maxSlippageBps (5%)
        "0x4200000000000000000000000000000000000006", // WETH
        "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"  // UNISWAP_QUOTER_V2
    );
    
    await flashLoanArbitrage.waitForDeployment();
    const contractAddress = await flashLoanArbitrage.getAddress();
    console.log('Contract deployed to:', contractAddress);
    await sendTelegramMessage(`🚀 Contract deployed to: ${contractAddress}`);

    // Add initial supported tokens
    console.log('\nChecking and adding supported tokens...');
    const USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913";
    const WETH = "0x4200000000000000000000000000000000000006";
    
    // Check if tokens are supported
    const isUsdcSupported = await flashLoanArbitrage.isTokenSupported(USDC);
    if (!isUsdcSupported) {
        const contract = flashLoanArbitrage.connect(signer) as any;
        await contract.addSupportedToken(USDC);
        console.log('Added USDC as supported token');
        await sendTelegramMessage('✅ Added USDC as supported token');
    }

    const isWethSupported = await flashLoanArbitrage.isTokenSupported(WETH);
    if (!isWethSupported) {
        const contract = flashLoanArbitrage.connect(signer) as any;
        await contract.addSupportedToken(WETH);
        console.log('Added WETH as supported token');
        await sendTelegramMessage('✅ Added WETH as supported token');
    }

    return flashLoanArbitrage;
}

// Initialize services
const boostOptimizer = new BoostOptimizer();
const taxManager = new TaxDataManager();

// Types
interface TaxTransaction {
    // ... existing code ...
}

interface EmergencyCondition {
    type: string;
    threshold: number;
    multiplier: number;
    active: boolean;
}

// Security constants
const SECURITY_CONSTANTS = {
    MAX_GAS_PRICE: BigInt(process.env.GAS_PRICE_LIMIT || '100') * BigInt(1e9),
    SLIPPAGE_TOLERANCE: 0.005, // 0.5%
    MIN_NOTIFICATION_INTERVAL: 60000, // 1 minute
    CRITICAL_SLIPPAGE_THRESHOLD: 0.02, // 2%
    MAX_RETRIES: 3,
    MONITORING_INTERVAL: 15000, // 15 seconds
    MAX_ALERT_HISTORY: 1000
};

interface SecurityState {
    isPaused: boolean;
    lastNotificationTime: number;
    failedAttempts: number;
    isEmergencyShutdown: boolean;
}

class DashboardManager {
    private updateInterval: NodeJS.Timeout;
    private refreshRate: number;
    private client: any;  // Replace 'any' with your actual client type
    private gasCosts: { current: bigint; average: bigint };
    private securityState: SecurityState;
    private alertHistory: Array<{ timestamp: number; message: string }>;

    constructor() {
        this.refreshRate = 5000; // 5 seconds default
        this.gasCosts = { current: 0n, average: 0n };
        this.securityState = {
            isPaused: false,
            lastNotificationTime: 0,
            failedAttempts: 0,
            isEmergencyShutdown: false
        };
        this.alertHistory = [];
        this.initializeSecurity();
        this.updateInterval = setInterval(() => this.fetchUpdates(), this.refreshRate);
    }

    private initializeSecurity(): void {
        // Cleanup on process termination
        process.on('SIGINT', () => this.cleanup());
        process.on('SIGTERM', () => this.cleanup());
    }

    private async cleanup(): Promise<void> {
        clearInterval(this.updateInterval);
        await this.notifyAdmins('System shutdown initiated');
        // Allow time for final notifications
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    private async fetchUpdates(): Promise<void> {
        if (this.securityState.isEmergencyShutdown) {
            console.log('System is in emergency shutdown mode');
            return;
        }
        try {
            await this.initializeGasMonitoring();
            this.securityState.failedAttempts = 0;
        } catch (error) {
            this.securityState.failedAttempts++;
            if (this.securityState.failedAttempts >= SECURITY_CONSTANTS.MAX_RETRIES) {
                await this.initiateEmergencyShutdown('Too many failed update attempts');
            }
            throw error;
        }
    }

    private async initializeGasMonitoring(): Promise<void> {
        try {
            const feeHistory = await this.client.getFeeHistory({
                blockCount: 1,
                rewardPercentiles: [50]
            });
            
            const newGasPrice = feeHistory.baseFeePerGas[0] || 0n;
            if (!await this.validateGasPrice(newGasPrice)) {
                return;
            }

            // Atomic update of gas costs
            this.gasCosts = {
                current: newGasPrice,
                average: this.gasCosts.current
            };

        } catch (error) {
            console.error('Error initializing gas monitoring:', error);
            throw error;
        }
    }

    private async validateGasPrice(gasPrice: bigint): Promise<boolean> {
        if (gasPrice > SECURITY_CONSTANTS.MAX_GAS_PRICE) {
            await this.notifyAdmins(`Gas price ${gasPrice} exceeds limit ${SECURITY_CONSTANTS.MAX_GAS_PRICE}`);
            return false;
        }
        return true;
    }

    private async handleEmergencyConditions(
        tokenAddress: string, 
        emergency: { slippage: number; activeConditions: EmergencyCondition[] }
    ): Promise<void> {
        if (this.securityState.isEmergencyShutdown) {
            throw new Error('System is in emergency shutdown mode');
        }

        try {
            // Validate slippage
            if (Math.abs(emergency.slippage) > SECURITY_CONSTANTS.SLIPPAGE_TOLERANCE) {
                await this.initiateEmergencyShutdown(`Slippage ${emergency.slippage} exceeds tolerance ${SECURITY_CONSTANTS.SLIPPAGE_TOLERANCE}`);
                return;
            }

            await this.recordSlippageEvent({
                type: 'EMERGENCY',
                timestamp: Date.now(),
                tokenAddress,
                slippage: emergency.slippage,
                reason: 'Emergency conditions detected',
                conditions: emergency.activeConditions,
                recommendation: 'Monitor closely'
            });

            if (!this.securityState.isPaused) {
                await this.executeEmergencyProtocol(tokenAddress, emergency);
            }
        } catch (error: unknown) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error('Failed to handle emergency conditions:', error);
            await this.notifyAdmins(`Emergency handling failed for ${tokenAddress}: ${errorMessage}`);
            throw error; // Propagate error for proper handling
        }
    }

    private async recordSlippageEvent(event: { type: string; timestamp: number; tokenAddress: string; slippage: number; reason: string; conditions: EmergencyCondition[]; recommendation: string }): Promise<void> {
        // Implement slippage event recording logic
        console.log('Recording slippage event:', event);
    }

    private async notifyAdmins(message: string): Promise<void> {
        const now = Date.now();
        if (now - this.securityState.lastNotificationTime < SECURITY_CONSTANTS.MIN_NOTIFICATION_INTERVAL) {
            console.log('Rate limited notification:', message);
            return;
        }
        this.securityState.lastNotificationTime = now;
        
        // Add to alert history with cleanup
        this.alertHistory.unshift({ timestamp: now, message });
        if (this.alertHistory.length > SECURITY_CONSTANTS.MAX_ALERT_HISTORY) {
            this.alertHistory.pop();
        }
        
        await sendTelegramMessage(`👨‍💼 Admin Alert: ${message}`);
    }

    private async initiateEmergencyShutdown(reason: string): Promise<void> {
        this.securityState.isEmergencyShutdown = true;
        this.securityState.isPaused = true;
        await this.notifyAdmins(`🚨 EMERGENCY SHUTDOWN: ${reason}`);
        await this.cleanup();
    }

    private async executeEmergencyProtocol(tokenAddress: string, emergency: { slippage: number; activeConditions: EmergencyCondition[] }): Promise<void> {
        if (emergency.slippage > SECURITY_CONSTANTS.CRITICAL_SLIPPAGE_THRESHOLD) {
            await this.initiateEmergencyShutdown(`Critical slippage threshold exceeded: ${emergency.slippage}`);
            return;
        }
        console.log('Executing emergency protocol for token:', tokenAddress, 'Emergency:', emergency);
    }
}

async function main() {
    log("MAIN", "Starting integration test");
    try {
        await validateEnvironment();
        log("MAIN", "Environment validated successfully");

        const securityPassed = await runSecurityChecks();
        if (!securityPassed) {
            log("MAIN", "Security checks failed - aborting", true);
            return;
        }

        console.log('\n1. Deploying FlashLoanArbitrage contract...');
        const [deployer] = await hre.ethers.getSigners();
        console.log('Using deployer address:', deployer.address);

        const FlashLoanArbitrage = await hre.ethers.getContractFactory('FlashLoanArbitrage');
        const flashLoanArbitrage = await FlashLoanArbitrage.deploy(
            "0x33128a8fc17869897dce68ed026d694621f6fdfd", // Uniswap v3 factory on Base
            "0x2626664c2603336e57b271c5c0b26f421741e481", // Uniswap v3 router on Base
            "0x6a6b34eac2a5045d1890ed6a4b0c1e40936d9d08", // SushiSwap router
            "0x678aa4bf4e210cf2166753e054d5b7c31cc7fa86", // PancakeSwap router
            50n,  // minProfitBps (0.5%)
            500n,  // maxSlippageBps (5%)
            "0x4200000000000000000000000000000000000006", // WETH
            "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"  // UNISWAP_QUOTER_V2
        );

        await flashLoanArbitrage.waitForDeployment();
        const contractAddress = await flashLoanArbitrage.getAddress();
        console.log('Contract deployed to:', contractAddress);
        await sendTelegramMessage(`🚀 Contract deployed to: ${contractAddress}`);

        // Add initial supported tokens
        console.log('\nChecking and adding supported tokens...');
        const USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913";
        const WETH = "0x4200000000000000000000000000000000000006";
        
        // Check if tokens are supported
        const isUsdcSupported = await flashLoanArbitrage.isTokenSupported(USDC);
        if (!isUsdcSupported) {
            const contract = flashLoanArbitrage.connect(deployer) as any;
            await contract.addSupportedToken(USDC);
            console.log('Added USDC as supported token');
            await sendTelegramMessage('✅ Added USDC as supported token');
        }

        const isWethSupported = await flashLoanArbitrage.isTokenSupported(WETH);
        if (!isWethSupported) {
            const contract = flashLoanArbitrage.connect(deployer) as any;
            await contract.addSupportedToken(WETH);
            console.log('Added WETH as supported token');
            await sendTelegramMessage('✅ Added WETH as supported token');
        }

        console.log('\n2. Starting web dashboard...');
        const dashboardProcess = spawn('node', ['-r', 'ts-node/register', join(__dirname, '../web/dashboard/server.ts')], {
            stdio: 'pipe',
            env: { 
                ...process.env, 
                PORT: '3002',
                TS_NODE_FILES: 'true',
                CONTRACT_ADDRESS: contractAddress
            }
        });

        dashboardProcess.stdout.on('data', (data) => {
            console.log(`Dashboard: ${data}`);
        });

        dashboardProcess.stderr.on('data', (data) => {
            console.error(`Dashboard Error: ${data}`);
        });

        await sendTelegramMessage('🌐 Web dashboard started at http://localhost:3002');

        console.log('\n3. Starting opportunity monitoring...');
        await sendTelegramMessage('🔍 Starting opportunity monitoring');

        // Initialize provider
        const provider = new JsonRpcProvider(process.env.BASE_RPC_URL);

        // Monitor for opportunities every 30 seconds
        setInterval(async () => {
            try {
                // Common tokens on Base
                const tokens = {
                    USDC: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                    WETH: "0x4200000000000000000000000000000000000006",
                    USDbC: "0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA",
                    DAI: "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb",
                    USDT: "0x4A3A6Dd60A34bB2Aba60D73B4C88315E9CeB6A3D",
                    cbETH: "0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22"
                };

                // DEX Router Addresses
                const dexes = {
                    UniswapV3: "0x2626664c2603336E57B271c5C0b26F421741e481",
                    BaseSwap: "0x327Df1E6de05895d2ab08513aaDD9313Fe505d86",
                    Aerodrome: "0x0000000000A42Be87De91E934B8f3b858851c076",
                    PancakeV3: "0x678Aa4bF4E210cf2166753e054d5b7c31cc7fa86",
                    SushiSwapV3: "0x0389879e0156033202C44BF784ac18fC02edeE27"
                };

                console.log('Checking prices across all DEXes...');
                
                // Create all possible token pairs
                const tokenPairs = [];
                const tokenAddresses = Object.values(tokens);
                for (let i = 0; i < tokenAddresses.length; i++) {
                    for (let j = i + 1; j < tokenAddresses.length; j++) {
                        tokenPairs.push([tokenAddresses[i], tokenAddresses[j]]);
                    }
                }

                // Check each pair across all DEXes
                for (const [token0, token1] of tokenPairs) {
                    try {
                        const prices = new Map<string, bigint>();
                        
                        // Get prices from each DEX
                        for (const [dexName, routerAddress] of Object.entries(dexes)) {
                            try {
                                // Create interface for price query (implementation needed)
                                const price = await flashLoanArbitrage.getPrice(token0, token1, routerAddress);
                                prices.set(dexName, price);
                            } catch (error) {
                                // Skip if pair doesn't exist on this DEX
                                continue;
                            }
                        }

                        // Find best opportunities
                        if (prices.size >= 2) {
                            const priceArray = Array.from(prices.entries());
                            for (let i = 0; i < priceArray.length; i++) {
                                for (let j = i + 1; j < priceArray.length; j++) {
                                    const [dex1, price1] = priceArray[i];
                                    const [dex2, price2] = priceArray[j];
                                    
                                    const diff = (price1 > price2) ? (price1 - price2) : (price2 - price1);
                                    const priceDiff = (diff * 100n) / price1;
                                    
                                    if (priceDiff > 50n) { // 0.5% threshold
                                        const token0Symbol = Object.entries(tokens).find(([_, addr]) => addr === token0)?.[0];
                                        const token1Symbol = Object.entries(tokens).find(([_, addr]) => addr === token1)?.[0];
                                        
                                        // Record the trade in BoostOptimizer
                                        const pairName = `${token0Symbol}-${token1Symbol}`;
                                        const estimatedGas = 250000n; // Typical gas for a swap
                                        const gasPrice = 1000000000n; // 1 gwei
                                        const gasUsed = Number(estimatedGas);
                                        const profit = Number(priceDiff) / 100;
                                        
                                        boostOptimizer.recordTrade(
                                            pairName,
                                            true, // success
                                            profit,
                                            gasUsed,
                                            1.5 // using 1.5x boost for simulation
                                        );
                                        
                                        const message = `💰 Arbitrage opportunity found!\n` +
                                            `Pair: ${pairName}\n` +
                                            `${dex1}: ${formatEther(price1)} ETH\n` +
                                            `${dex2}: ${formatEther(price2)} ETH\n` +
                                            `Price Difference: ${profit}%\n` +
                                            `Gas Used: ${gasUsed}\n` +
                                            `Boost Used: 1.5x`;
                                        
                                        console.log(message);
                                        await sendTelegramMessage(message);
                                    }
                                }
                            }
                        }
                    } catch (error) {
                        console.error(`Error checking pair ${token0}-${token1}:`, error);
                    }
                }
            } catch (error) {
                console.error('Error monitoring opportunities:', error);
                await sendTelegramMessage('⚠️ Error monitoring opportunities: ' + 
                    (error instanceof Error ? error.message : 'Unknown error'));
            }
        }, 30000);

        // Keep the process running
        process.on('SIGINT', async () => {
            console.log('\nShutting down...');
            await sendTelegramMessage('🛑 Bot stopped');
            dashboardProcess.kill();
            process.exit();
        });

        log("MAIN", `${colors.green}Integration test completed successfully${colors.reset}`);
    } catch (error) {
        log("MAIN", `Integration test failed: ${error}`, true);
        await sendTelegramMessage(`❌ Integration test failed: ${error}`);
        process.exit(1);
    }
}

main().catch(async (error) => {
    console.error(error);
    await sendTelegramMessage('❌ Fatal error: ' + error.message);
    process.exit(1);
}); 