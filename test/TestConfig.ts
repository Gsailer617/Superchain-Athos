import chalk from 'chalk';
import fs from 'fs';
import path from 'path';

export class TestLogger {
    private static logFile = 'test_logs.txt';
    private static isTestMode = true;

    static init() {
        // Clear existing log file
        fs.writeFileSync(this.logFile, '');
        this.log('TEST', 'Starting test run...');
    }

    static log(type: string, message: string, data?: any) {
        const timestamp = new Date().toISOString();
        const logMessage = `[${timestamp}] ${type}: ${message}`;
        
        // Console output with colors
        switch (type) {
            case 'TEST':
                console.log(chalk.blue(logMessage));
                break;
            case 'OPPORTUNITY':
                console.log(chalk.green(logMessage));
                break;
            case 'ERROR':
                console.log(chalk.red(logMessage));
                break;
            default:
                console.log(chalk.gray(logMessage));
        }

        // Add data if present
        if (data) {
            console.log(chalk.yellow('Data:'), data);
        }

        // Write to file
        fs.appendFileSync(this.logFile, logMessage + '\n');
        if (data) {
            fs.appendFileSync(this.logFile, JSON.stringify(data, null, 2) + '\n');
        }
    }

    static getTestMode(): boolean {
        return this.isTestMode;
    }
}

export const TEST_CONFIG = {
    // Test mode settings
    MOCK_PRICES: true,
    MOCK_GAS: true,
    POLLING_INTERVAL: 1000,  // 1 second for faster testing
    
    // Test tokens (Base mainnet addresses)
    TEST_TOKENS: {
        USDC: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        WETH: "0x4200000000000000000000000000000000000006",
        DAI: "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb"
    },

    // Test scenarios
    SCENARIOS: {
        PROFITABLE_TRADE: {
            tokenIn: "USDC",
            tokenOut: "WETH",
            expectedProfit: "0.05",  // 5% profit
            route: ["Uniswap", "Sushiswap"]
        },
        UNPROFITABLE_TRADE: {
            tokenIn: "WETH",
            tokenOut: "DAI",
            expectedProfit: "-0.02", // -2% profit
            route: ["Uniswap", "Balancer"]
        }
    }
}; 