const { ethers } = require("hardhat");
const { BigNumber } = require("ethers");

class ArbitrageBot {
    constructor(contract, wallet) {
        this.contract = contract;
        this.wallet = wallet;
        this.isRunning = false;
        this.minProfitBps = 50; // 0.5%
        
        // Token pairs to monitor
        this.tokenPairs = [
            {
                token0: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913", // USDC
                token1: "0x4200000000000000000000000000000000000006", // WETH
                minAmount: ethers.utils.parseUnits("1000", 6)  // 1000 USDC
            },
            // Add more popular Base pairs
            {
                token0: "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb", // DAI
                token1: "0x4200000000000000000000000000000000000006", // WETH
                minAmount: ethers.utils.parseUnits("1000", 18)  // 1000 DAI
            },
            {
                token0: "0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22", // cbETH
                token1: "0x4200000000000000000000000000000000000006", // WETH
                minAmount: ethers.utils.parseUnits("1", 18)  // 1 cbETH
            }
        ];

        // Flash loan sources with their priorities
        this.flashLoanSources = [
            {
                name: "Moonwell",
                execute: async (pair) => await this.contract.executeMoonwellFlashLoan(pair.token0, pair.minAmount),
                priority: 1
            },
            {
                name: "Morpho",
                execute: async (pair) => await this.contract.executeMorphoFlashLoan(pair.token0, pair.minAmount),
                priority: 2
            },
            {
                name: "Aave",
                execute: async (pair) => await this.contract.executeAaveFlashLoan(pair.token0, pair.minAmount),
                priority: 3
            }
        ];
    }

    async start() {
        if (this.isRunning) return;
        this.isRunning = true;
        console.log("Starting arbitrage bot...");
        
        // Start monitoring loop
        while (this.isRunning) {
            try {
                await this.checkOpportunities();
            } catch (error) {
                console.error("Error in monitoring loop:", error);
            }
            
            // Wait before next check
            await new Promise(resolve => setTimeout(resolve, 1000)); // 1 second delay
        }
    }

    stop() {
        this.isRunning = false;
        console.log("Stopping arbitrage bot...");
    }

    async checkOpportunities() {
        for (const pair of this.tokenPairs) {
            try {
                const [prices, bestSourceDex, bestTargetDex, maxSpread] = await this.contract.checkPrices(
                    pair.token0,
                    pair.token1
                );

                // Check if spread is profitable
                if (maxSpread > this.minProfitBps) {
                    console.log(`Found profitable opportunity:`);
                    console.log(`- Token pair: ${pair.token0} / ${pair.token1}`);
                    console.log(`- Spread: ${maxSpread / 100}%`);
                    console.log(`- Source DEX: ${bestSourceDex}`);
                    console.log(`- Target DEX: ${bestTargetDex}`);

                    // Execute the arbitrage
                    await this.executeArbitrage(pair, bestSourceDex, prices);
                }
            } catch (error) {
                console.error(`Error checking pair ${pair.token0}/${pair.token1}:`, error);
            }
        }
    }

    async executeArbitrage(pair, sourceDex, prices) {
        try {
            // Sort flash loan sources by priority
            const sortedSources = [...this.flashLoanSources].sort((a, b) => a.priority - b.priority);

            // Try each flash loan source in priority order
            for (const source of sortedSources) {
                try {
                    console.log(`Attempting flash loan from ${source.name}...`);
                    const gasPrice = await this.wallet.provider.getGasPrice();
                    
                    // Estimate gas cost
                    const gasEstimate = await source.execute(pair).estimateGas();
                    const gasCost = gasEstimate.mul(gasPrice);
                    
                    // Calculate potential profit (simplified)
                    const profitInWei = prices[sourceDex].sub(prices[bestTargetDex]);
                    
                    // Only execute if profitable after gas costs
                    if (profitInWei.gt(gasCost)) {
                        const tx = await source.execute(pair, {
                            gasLimit: gasEstimate.mul(120).div(100), // Add 20% buffer
                            gasPrice: gasPrice
                        });
                        await tx.wait();
                        console.log(`Arbitrage executed successfully using ${source.name}`);
                        return;
                    } else {
                        console.log(`Skipping ${source.name} - not profitable after gas costs`);
                    }
                } catch (error) {
                    console.error(`Failed to execute flash loan from ${source.name}:`, error);
                }
            }
        } catch (error) {
            console.error("Failed to execute arbitrage:", error);
        }
    }
}

async function main() {
    // Get the deployed contract
    const FlashLoanArbitrage = await ethers.getContractFactory("FlashLoanArbitrage");
    const contract = await FlashLoanArbitrage.attach("YOUR_DEPLOYED_CONTRACT_ADDRESS");
    
    // Get the signer
    const [wallet] = await ethers.getSigners();
    
    // Create and start the bot
    const bot = new ArbitrageBot(contract, wallet);
    await bot.start();
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    }); 