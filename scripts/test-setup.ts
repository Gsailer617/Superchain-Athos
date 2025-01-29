import { ethers } from "hardhat";
import "@nomicfoundation/hardhat-ethers";
import * as dotenv from "dotenv";

dotenv.config();

async function main() {
  try {
    console.log("Starting test setup...");
    
    // Get the deployer's address and balance
    const [deployer] = await ethers.getSigners();
    console.log("Using account:", deployer.address);
    
    const balance = await deployer.provider.getBalance(deployer.address);
    console.log("Account balance:", ethers.formatEther(balance));

    // Get contract instance
    const flashLoanArbitrage = await ethers.getContractAt(
      "FlashLoanArbitrage",
      process.env.CONTRACT_ADDRESS || ""
    );
    console.log("Contract address:", await flashLoanArbitrage.getAddress());

    // Whitelist test tokens
    const tokens = {
      USDC: process.env.USDC || "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
      DAI: process.env.DAI || "0x7D691e6b03b46B5A5769299fC9a32EaC690B7abc",
      WETH: process.env.WETH || "0x4200000000000000000000000000000000000006"
    };

    console.log("\nWhitelisting tokens...");
    for (const [symbol, address] of Object.entries(tokens)) {
      console.log(`Whitelisting ${symbol} at ${address}...`);
      const tx = await flashLoanArbitrage.whitelistToken(address, true);
      await tx.wait();
      console.log(`${symbol} whitelisted successfully`);
    }

    // Verify contract parameters
    const minProfitBps = await flashLoanArbitrage.minProfitBps();
    const maxSlippageBps = await flashLoanArbitrage.maxSlippageBps();
    
    console.log("\nContract Parameters:");
    console.log("Min Profit BPS:", minProfitBps);
    console.log("Max Slippage BPS:", maxSlippageBps);

    console.log("\nTest setup completed successfully!");
  } catch (error) {
    console.error("Error during test setup:", error);
    process.exit(1);
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  }); 
