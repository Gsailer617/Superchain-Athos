const hre = require("hardhat");
const { ethers } = require("hardhat");

async function main() {
  try {
    const [deployer] = await hre.ethers.getSigners();
    console.log("Checking transactions for address:", deployer.address);

    // Get the latest block number
    const latestBlock = await deployer.provider.getBlockNumber();
    console.log("Current block:", latestBlock);

    // Look back through the last 1000 blocks or to block 0, whichever is greater
    const fromBlock = Math.max(0, latestBlock - 1000);
    console.log("Scanning blocks from", fromBlock, "to", latestBlock);

    // Get all transactions in this range
    const deployerCode = await deployer.provider.getCode(deployer.address);
    console.log("\nScanning for contract deployments...");

    // Alternative approach: Get the transaction count (nonce)
    const nonce = await deployer.provider.getTransactionCount(deployer.address);
    console.log(`Total transactions sent from this address: ${nonce}`);

    // Let's try to get the Base Sepolia explorer URL for the address
    console.log("\nYou can view all transactions for this address at:");
    console.log(`https://sepolia.basescan.org/address/${deployer.address}`);
    
    console.log("\nTo find your contract:");
    console.log("1. Click the link above to view your address on BaseScan");
    console.log("2. Go to the 'Transactions' tab");
    console.log("3. Look for transactions marked as 'Contract Creation'");
    
  } catch (error) {
    console.error("Error scanning for contract:", error);
    process.exit(1);
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  }); 