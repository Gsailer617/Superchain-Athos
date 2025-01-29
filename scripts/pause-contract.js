const hre = require("hardhat");

async function main() {
  try {
    const [signer] = await hre.ethers.getSigners();
    console.log("Using account:", signer.address);

    // Get the contract factory and attach to deployed contract
    const FlashLoanArbitrage = await hre.ethers.getContractFactory("FlashLoanArbitrage");
    const contractAddress = "0x50779168d3Cd679C1Baf9dac8c33D2F3dF3b66c4";
    
    console.log("Connecting to contract at address:", contractAddress);
    const contract = FlashLoanArbitrage.attach(contractAddress);
    
    console.log("Attempting to pause contract...");
    const tx = await contract.pause();
    console.log("Waiting for transaction confirmation...");
    await tx.wait();
    
    console.log("Contract successfully paused!");
    console.log("Transaction hash:", tx.hash);
  } catch (error) {
    console.error("Error pausing contract:", error);
    if (error.message.includes("caller is not the owner")) {
      console.log("\nError: You are not the owner of this contract.");
      console.log("Make sure you're using the same wallet that deployed the contract.");
    }
    process.exit(1);
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  }); 