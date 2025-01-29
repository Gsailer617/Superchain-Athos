const hre = require("hardhat");
const { ethers } = require("hardhat");

async function validateAddress(address, name) {
  // Allow zero addresses for protocols not yet deployed
  if (address === "0x0000000000000000000000000000000000000000") {
    console.log(`Warning: ${name} using zero address placeholder (protocol not yet deployed)`);
    return;
  }
  
  // For non-zero addresses, validate they are proper addresses
  if (!ethers.isAddress(address)) {
    throw new Error(`Invalid ${name} address: ${address}`);
  }
}

async function main() {
  try {
    console.log("Starting deployment process...");
    
    // Get the deployer's address and balance
    const [deployer] = await hre.ethers.getSigners();
    console.log("Deploying contracts with the account:", deployer.address);
    
    const balance = await deployer.provider.getBalance(deployer.address);
    console.log("Account balance:", hre.ethers.formatEther(balance));

    // Constructor arguments for Base Sepolia
    // Protocol addresses - Base Sepolia
    const AAVE_POOL_ADDRESS_PROVIDER = "0x0000000000000000000000000000000000000000"; // Placeholder until Aave is deployed
    const MOONWELL_COMPTROLLER = "0x0000000000000000000000000000000000000000"; // Placeholder until Moonwell is deployed
    const MORPHO = "0x0000000000000000000000000000000000000000"; // Placeholder until Morpho is deployed
    const BALANCER_VAULT = "0x0000000000000000000000000000000000000000"; // Placeholder until Balancer is deployed
    
    // DEXes - Official Uniswap V3 Base Sepolia Deployments
    const UNISWAP_V3_FACTORY = "0x4752ba5DBc23f44D87826276BF6Fd6b1C372aD24";  // Uniswap V3 Factory on Base Sepolia
    const UNISWAP_V3_ROUTER = "0x94cC0AaC535CCDB3C01d6787D6413C739ae12bc4";  // SwapRouter02 on Base Sepolia
    const UNISWAP_QUOTER_V2 = "0xC5290058841028F1614F3A6F0F5816cAd0df5E27"; // QuoterV2 on Base Sepolia
    
    // Using WETH for other DEX routers since they're not deployed yet
    const SUSHISWAP_ROUTER = "0x4200000000000000000000000000000000000006";
    const PANCAKESWAP_ROUTER = "0x4200000000000000000000000000000000000006";
    
    // Base tokens - Official Base Sepolia addresses
    const WETH = "0x4200000000000000000000000000000000000006"; // Base Sepolia WETH
    
    // Profit and slippage settings
    const MIN_PROFIT_BPS = 50n; // 0.5%
    const MAX_SLIPPAGE_BPS = 500n; // 5%

    console.log("\nValidating contract addresses...");
    
    // Validate critical addresses
    await validateAddress(AAVE_POOL_ADDRESS_PROVIDER, "AAVE_POOL_ADDRESS_PROVIDER");
    await validateAddress(MOONWELL_COMPTROLLER, "MOONWELL_COMPTROLLER");
    await validateAddress(MORPHO, "MORPHO");
    await validateAddress(BALANCER_VAULT, "BALANCER_VAULT");
    await validateAddress(UNISWAP_QUOTER_V2, "UNISWAP_QUOTER_V2");
    await validateAddress(UNISWAP_V3_FACTORY, "UNISWAP_V3_FACTORY");
    await validateAddress(UNISWAP_V3_ROUTER, "UNISWAP_V3_ROUTER");
    await validateAddress(SUSHISWAP_ROUTER, "SUSHISWAP_ROUTER");
    await validateAddress(PANCAKESWAP_ROUTER, "PANCAKESWAP_ROUTER");
    await validateAddress(WETH, "WETH");
    
    // Deployment parameters
    const params = [
      UNISWAP_V3_FACTORY,
      UNISWAP_V3_ROUTER,
      SUSHISWAP_ROUTER,
      PANCAKESWAP_ROUTER,
      MIN_PROFIT_BPS,
      MAX_SLIPPAGE_BPS,
      WETH,
      UNISWAP_QUOTER_V2
    ];

    console.log("\nDeployment Parameters:");
    console.log("UNISWAP_QUOTER_V2:", UNISWAP_QUOTER_V2);
    console.log("UNISWAP_V3_FACTORY:", UNISWAP_V3_FACTORY);
    console.log("UNISWAP_V3_ROUTER:", UNISWAP_V3_ROUTER);
    console.log("SUSHISWAP_ROUTER:", SUSHISWAP_ROUTER);
    console.log("PANCAKESWAP_ROUTER:", PANCAKESWAP_ROUTER);
    console.log("MIN_PROFIT_BPS:", MIN_PROFIT_BPS.toString());
    console.log("MAX_SLIPPAGE_BPS:", MAX_SLIPPAGE_BPS.toString());
    console.log("WETH:", WETH);

    console.log("\nDeploying FlashLoanArbitrage contract...");
    const FlashLoanArbitrage = await hre.ethers.getContractFactory("FlashLoanArbitrage");
    
    console.log("Creating deployment transaction...");
    const flashLoanArbitrage = await FlashLoanArbitrage.deploy(
        ...params
    );

    console.log("Waiting for deployment transaction...");
    await flashLoanArbitrage.waitForDeployment();

    const deployedAddress = await flashLoanArbitrage.getAddress();
    console.log("\nFlashLoanArbitrage deployed to:", deployedAddress);
    
    // Verify the contract on Basescan
    console.log("\nVerifying contract on Basescan...");
    try {
      await hre.run("verify:verify", {
        address: deployedAddress,
        constructorArguments: params,
      });
      console.log("Contract verified successfully!");
    } catch (error) {
      console.log("Verification failed:", error.message);
    }
  } catch (error) {
    console.error("\nDeployment failed with error:", error);
    throw error;
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  }); 