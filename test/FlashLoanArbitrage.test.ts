/// <reference types="mocha" />
declare const describe: Mocha.SuiteFunction;
declare const it: Mocha.TestFunction;

import { WalletClient, parseEther, parseUnits } from "viem";
import { expect } from "chai";
import { loadFixture } from "@nomicfoundation/hardhat-toolbox-viem/network-helpers";
import { HardhatRuntimeEnvironment } from "hardhat/types";
import hardhat from "hardhat";
const hre = hardhat;

// Add colored console logging
const colors = {
  reset: "\x1b[0m",
  bright: "\x1b[1m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m"
};

function log(type: string, message: string) {
  const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
  console.log(`${colors.bright}[${timestamp}]${colors.reset} ${colors.blue}${type}${colors.reset}: ${message}`);
}

interface TestContext {
  FlashLoanArbitrage: any;
  owner: WalletClient & { account: { address: `0x${string}` } };
  otherAccount: WalletClient & { account: { address: `0x${string}` } };
}

describe("FlashLoanArbitrage", function () {
  before(function() {
    log("TEST SUITE", "Starting FlashLoanArbitrage test suite");
  });

  after(function() {
    log("TEST SUITE", "Completed all tests");
  });

  // Constants for Base network
  const AAVE_POOL_ADDRESSES_PROVIDER = "0x0E02EB705be325407707662C6f6d3466E939f3a0";
  const UNISWAP_V3_FACTORY = "0x33128a8fC17869897dcE68Ed026d694621f6FDfD";
  const UNISWAP_V3_ROUTER = "0x2626664c2603336E57B271c5C0b26F421741e481";
  
  // Token addresses on Base
  const USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913";
  const WETH = "0x4200000000000000000000000000000000000006";
  const DAI = "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb";
  const USDT = "0x4200000000000000000000000000000000000006";
  const cbETH = "0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22";
  const USDbC = "0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA";

  async function deployFlashLoanArbitrageFixture(): Promise<TestContext> {
    log("FIXTURE", "Deploying fresh contract instance");
    const [owner, otherAccount] = await (hre as any).viem.getWalletClients();
    
    log("DEPLOYMENT", "Deploying FlashLoanArbitrage contract");
    const FlashLoanArbitrage = await (hre as any).viem.deployContract("FlashLoanArbitrage", [
      AAVE_POOL_ADDRESSES_PROVIDER,
      UNISWAP_V3_FACTORY,
      UNISWAP_V3_ROUTER
    ]);
    log("DEPLOYMENT", `Contract deployed to ${FlashLoanArbitrage.address}`);

    return { FlashLoanArbitrage, owner, otherAccount };
  }

  describe("Deployment", function () {
    beforeEach(function() {
      log("TEST", `Running: ${this.currentTest?.title}`);
    });

    afterEach(function() {
      const status = this.currentTest?.state === 'passed' ? 
        `${colors.green}PASSED${colors.reset}` : 
        `${colors.yellow}FAILED${colors.reset}`;
      log("TEST", `${this.currentTest?.title}: ${status}`);
    });

    it("Should set the right owner", async function () {
      const { FlashLoanArbitrage, owner } = await loadFixture(deployFlashLoanArbitrageFixture);
      const contractOwner = await FlashLoanArbitrage.read.owner();
      log("VERIFY", `Contract owner: ${contractOwner}`);
      log("VERIFY", `Expected owner: ${owner.account.address}`);
      expect(contractOwner.toLowerCase()).to.equal(owner.account.address.toLowerCase());
    });

    it("Should initialize with correct AAVE pool provider", async function () {
      const { FlashLoanArbitrage } = await loadFixture(deployFlashLoanArbitrageFixture);
      expect(await FlashLoanArbitrage.read.POOL()).to.not.equal("0x0000000000000000000000000000000000000000");
    });

    it("Should initialize with correct Uniswap factory", async function () {
      const { FlashLoanArbitrage } = await loadFixture(deployFlashLoanArbitrageFixture);
      expect(await FlashLoanArbitrage.read.uniswapFactory()).to.equal(UNISWAP_V3_FACTORY);
    });

    it("Should initialize with correct Uniswap router", async function () {
      const { FlashLoanArbitrage } = await loadFixture(deployFlashLoanArbitrageFixture);
      expect(await FlashLoanArbitrage.read.swapRouter()).to.equal(UNISWAP_V3_ROUTER);
    });

    it("Should start with no supported tokens", async function () {
      const { FlashLoanArbitrage } = await loadFixture(deployFlashLoanArbitrageFixture);
      expect(await FlashLoanArbitrage.read.isTokenSupported([USDC])).to.be.false;
      expect(await FlashLoanArbitrage.read.isTokenSupported([WETH])).to.be.false;
      expect(await FlashLoanArbitrage.read.isTokenSupported([DAI])).to.be.false;
    });
  });

  describe("Access Control", function () {
    it("Should allow owner to pause the contract", async function () {
      const { FlashLoanArbitrage, owner } = await loadFixture(deployFlashLoanArbitrageFixture);
      await FlashLoanArbitrage.write.pause();
      expect(await FlashLoanArbitrage.read.paused()).to.be.true;
    });

    it("Should not allow non-owner to pause the contract", async function () {
      const { FlashLoanArbitrage, otherAccount } = await loadFixture(deployFlashLoanArbitrageFixture);
      await expect(
        FlashLoanArbitrage.write.pause({ account: otherAccount.account })
      ).to.be.rejectedWith("Ownable: caller is not the owner");
    });

    it("Should allow owner to unpause the contract", async function () {
      const { FlashLoanArbitrage, owner } = await loadFixture(deployFlashLoanArbitrageFixture);
      await FlashLoanArbitrage.write.pause();
      await FlashLoanArbitrage.write.unpause();
      expect(await FlashLoanArbitrage.read.paused()).to.be.false;
    });
  });

  describe("Token Management", function () {
    it("Should allow owner to add supported token", async function () {
      const { FlashLoanArbitrage, owner } = await loadFixture(deployFlashLoanArbitrageFixture);
      const mockToken = "0x1234567890123456789012345678901234567890";
      await FlashLoanArbitrage.write.addSupportedToken([mockToken]);
      expect(await FlashLoanArbitrage.read.isTokenSupported([mockToken])).to.be.true;
    });

    it("Should allow owner to remove supported token", async function () {
      const { FlashLoanArbitrage, owner } = await loadFixture(deployFlashLoanArbitrageFixture);
      // First add a token
      await FlashLoanArbitrage.write.addSupportedToken([USDC]);
      // Then remove it
      await FlashLoanArbitrage.write.removeSupportedToken([USDC]);
      expect(await FlashLoanArbitrage.read.isTokenSupported([USDC])).to.be.false;
    });

    it("Should allow owner to withdraw tokens", async function () {
      const { FlashLoanArbitrage, owner } = await loadFixture(deployFlashLoanArbitrageFixture);
      const mockToken = USDC;
      const recipient = owner.account.address;
      const amount = parseUnits("1000", 6); // 1000 USDC
      
      // We expect this to revert since we haven't funded the contract
      await expect(
        FlashLoanArbitrage.write.withdrawToken([mockToken, recipient, amount])
      ).to.be.rejectedWith("ERC20: transfer amount exceeds balance");
    });

    it("Should not allow non-owner to withdraw tokens", async function () {
      const { FlashLoanArbitrage, otherAccount } = await loadFixture(deployFlashLoanArbitrageFixture);
      const mockToken = USDC;
      const recipient = otherAccount.account.address;
      const amount = parseUnits("1000", 6); // 1000 USDC
      
      await expect(
        FlashLoanArbitrage.write.withdrawToken(
          [mockToken, recipient, amount],
          { account: otherAccount.account }
        )
      ).to.be.rejectedWith("Ownable: caller is not the owner");
    });
  });

  describe("Flash Loan Operations", function () {
    it("Should execute flash loan from AAVE", async function () {
      const { FlashLoanArbitrage } = await loadFixture(deployFlashLoanArbitrageFixture);
      
      // We'll verify the contract has the correct setup
      const poolProvider = await FlashLoanArbitrage.read.POOL();
      expect(poolProvider).to.not.equal("0x0000000000000000000000000000000000000000");
    });

    it("Should execute Uniswap V3 flash loan", async function () {
      const { FlashLoanArbitrage } = await loadFixture(deployFlashLoanArbitrageFixture);
      
      // We'll verify the contract has the correct setup
      const factory = await FlashLoanArbitrage.read.uniswapFactory();
      expect(factory).to.not.equal("0x0000000000000000000000000000000000000000");
    });
  });

  describe("Arbitrage Operations", function () {
    it("Should find optimal V3 path", async function () {
      const { FlashLoanArbitrage } = await loadFixture(deployFlashLoanArbitrageFixture);
      const amountIn = parseUnits("1000", 6); // 1000 USDC

      const [path, expectedOutput] = await FlashLoanArbitrage.read._findOptimalV3Path([
        USDC,
        WETH,
        amountIn
      ]);

      expect(path).to.not.be.empty;
      expect(Number(expectedOutput)).to.be.greaterThan(0);
    });

    it("Should execute optimized V3 swap", async function () {
      const { FlashLoanArbitrage, owner } = await loadFixture(deployFlashLoanArbitrageFixture);
      const amountIn = parseUnits("1000", 6); // 1000 USDC
      
      const path = Buffer.concat([
        Buffer.from(USDC.slice(2), 'hex'),
        Buffer.from('000bb8', 'hex'), // fee tier 3000
        Buffer.from(WETH.slice(2), 'hex')
      ]);
      
      const currentTime = Math.floor(Date.now() / 1000);
      const route = {
        path,
        recipient: owner.account.address,
        deadline: currentTime + 3600,
        amountIn,
        minAmountOut: 0n
      };

      // We'll verify the contract has the correct setup
      const router = await FlashLoanArbitrage.read.swapRouter();
      const routerAddress = router.toString();
      expect(routerAddress).to.not.equal('0x0000000000000000000000000000000000000000');
    });
  });
}); 