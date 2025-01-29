import { ethers } from 'ethers';
import { Provider } from '@ethersproject/providers';
import { Signer } from '@ethersproject/abstract-signer';
import { Contract } from '@ethersproject/contracts';
import { FlashLoanProvider, FlashLoanParams, AerodromePool, DEXType } from '../types/dex';
import { PublicClient } from 'viem';

// ABI fragments for flash loan contracts
const MORPHO_FLASH_ABI = [
    'function flashLoan(address receiver, address[] calldata tokens, uint256[] calldata amounts, bytes calldata data) external returns (bool)',
    'function maxFlashLoan(address token) external view returns (uint256)'
];

const AAVE_POOL_ABI = [
    'function flashLoan(address receiverAddress, address[] calldata assets, uint256[] calldata amounts, uint256[] calldata modes, address onBehalfOf, bytes calldata params, uint16 referralCode) external returns (bool)',
    'function FLASHLOAN_PREMIUM_TOTAL() external view returns (uint128)',
    'function getReserveData(address asset) external view returns (tuple(uint256 unbacked, uint256 accruedToTreasury, uint256 totalAToken, uint256 totalStableDebt, uint256 totalVariableDebt, uint256 liquidityRate, uint256 variableBorrowRate, uint256 stableBorrowRate, uint256 averageStableBorrowRate, uint256 liquidityIndex, uint256 variableBorrowIndex, uint40 lastUpdateTimestamp))'
];

const MOONWELL_FLASH_ABI = [
    'function flashLoan(address receiver, address[] calldata tokens, uint256[] calldata amounts, uint256[] calldata maxBorrowRates, bytes calldata data) external returns (bool)',
    'function getFlashLoanFee(address token) external view returns (uint256)',
    'function getMarketData(address token) external view returns (tuple(uint256 exchangeRate, uint256 supplyRate, uint256 borrowRate, uint256 totalSupply, uint256 totalBorrows, uint256 totalReserves, uint256 totalCash))'
];

const BASESWAP_FLASH_ABI = [
    'function flash(address recipient, uint256 amount0, uint256 amount1, bytes calldata data) external',
    'function swap(uint256 amount0Out, uint256 amount1Out, address to, bytes calldata data) external'
];

interface Pool {
    address: `0x${string}`;
    token0: `0x${string}`;
    token1: `0x${string}`;
    reserve0: bigint;
    reserve1: bigint;
    fee: number;
}

export interface DexPriceData {
    price: number;
    liquidity: bigint;
    volume24h: bigint;
    lastUpdate: number;
}

export interface DexPoolData {
    address: string;
    token0: string;
    token1: string;
    reserve0: bigint;
    reserve1: bigint;
    fee: number;
    isStable?: boolean;
    provider?: string;
}

export class DexDataProvider {
    private readonly client: PublicClient;
    private provider: Provider | Signer;
    private poolCache: Map<string, DexPoolData>;
    private priceCache: Map<string, DexPriceData>;
    private readonly CACHE_DURATION = 30000; // 30 seconds
    
    // Flash loan contract addresses on Base
    private readonly MORPHO_FLASH_ADDRESS = '0x33333333333333333333333333333333333333333'; // Replace with actual address
    private readonly AAVE_POOL_ADDRESS = '0x44444444444444444444444444444444444444444'; // Replace with actual address
    private readonly MOONWELL_FLASH_ADDRESS = '0x55555555555555555555555555555555555555555'; // Replace with actual address

    constructor(client: PublicClient, provider: Provider | Signer) {
        this.client = client;
        this.provider = provider;
        this.poolCache = new Map();
        this.priceCache = new Map();
    }

    async getAllPools(dexType: DEXType): Promise<Pool[]> {
        // Implementation
        return [];
    }

    async getPoolData(dexType: DEXType, address: `0x${string}`): Promise<Pool | null> {
        // Implementation
        return null;
    }

    async getPriceData(dexType: DEXType, tokenAddress: string): Promise<DexPriceData> {
        const cacheKey = `${dexType}-${tokenAddress}`;
        const cachedData = this.priceCache.get(cacheKey);
        
        if (cachedData && Date.now() - cachedData.lastUpdate < this.CACHE_DURATION) {
            return cachedData;
        }

        // Implementation will vary based on DEX type
        const priceData = await this.fetchPriceData(dexType, tokenAddress);
        this.priceCache.set(cacheKey, priceData);
        
        return priceData;
    }

    private async fetchAerodromePoolData(poolAddress: string): Promise<DexPoolData> {
        const poolContract = new Contract(
            poolAddress,
            [
                'function token0() view returns (address)',
                'function token1() view returns (address)',
                'function getReserves() view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast)',
                'function stable() view returns (bool)',
                'function fees() view returns (address)',
                'function factory() view returns (address)'
            ],
            this.provider
        );

        const [token0, token1, reserves, isStable, fees, factory] = await Promise.all([
            poolContract.token0(),
            poolContract.token1(),
            poolContract.getReserves(),
            poolContract.stable(),
            poolContract.fees(),
            poolContract.factory()
        ]);

        return {
            address: poolAddress,
            token0,
            token1,
            reserve0: reserves[0],
            reserve1: reserves[1],
            fee: isStable ? 0.0004 : 0.0025, // 0.04% for stable, 0.25% for volatile
            isStable,
            provider: 'Aerodrome'
        };
    }

    private async fetchStandardPoolData(poolAddress: string): Promise<DexPoolData> {
        const poolContract = new Contract(
            poolAddress,
            [
                'function token0() view returns (address)',
                'function token1() view returns (address)',
                'function getReserves() view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast)',
                'function fee() view returns (uint24)'
            ],
            this.provider
        );

        const [token0, token1, reserves, fee] = await Promise.all([
            poolContract.token0(),
            poolContract.token1(),
            poolContract.getReserves(),
            poolContract.fee().catch(() => 3000) // Default to 0.3% if fee function doesn't exist
        ]);

        return {
            address: poolAddress,
            token0,
            token1,
            reserve0: reserves[0],
            reserve1: reserves[1],
            fee: fee / 10000 // Convert from basis points to percentage
        };
    }

    private async fetchPriceData(dexType: DEXType, tokenAddress: string): Promise<DexPriceData> {
        // This is a placeholder implementation
        // Actual implementation would fetch real price data from the DEX
        return {
            price: 0,
            liquidity: 0n,
            volume24h: 0n,
            lastUpdate: Date.now()
        };
    }

    async executeFlashLoan(params: FlashLoanParams): Promise<boolean> {
        this.validateFlashLoanParams(params);

        try {
            switch(params.provider) {
                case FlashLoanProvider.MORPHO:
                    return await this.executeMorphoFlashLoan(params);
                case FlashLoanProvider.AAVE:
                    return await this.executeAaveFlashLoan(params);
                case FlashLoanProvider.MOONWELL:
                    return await this.executeMoonwellFlashLoan(params);
                case FlashLoanProvider.BASESWAP:
                    return await this.executeBaseSwapFlashLoan(params);
                default:
                    throw new Error('Unsupported flash loan provider');
            }
        } catch (error: unknown) {
            if (error instanceof Error) {
                console.error(`Flash loan execution failed: ${error.message}`);
                throw new Error(`Flash loan execution failed: ${error.message}`);
            }
            throw new Error('Flash loan execution failed with unknown error');
        }
    }

    private validateFlashLoanParams(params: FlashLoanParams): void {
        if (!params.assets || !params.amounts || params.assets.length !== params.amounts.length) {
            throw new Error('Invalid flash loan parameters: assets and amounts arrays must match');
        }

        if (params.assets.length === 0) {
            throw new Error('Invalid flash loan parameters: at least one asset is required');
        }

        switch (params.provider) {
            case FlashLoanProvider.AAVE:
                if (!params.modes || params.modes.length !== params.assets.length) {
                    throw new Error('Invalid Aave flash loan parameters: modes array must match assets array');
                }
                if (!params.onBehalfOf) {
                    throw new Error('Invalid Aave flash loan parameters: onBehalfOf address is required');
                }
                break;
            case FlashLoanProvider.MOONWELL:
                if (!params.maxBorrowRates || params.maxBorrowRates.length !== params.assets.length) {
                    throw new Error('Invalid Moonwell flash loan parameters: maxBorrowRates array must match assets array');
                }
                break;
            case FlashLoanProvider.BASESWAP:
                if (!params.pairs || params.pairs.length === 0) {
                    throw new Error('Invalid BaseSwap flash loan parameters: pairs array is required');
                }
                break;
        }
    }

    private async executeMorphoFlashLoan(params: FlashLoanParams): Promise<boolean> {
        const morphoFlash = new Contract(
            this.MORPHO_FLASH_ADDRESS,
            MORPHO_FLASH_ABI,
            this.provider
        );

        // Check if flash loan amounts are within limits
        for (let i = 0; i < params.assets.length; i++) {
            const maxAmount = await morphoFlash.maxFlashLoan(params.assets[i]);
            if (params.amounts[i] > maxAmount) {
                throw new Error(`Flash loan amount exceeds maximum for asset ${params.assets[i]}`);
            }
        }

        try {
            if (!Signer.isSigner(this.provider)) {
                throw new Error('Provider must be a signer to execute flash loans');
            }
            
            const tx = await morphoFlash.connect(this.provider).flashLoan(
                params.onBehalfOf || await this.provider.getAddress(),
                params.assets,
                params.amounts,
                params.params
            );
            await tx.wait();
            return true;
        } catch (error: unknown) {
            if (error instanceof Error) {
                console.error('Morpho flash loan execution failed:', error.message);
                throw error;
            }
            throw new Error('Morpho flash loan execution failed with unknown error');
        }
    }

    private async executeAaveFlashLoan(params: FlashLoanParams): Promise<boolean> {
        const aavePool = new Contract(
            this.AAVE_POOL_ADDRESS,
            AAVE_POOL_ABI,
            this.provider
        );

        // Check if assets are available in Aave
        for (let i = 0; i < params.assets.length; i++) {
            const reserveData = await aavePool.getReserveData(params.assets[i]);
            if (!reserveData || reserveData.totalAToken.isZero()) {
                throw new Error(`Asset ${params.assets[i]} not available for flash loan in Aave`);
            }
        }

        try {
            if (!Signer.isSigner(this.provider)) {
                throw new Error('Provider must be a signer to execute flash loans');
            }
            
            const signerAddress = await this.provider.getAddress();
            
            const tx = await aavePool.connect(this.provider).flashLoan(
                params.onBehalfOf || signerAddress,
                params.assets,
                params.amounts,
                params.modes || params.assets.map(() => 0), // Default to no debt mode
                params.onBehalfOf || signerAddress,
                params.params,
                params.referralCode || 0
            );
            await tx.wait();
            return true;
        } catch (error: unknown) {
            if (error instanceof Error) {
                console.error('Aave flash loan execution failed:', error.message);
                throw error;
            }
            throw new Error('Aave flash loan execution failed with unknown error');
        }
    }

    private async executeMoonwellFlashLoan(params: FlashLoanParams): Promise<boolean> {
        const moonwellFlash = new Contract(
            this.MOONWELL_FLASH_ADDRESS,
            MOONWELL_FLASH_ABI,
            this.provider
        );

        const MAX_UINT256 = ethers.getBigInt("0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");

        // Check market data and borrow rates
        for (let i = 0; i < params.assets.length; i++) {
            const marketData = await moonwellFlash.getMarketData(params.assets[i]);
            if (!marketData || marketData.totalCash.lt(params.amounts[i])) {
                throw new Error(`Insufficient liquidity for asset ${params.assets[i]} in Moonwell`);
            }
            
            if (marketData.borrowRate.gt(params.maxBorrowRates?.[i] || MAX_UINT256)) {
                throw new Error(`Borrow rate exceeds maximum for asset ${params.assets[i]}`);
            }
        }

        try {
            if (!Signer.isSigner(this.provider)) {
                throw new Error('Provider must be a signer to execute flash loans');
            }

            const tx = await moonwellFlash.connect(this.provider).flashLoan(
                params.onBehalfOf || await this.provider.getAddress(),
                params.assets,
                params.amounts,
                params.maxBorrowRates || params.assets.map(() => MAX_UINT256),
                params.params
            );
            await tx.wait();
            return true;
        } catch (error: unknown) {
            if (error instanceof Error) {
                console.error('Moonwell flash loan execution failed:', error.message);
                throw error;
            }
            throw new Error('Moonwell flash loan execution failed with unknown error');
        }
    }

    private async executeBaseSwapFlashLoan(params: FlashLoanParams): Promise<boolean> {
        if (!params.pairs || params.pairs.length === 0) {
            throw new Error('BaseSwap flash loan requires at least one pair address');
        }

        try {
            if (!Signer.isSigner(this.provider)) {
                throw new Error('Provider must be a signer to execute flash loans');
            }

            const signerAddress = await this.provider.getAddress();

            // Execute flash swaps for each pair
            for (let i = 0; i < params.pairs.length; i++) {
                const pair = new Contract(
                    params.pairs[i],
                    BASESWAP_FLASH_ABI,
                    this.provider
                );

                const amount0 = params.isExactInput ? params.amounts[i * 2] : 0n;
                const amount1 = params.isExactInput ? params.amounts[i * 2 + 1] : 0n;

                const tx = await pair.connect(this.provider).flash(
                    params.onBehalfOf || signerAddress,
                    amount0,
                    amount1,
                    params.params
                );
                await tx.wait();
            }
            return true;
        } catch (error: unknown) {
            if (error instanceof Error) {
                console.error('BaseSwap flash loan execution failed:', error.message);
                throw error;
            }
            throw new Error('BaseSwap flash loan execution failed with unknown error');
        }
    }

    async getFlashLoanFee(provider: FlashLoanProvider, asset: string): Promise<number> {
        try {
            switch (provider) {
                case FlashLoanProvider.MORPHO:
                    return 0; // Morpho has no flash loan fees
                case FlashLoanProvider.AAVE:
                    const aavePool = new Contract(
                        this.AAVE_POOL_ADDRESS,
                        AAVE_POOL_ABI,
                        this.provider
                    );
                    const premium = await aavePool.FLASHLOAN_PREMIUM_TOTAL();
                    return Number(premium) / 10000; // Convert basis points to percentage
                case FlashLoanProvider.MOONWELL:
                    const moonwellFlash = new Contract(
                        this.MOONWELL_FLASH_ADDRESS,
                        MOONWELL_FLASH_ABI,
                        this.provider
                    );
                    const fee = await moonwellFlash.getFlashLoanFee(asset);
                    return Number(fee) / 10000; // Convert basis points to percentage
                case FlashLoanProvider.BASESWAP:
                    return 0.003; // 0.3% fee for BaseSwap flash swaps
                default:
                    throw new Error('Unsupported flash loan provider');
            }
        } catch (error: unknown) {
            if (error instanceof Error) {
                console.error(`Failed to get flash loan fee: ${error.message}`);
                throw error;
            }
            throw new Error('Failed to get flash loan fee with unknown error');
        }
    }

    clearCache(): void {
        this.poolCache.clear();
        this.priceCache.clear();
    }
} 