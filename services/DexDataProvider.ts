import { PublicClient, Transport, getContract, createPublicClient, http, BlockTag, Block, GetBlockParameters, Chain, GetBlockReturnType, Transaction, TransactionType } from 'viem';
import { base } from 'viem/chains';
import { DEXType } from '../scripts/types/dex';
import { FACTORY_ABI, POOL_ABI } from '../abis/UniswapV2';
import type { Hex, TransactionReceipt } from 'viem';
import { ProcessedBlock, processBlock, ProcessedTransaction } from '../scripts/types/blocks';
import { Provider, JsonRpcProvider } from '@ethersproject/providers';
import { Signer } from '@ethersproject/abstract-signer';
import { ethers } from 'ethers';
import { Pool } from '../scripts/types/dex';

const DEX_FACTORY_ADDRESSES: Record<DEXType, `0x${string}`> = {
    [DEXType.UNISWAP_V2]: '0x1F98431c8aD98523631AE4a59f267346ea31F984',
    [DEXType.SUSHISWAP]: '0xc35DADB65012eC5796536bD9864eD8773aBc74C4',
    [DEXType.BASESWAP]: '0x0000000000000000000000000000000000000000',
    [DEXType.AERODROME]: '0x0000000000000000000000000000000000000000'
} as const;

type PoolContract = ReturnType<typeof getContract>;
type StableCheckContract = ReturnType<typeof getContract>;

export class DexDataProvider {
    private readonly client: PublicClient;
    private poolCache: Map<string, Pool & { lastUpdate: number }> = new Map();
    private readonly CACHE_DURATION = 30000; // 30 seconds
    private readonly provider: Provider;

    constructor(client?: PublicClient) {
        if (!client) {
            const rpcUrl = process.env.BASE_RPC_URL || 'http://localhost:8545';
            this.provider = new JsonRpcProvider(rpcUrl);
            this.client = createPublicClient<Transport, Chain>({
                chain: base,
                transport: http(rpcUrl),
                cacheTime: this.CACHE_DURATION
            });
        } else {
            this.client = client;
            const rpcUrl = (client.transport as any).url;
            this.provider = new JsonRpcProvider(rpcUrl);
        }
    }

    async getAllPools(dexType: DEXType): Promise<Pool[]> {
        const now = Date.now();
        const factoryAddress = DEX_FACTORY_ADDRESSES[dexType];
        const cachedPool = this.poolCache.get(factoryAddress);
        if (cachedPool && now - cachedPool.lastUpdate < this.CACHE_DURATION && this.poolCache.size > 0) {
            return Array.from(this.poolCache.values());
        }

        try {
            const factory = getContract({
                address: factoryAddress,
                abi: FACTORY_ABI,
                client: this.client
            });

            // Get pool count
            const poolCount = await factory.read.allPairsLength();
            const pools: Pool[] = [];

            // Fetch pools in batches
            const batchSize = 100;
            for (let i = 0; i < Number(poolCount) && i < 1000; i += batchSize) {
                const batch = await Promise.all(
                    Array.from({ length: Math.min(batchSize, Number(poolCount) - i) }, async (_, j) => {
                        const poolAddress = await factory.read.allPairs([BigInt(i + j)]) as `0x${string}`;
                        return this.getPoolData(dexType, poolAddress);
                    })
                );
                pools.push(...batch.filter(Boolean));
            }

            // Update cache
            this.poolCache.clear();
            pools.forEach(pool => this.updatePoolCache(pool));

            return pools;
        } catch (error) {
            console.error(`Error fetching pools for ${dexType}:`, error);
            return Array.from(this.poolCache.values());
        }
    }

    async getPoolData(dexType: DEXType, poolAddress: `0x${string}`): Promise<Pool> {
        try {
            const pool = getContract({
                address: poolAddress,
                abi: POOL_ABI,
                client: this.client
            });

            const [token0, token1, reserves, fee] = await Promise.all([
                pool.read.token0(),
                pool.read.token1(),
                pool.read.getReserves(),
                this.getPoolFee(dexType, poolAddress)
            ]);

            return {
                address: poolAddress,
                token0,
                token1,
                reserve0: BigInt(reserves[0]),
                reserve1: BigInt(reserves[1]),
                fee: Number(fee) / 10000,
                isStable: dexType === DEXType.AERODROME ? await this.checkIsStablePool(poolAddress) : undefined
            };
        } catch (error) {
            console.error(`Error fetching pool data for ${poolAddress}:`, error);
            throw error;
        }
    }

    private async getPoolFee(dexType: DEXType, poolAddress: `0x${string}`): Promise<number> {
        const defaultFees: Record<DEXType, number> = {
            [DEXType.UNISWAP_V2]: 3000,
            [DEXType.SUSHISWAP]: 3000,
            [DEXType.BASESWAP]: 3000,
            [DEXType.AERODROME]: 3000
        };

        try {
            const pool = getContract({
                address: poolAddress,
                abi: POOL_ABI,
                client: this.client
            });

            try {
                const fee = await pool.read.fee();
                return Number(fee);
            } catch {
                return defaultFees[dexType];
            }
        } catch {
            return defaultFees[dexType];
        }
    }

    async getPriceData(dexType: DEXType, poolAddress: string) {
        // Mock implementation for testing
        return {
            price: 1800n,
            volume24h: 1000000n,
            lastUpdate: Date.now()
        };
    }

    async getProvider(): Promise<PublicClient> {
        return this.client;
    }

    public async getBlock(blockTag: BlockTag = 'latest') {
        return this.client.getBlock({
            blockTag,
            includeTransactions: true
        });
    }

    public async getBlockForBot(blockTag: BlockTag = 'latest'): Promise<ProcessedBlock> {
        const block = await this.client.getBlock({
            blockTag,
            includeTransactions: true
        });

        if (!block) throw new Error('Failed to fetch block');
        return processBlock(block as GetBlockReturnType<typeof base, true>);
    }

    private isDexContract(address: `0x${string}` | null): boolean {
        if (!address) return false;
        
        // Check if the address is a known DEX factory or router
        return Object.values(DEX_FACTORY_ADDRESSES).some(
            factoryAddress => factoryAddress.toLowerCase() === address.toLowerCase()
        ) || this.poolCache.has(address.toLowerCase());
    }

    private isSwapTransaction(tx: ProcessedTransaction): boolean {
        if (!tx.input || !tx.to) return false;
        return tx.input.length >= 10 && this.isDexContract(tx.to);
    }

    private async checkIsStablePool(poolAddress: `0x${string}`): Promise<boolean> {
        try {
            // Create a separate ABI for stable check
            const STABLE_CHECK_ABI = [{
                inputs: [],
                name: 'stable',
                outputs: [{ type: 'bool' }],
                stateMutability: 'view',
                type: 'function'
            }] as const;

            const pool = getContract({
                address: poolAddress,
                abi: STABLE_CHECK_ABI,
                client: this.client
            });

            const isStable = await pool.read.stable();
            // Explicitly type the return value
            return Boolean(isStable);
        } catch {
            return false;
        }
    }

    private async processTransaction(tx: ProcessedTransaction): Promise<boolean> {
        try {
            if (!this.isSwapTransaction(tx)) return false;

            const gasPriceValue = tx.maxFeePerGas ?? tx.maxPriorityFeePerGas ?? 0n;
            
            if (gasPriceValue === 0n) return false;

            await this.updatePoolStates(tx);
            return true;
        } catch (error) {
            console.error('Error processing transaction:', error);
            return false;
        }
    }

    private async updatePoolStates(tx: ProcessedTransaction): Promise<void> {
        if (!tx.to) return;
        
        const poolData = this.poolCache.get(tx.to.toLowerCase());
        if (!poolData) return;

        // Force cache refresh for affected pool
        await this.getPoolData(DEXType.UNISWAP_V2, poolData.address);
    }

    private async getPoolContract(address: `0x${string}`): Promise<PoolContract> {
        return getContract({
            address,
            abi: POOL_ABI,
            client: this.client
        });
    }

    private updatePoolCache(pool: Pool): void {
        this.poolCache.set(pool.address.toLowerCase(), {
            ...pool,
            lastUpdate: Date.now()
        });
    }

    async getPools(): Promise<Pool[]> {
        return [];
    }
} 