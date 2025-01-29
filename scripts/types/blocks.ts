import { Block, Transaction, Chain, GetBlockReturnType } from 'viem';
import { base } from 'viem/chains';

export type TransactionType = 'legacy' | 'eip2930' | 'eip1559' | 'eip4844' | 'eip7702' | 'deposit';

export interface ProcessedTransaction {
    type: TransactionType;
    hash: `0x${string}`;
    blockNumber: bigint;
    from: `0x${string}`;
    to: `0x${string}` | null;
    value: bigint;
    gas: bigint;
    nonce: number;
    maxFeePerGas: bigint;
    maxPriorityFeePerGas?: bigint;
    input: `0x${string}`;
}

export interface ProcessedBlock {
    number: bigint;
    timestamp: bigint;
    transactions: ProcessedTransaction[];
}

export function processBlock(block: GetBlockReturnType<typeof base, true>): ProcessedBlock {
    if (!block.number || !block.hash || !block.nonce || !block.parentHash || !block.timestamp || 
        !block.difficulty || !block.gasLimit || !block.gasUsed || !block.miner || !block.extraData) {
        throw new Error(`Required block data is missing: ${JSON.stringify({
            number: !!block.number,
            hash: !!block.hash,
            nonce: !!block.nonce,
            parentHash: !!block.parentHash,
            timestamp: !!block.timestamp,
            difficulty: !!block.difficulty,
            gasLimit: !!block.gasLimit,
            gasUsed: !!block.gasUsed,
            miner: !!block.miner,
            extraData: !!block.extraData
        })}`);
    }

    const transactions = block.transactions.map(tx => {
        if (typeof tx === 'string') {
            throw new Error('Expected full transaction object');
        }

        const processedTx: ProcessedTransaction = {
            type: tx.type ?? 'legacy',
            hash: tx.hash,
            blockNumber: block.number!,
            from: tx.from,
            to: tx.to,
            value: tx.value,
            gas: tx.gas,
            nonce: tx.nonce,
            maxFeePerGas: tx.maxFeePerGas ?? 0n,
            maxPriorityFeePerGas: tx.maxPriorityFeePerGas,
            input: tx.input ?? '0x'
        };
        return processedTx;
    });

    return {
        number: block.number,
        timestamp: block.timestamp,
        transactions
    };
} 