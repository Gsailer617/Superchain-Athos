import path from 'path';
import { DatabaseManager, BackupMetadata } from './DatabaseManager';

interface TaxTransaction {
    acquiredDate: number;
    soldDate: number;
    type: string;
    asset: string;
    quantity: number;
    costBasis: number;
    proceeds: number;
    gasUsed: number;
    gasPrice: number;
    txHash: string;
}

export class TaxDataManager {
    private transactions: TaxTransaction[] = [];
    private readonly LONG_TERM_THRESHOLD = 365 * 24 * 60 * 60 * 1000; // 1 year in milliseconds
    private readonly db: DatabaseManager;
    private readonly COLLECTION_NAME = 'tax_transactions';

    constructor() {
        this.db = new DatabaseManager({
            dataDir: path.join(process.cwd(), 'data'),
            backupDir: path.join(process.cwd(), 'backups'),
            indexedFields: ['txHash', 'asset', 'type'],
            maxBackups: 10,
            compressionLevel: 6
        });
        this.loadTransactions();
    }

    private async loadTransactions() {
        this.transactions = await this.db.loadCollection(this.COLLECTION_NAME);
    }

    async recordTransaction(transaction: TaxTransaction) {
        if (!this.validateTransaction(transaction)) {
            throw new Error('Invalid transaction data');
        }

        this.transactions.push(transaction);
        await this.db.saveCollection(this.COLLECTION_NAME, this.transactions);
    }

    private validateTransaction(transaction: TaxTransaction): boolean {
        return (
            typeof transaction.acquiredDate === 'number' &&
            typeof transaction.soldDate === 'number' &&
            typeof transaction.type === 'string' &&
            typeof transaction.asset === 'string' &&
            typeof transaction.quantity === 'number' &&
            typeof transaction.costBasis === 'number' &&
            typeof transaction.proceeds === 'number' &&
            typeof transaction.gasUsed === 'number' &&
            typeof transaction.gasPrice === 'number' &&
            typeof transaction.txHash === 'string' &&
            transaction.acquiredDate <= transaction.soldDate &&
            transaction.quantity > 0 &&
            transaction.costBasis >= 0 &&
            transaction.proceeds >= 0
        );
    }

    async getAllTransactions(): Promise<TaxTransaction[]> {
        return this.transactions;
    }

    async getTransactionsForYear(year: number): Promise<TaxTransaction[]> {
        const startOfYear = new Date(year, 0, 1).getTime();
        const endOfYear = new Date(year + 1, 0, 1).getTime() - 1;

        return this.db.query(this.COLLECTION_NAME, {
            soldDate: (date: number) => date >= startOfYear && date <= endOfYear
        });
    }

    isLongTerm(acquiredDate: number, soldDate: number): boolean {
        return soldDate - acquiredDate >= this.LONG_TERM_THRESHOLD;
    }

    async calculateGains(startDate: number, endDate: number) {
        const transactions = await this.db.query(this.COLLECTION_NAME, {
            soldDate: (date: number) => date >= startDate && date <= endDate
        });

        return {
            shortTermGains: this.calculateGainsByType(transactions, false),
            longTermGains: this.calculateGainsByType(transactions, true),
            totalGains: transactions.reduce((sum, tx) => sum + (tx.proceeds - tx.costBasis), 0)
        };
    }

    private calculateGainsByType(transactions: TaxTransaction[], longTerm: boolean): number {
        return transactions
            .filter(tx => this.isLongTerm(tx.acquiredDate, tx.soldDate) === longTerm)
            .reduce((sum, tx) => sum + (tx.proceeds - tx.costBasis), 0);
    }

    async getAnnualSummary(year: number) {
        const transactions = await this.getTransactionsForYear(year);
        const gains = {
            shortTerm: 0,
            longTerm: 0,
            total: 0
        };

        const assetSummary: {
            [asset: string]: {
                totalVolume: number;
                totalGains: number;
                tradeCount: number;
                averageHoldingPeriod: number;
            };
        } = {};

        for (const tx of transactions) {
            const gain = tx.proceeds - tx.costBasis;
            const isLong = this.isLongTerm(tx.acquiredDate, tx.soldDate);
            
            if (isLong) {
                gains.longTerm += gain;
            } else {
                gains.shortTerm += gain;
            }
            gains.total += gain;

            if (!assetSummary[tx.asset]) {
                assetSummary[tx.asset] = {
                    totalVolume: 0,
                    totalGains: 0,
                    tradeCount: 0,
                    averageHoldingPeriod: 0
                };
            }

            const summary = assetSummary[tx.asset];
            summary.totalVolume += tx.quantity;
            summary.totalGains += gain;
            summary.tradeCount++;
            summary.averageHoldingPeriod = (
                (summary.averageHoldingPeriod * (summary.tradeCount - 1) +
                (tx.soldDate - tx.acquiredDate)) / summary.tradeCount
            );
        }

        return {
            year,
            gains,
            assetSummary,
            totalTransactions: transactions.length,
            totalGasCost: transactions.reduce((sum, tx) => sum + (tx.gasUsed * tx.gasPrice), 0)
        };
    }

    async getMonthlyBreakdown(year: number) {
        const transactions = await this.getTransactionsForYear(year);
        const monthlyData: {
            [month: number]: {
                gains: { shortTerm: number; longTerm: number; total: number };
                volume: number;
                tradeCount: number;
            };
        } = {};

        for (let i = 0; i < 12; i++) {
            monthlyData[i] = {
                gains: { shortTerm: 0, longTerm: 0, total: 0 },
                volume: 0,
                tradeCount: 0
            };
        }

        for (const tx of transactions) {
            const month = new Date(tx.soldDate).getMonth();
            const gain = tx.proceeds - tx.costBasis;
            const isLong = this.isLongTerm(tx.acquiredDate, tx.soldDate);

            monthlyData[month].gains[isLong ? 'longTerm' : 'shortTerm'] += gain;
            monthlyData[month].gains.total += gain;
            monthlyData[month].volume += tx.quantity;
            monthlyData[month].tradeCount++;
        }

        return monthlyData;
    }

    async getCostBasisReport() {
        const transactions = await this.getAllTransactions();
        const assetHoldings: {
            [asset: string]: {
                quantity: number;
                costBasis: number;
                averagePrice: number;
                lastUpdated: number;
                unrealizedGains?: number;
            };
        } = {};

        // Sort transactions by date
        transactions.sort((a, b) => a.acquiredDate - b.acquiredDate);

        for (const tx of transactions) {
            if (!assetHoldings[tx.asset]) {
                assetHoldings[tx.asset] = {
                    quantity: 0,
                    costBasis: 0,
                    averagePrice: 0,
                    lastUpdated: tx.acquiredDate
                };
            }

            const holding = assetHoldings[tx.asset];
            holding.quantity += tx.quantity;
            holding.costBasis += tx.costBasis;
            holding.averagePrice = holding.costBasis / holding.quantity;
            holding.lastUpdated = Math.max(holding.lastUpdated, tx.soldDate);
        }

        return assetHoldings;
    }

    async generateTaxReport(year: number) {
        const annualSummary = await this.getAnnualSummary(year);
        const monthlyBreakdown = await this.getMonthlyBreakdown(year);
        const costBasisReport = await this.getCostBasisReport();

        return {
            year,
            summary: annualSummary,
            monthlyBreakdown,
            costBasis: costBasisReport,
            generatedAt: Date.now()
        };
    }

    async createBackup(): Promise<void> {
        await this.db.createBackup();
    }

    async restoreFromBackup(timestamp: number): Promise<void> {
        await this.db.restoreFromBackup(timestamp);
        await this.loadTransactions();
    }

    async getBackupsList(): Promise<BackupMetadata[]> {
        return this.db.getBackupsList();
    }
} 