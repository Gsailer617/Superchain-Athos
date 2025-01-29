import { BoostOptimizer } from './BoostOptimizer';
import fs from 'fs';
import path from 'path';

interface ExportOptions {
    includeTimestamps?: boolean;
    dateFormat?: string;
    precision?: number;
}

export class CSVExporter {
    private readonly DEFAULT_OPTIONS: ExportOptions = {
        includeTimestamps: true,
        dateFormat: 'YYYY-MM-DD HH:mm:ss',
        precision: 6
    };

    constructor(private boostOptimizer: BoostOptimizer) {}

    exportTradeHistory(pair: string, filepath: string, options: ExportOptions = {}): void {
        const opts = { ...this.DEFAULT_OPTIONS, ...options };
        const stats = this.boostOptimizer.getTradeStats(pair);
        
        const headers = [
            'Timestamp',
            'Pair',
            'Success',
            'Profit USD',
            'Gas Used (gwei)',
            'Boost Used',
            'Profit/Gas Ratio',
            'Running Success Rate',
            'Running Avg Profit'
        ].join(',');

        const rows = this.formatTradeRows(pair, opts);
        
        const summary = [
            '\nSummary Statistics',
            `Total Trades,${stats.totalTrades}`,
            `Overall Success Rate,${(stats.successRate * 100).toFixed(2)}%`,
            `Average Profit,$${stats.averageProfit.toFixed(2)}`,
            `Average Boost,${stats.averageBoost.toFixed(2)}x`,
            `Average Profit/Gas,${stats.profitPerGas.toFixed(4)}`
        ].join('\n');

        const content = [headers, ...rows, summary].join('\n');
        this.writeToFile(filepath, content);
    }

    exportBoostMetrics(pair: string, filepath: string, options: ExportOptions = {}): void {
        const opts = { ...this.DEFAULT_OPTIONS, ...options };
        
        const headers = [
            'Timestamp',
            'Pair',
            'Boost Multiplier',
            'Success Rate',
            'Average Profit',
            'Volatility Score',
            'Liquidity Score',
            'Gas Efficiency',
            'Adjustment Reason'
        ].join(',');

        const rows = this.formatBoostRows(pair, opts);
        const content = [headers, ...rows].join('\n');
        this.writeToFile(filepath, content);
    }

    exportPerformanceReport(outputDir: string, options: ExportOptions = {}): void {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const reportDir = path.join(outputDir, `performance_report_${timestamp}`);
        
        // Create report directory
        if (!fs.existsSync(reportDir)) {
            fs.mkdirSync(reportDir, { recursive: true });
        }

        // Export README with report overview
        const readme = this.generateReportReadme();
        this.writeToFile(path.join(reportDir, 'README.md'), readme);

        // Export trade history for each pair
        const pairs = this.getUniquePairs();
        pairs.forEach(pair => {
            this.exportTradeHistory(
                pair,
                path.join(reportDir, `${pair}_trades.csv`),
                options
            );
            this.exportBoostMetrics(
                pair,
                path.join(reportDir, `${pair}_boost_metrics.csv`),
                options
            );
        });

        // Export aggregated statistics
        this.exportAggregatedStats(path.join(reportDir, 'aggregated_stats.csv'));
    }

    private formatTradeRows(pair: string, options: ExportOptions): string[] {
        const history = this.getTradeHistory(pair);
        let runningSuccesses = 0;
        let runningProfitSum = 0;

        return history.map((trade, index) => {
            if (trade.success) runningSuccesses++;
            runningProfitSum += trade.profit;
            
            const timestamp = options.includeTimestamps
                ? this.formatDate(trade.timestamp, options.dateFormat)
                : '';
            
            return [
                timestamp,
                pair,
                trade.success ? 'Yes' : 'No',
                trade.profit.toFixed(options.precision),
                trade.gasUsed.toFixed(2),
                trade.boostUsed.toFixed(2),
                (trade.profit / trade.gasUsed).toFixed(4),
                ((runningSuccesses / (index + 1)) * 100).toFixed(2) + '%',
                (runningProfitSum / (index + 1)).toFixed(2)
            ].join(',');
        });
    }

    private formatBoostRows(pair: string, options: ExportOptions): string[] {
        const metrics = this.getBoostHistory(pair);
        
        return metrics.map(metric => {
            const timestamp = options.includeTimestamps
                ? this.formatDate(metric.timestamp, options.dateFormat)
                : '';
            
            return [
                timestamp,
                pair,
                metric.multiplier.toFixed(2),
                (metric.metrics.successRate * 100).toFixed(2) + '%',
                metric.metrics.averageProfit.toFixed(2),
                (metric.metrics.volatilityScore * 100).toFixed(2) + '%',
                (metric.metrics.liquidityScore * 100).toFixed(2) + '%',
                metric.metrics.gasEfficiency.toFixed(2),
                `"${metric.reason}"`
            ].join(',');
        });
    }

    private exportAggregatedStats(filepath: string): void {
        const pairs = this.getUniquePairs();
        const stats = pairs.map(pair => {
            const tradeStats = this.boostOptimizer.getTradeStats(pair);
            return {
                pair,
                ...tradeStats,
                profitability: tradeStats.averageProfit * tradeStats.successRate
            };
        });

        const headers = [
            'Pair',
            'Total Trades',
            'Success Rate',
            'Average Profit',
            'Average Boost',
            'Profit/Gas Ratio',
            'Overall Profitability'
        ].join(',');

        const rows = stats.map(stat => [
            stat.pair,
            stat.totalTrades,
            (stat.successRate * 100).toFixed(2) + '%',
            stat.averageProfit.toFixed(2),
            stat.averageBoost.toFixed(2),
            stat.profitPerGas.toFixed(4),
            stat.profitability.toFixed(2)
        ].join(','));

        const content = [headers, ...rows].join('\n');
        this.writeToFile(filepath, content);
    }

    private generateReportReadme(): string {
        const timestamp = new Date().toISOString();
        const pairs = this.getUniquePairs();
        const totalTrades = pairs.reduce((sum, pair) => 
            sum + this.boostOptimizer.getTradeStats(pair).totalTrades, 0);

        return `# Trading Performance Report
Generated: ${timestamp}

## Overview
- Total Pairs Traded: ${pairs.length}
- Total Trades: ${totalTrades}

## Files Included
1. \`aggregated_stats.csv\` - Overall performance metrics for all pairs
${pairs.map(pair => `2. \`${pair}_trades.csv\` - Detailed trade history for ${pair}
3. \`${pair}_boost_metrics.csv\` - Boost optimization metrics for ${pair}`).join('\n')}

## Analysis Tips
- Use pivot tables to analyze performance by time period
- Create charts to visualize boost vs. profitability
- Look for correlations between gas efficiency and profit
- Track success rate trends over time

## Metrics Explained
- Success Rate: Percentage of profitable trades
- Average Profit: Mean profit per trade in USD
- Gas Efficiency: Profit generated per unit of gas spent
- Boost Multiplier: Leverage factor used in trades
- Volatility Score: Market volatility metric (0-100%)
- Liquidity Score: Available market liquidity metric (0-100%)`;
    }

    private writeToFile(filepath: string, content: string): void {
        try {
            fs.writeFileSync(filepath, content, 'utf8');
            console.log(`Successfully exported to ${filepath}`);
        } catch (error) {
            console.error(`Failed to write to ${filepath}:`, error);
            throw error;
        }
    }

    private formatDate(timestamp: number, format: string = 'YYYY-MM-DD HH:mm:ss'): string {
        const date = new Date(timestamp);
        return date.toISOString().replace('T', ' ').split('.')[0];
    }

    private getUniquePairs(): string[] {
        // This would come from your BoostOptimizer's trade history
        return Array.from(this.boostOptimizer.getUniquePairs());
    }

    private getTradeHistory(pair: string): any[] {
        // This would come from your BoostOptimizer's trade history
        return [];
    }

    private getBoostHistory(pair: string): any[] {
        // This would come from your BoostOptimizer's boost history
        return [];
    }
} 