import { BoostOptimizer } from '../utils/BoostOptimizer';
import { CSVExporter } from '../utils/CSVExporter';
import path from 'path';

async function testCSVExport() {
    try {
        console.log('Starting CSV export test...');
        
        // Initialize BoostOptimizer with some test data
        const optimizer = new BoostOptimizer();
        
        // Record some sample trades
        const pairs = ['ETH-USDC', 'WBTC-USDC'];
        pairs.forEach(pair => {
            // Simulate 10 trades for each pair
            for (let i = 0; i < 10; i++) {
                optimizer.recordTrade(
                    pair,
                    Math.random() > 0.2,  // 80% success rate
                    Math.random() * 100,   // 0-100 USD profit
                    Math.random() * 50,    // 0-50 gwei gas
                    1 + Math.random() * 2  // 1-3x boost
                );
            }
        });

        // Create CSV exporter
        const exporter = new CSVExporter(optimizer);
        
        // Export performance report
        const reportDir = path.join(process.cwd(), 'reports');
        console.log('Exporting performance report...');
        exporter.exportPerformanceReport(reportDir);
        
        console.log('Test completed! Check the reports directory for CSV files.');
    } catch (error) {
        console.error('Test failed:', error);
    }
}

// Run the test
testCSVExport(); 