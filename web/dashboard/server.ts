import express from 'express';
import { WebSocket, WebSocketServer } from 'ws';
import http from 'http';
import path from 'path';
import { BoostOptimizer } from '../../scripts/utils/BoostOptimizer';
import { TaxDataManager } from '../../scripts/utils/TaxDataManager';
import PDFDocument from 'pdfkit';

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

// Initialize services
const boostOptimizer = new BoostOptimizer();
const taxManager = new TaxDataManager();

// Types
interface TaxTransaction {
    asset: string;
    quantity: number;
    acquiredDate: number;
    soldDate: number;
    proceeds: number;
    costBasis: number;
    gasUsed: number;
    gasPrice: number;
}

interface PairData {
    name: string;
    volume: number;
    successRate: number;
    avgProfit: number;
    avgBoost: number;
}

// Middleware
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// API Routes
app.get('/api/dashboard/metrics', (_req, res) => {
    const metrics = getDashboardMetrics();
    res.json(metrics);
});

app.get('/api/dashboard/tax-data', (_req, res) => {
    const taxData = getTaxData();
    res.json(taxData);
});

app.post('/api/dashboard/generate-8949', async (req, res) => {
    try {
        const { year } = req.body;
        const transactions = await taxManager.getTransactionsForYear(year);
        
        // Create PDF document
        const doc = new PDFDocument();
        res.setHeader('Content-Type', 'application/pdf');
        res.setHeader('Content-Disposition', `attachment; filename=form_8949_${year}.pdf`);
        
        // Pipe PDF to response
        doc.pipe(res);
        
        // Generate Form 8949
        generateForm8949(doc, transactions as TaxTransaction[], year);
        
        doc.end();
    } catch (error) {
        console.error('Failed to generate Form 8949:', error);
        res.status(500).json({ error: 'Failed to generate Form 8949' });
    }
});

// WebSocket connection handling
wss.on('connection', (ws: WebSocket) => {
    console.log('New client connected');
    
    // Send initial data
    ws.send(JSON.stringify(getDashboardMetrics()));
    
    // Setup periodic updates
    const updateInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(getDashboardMetrics()));
        }
    }, 15000);
    
    ws.on('close', () => {
        console.log('Client disconnected');
        clearInterval(updateInterval);
    });
});

function getDashboardMetrics() {
    const pairs = boostOptimizer.getAllPairs();
    const last24Hours = Date.now() - 24 * 60 * 60 * 1000;
    
    // Calculate overview metrics
    const overview = {
        totalProfit: 0,
        successRate: 0,
        activePairs: pairs.length,
        gasEfficiency: 0
    };
    
    let totalTrades = 0;
    let successfulTrades = 0;
    let totalGasUsed = 0;
    
    // Process trade history for each pair
    const pairsData: PairData[] = pairs.map(pair => {
        const stats = boostOptimizer.getTradeStats(pair);
        const recentTrades = boostOptimizer.getTradesForPair(pair)
            .filter(trade => trade.timestamp >= last24Hours);
        
        // Update overview metrics
        totalTrades += recentTrades.length;
        successfulTrades += recentTrades.filter(t => t.success).length;
        overview.totalProfit += recentTrades.reduce((sum, t) => sum + t.profit, 0);
        totalGasUsed += recentTrades.reduce((sum, t) => sum + t.gasUsed, 0);
        
        return {
            name: pair,
            volume: recentTrades.reduce((sum, t) => sum + (t.profit + t.gasUsed), 0),
            successRate: stats.successRate,
            avgProfit: stats.averageProfit,
            avgBoost: stats.averageBoost
        };
    });
    
    // Finalize overview calculations
    overview.successRate = totalTrades > 0 ? successfulTrades / totalTrades : 0;
    overview.gasEfficiency = totalGasUsed > 0 ? overview.totalProfit / totalGasUsed : 0;
    
    // Prepare chart data
    const chartData = {
        profit: getProfitChartData(),
        boost: getBoostChartData()
    };
    
    return {
        overview,
        pairs: pairsData,
        charts: chartData
    };
}

function getTaxData() {
    const transactions = taxManager.getAllTransactions();
    return {
        transactions: (Array.isArray(transactions) ? transactions : []).map((tx: TaxTransaction) => ({
            ...tx,
            costBasisWithGas: tx.costBasis + (tx.gasUsed * tx.gasPrice),
            netProceeds: tx.proceeds - (tx.gasUsed * tx.gasPrice)
        }))
    };
}

function generateForm8949(doc: PDFKit.PDFDocument, transactions: TaxTransaction[], year: number) {
    // Add form header
    doc.fontSize(16)
        .text(`Form 8949 - Crypto Trading Activity ${year}`, { align: 'center' })
        .moveDown();
    
    // Add taxpayer information section
    doc.fontSize(12)
        .text('Part I - Short-term transactions (1 year or less)', { underline: true })
        .moveDown();
    
    // Filter and sort transactions
    const shortTermTx = transactions
        .filter(tx => !taxManager.isLongTerm(tx.acquiredDate, tx.soldDate))
        .sort((a, b) => a.soldDate - b.soldDate);
    
    // Add transaction table headers
    const headers = [
        'Description',
        'Date Acquired',
        'Date Sold',
        'Proceeds',
        'Cost Basis',
        'Gain/Loss'
    ];
    
    let y = doc.y;
    const columnWidth = 90;
    headers.forEach((header, i) => {
        doc.text(header, 50 + (i * columnWidth), y, { width: columnWidth, align: 'left' });
    });
    
    // Add transactions
    y += 20;
    shortTermTx.forEach(tx => {
        const columns = [
            `${tx.quantity.toFixed(6)} ${tx.asset}`,
            new Date(tx.acquiredDate).toLocaleDateString(),
            new Date(tx.soldDate).toLocaleDateString(),
            `$${tx.proceeds.toFixed(2)}`,
            `$${tx.costBasis.toFixed(2)}`,
            `$${(tx.proceeds - tx.costBasis).toFixed(2)}`
        ];
        
        columns.forEach((col, i) => {
            doc.text(col, 50 + (i * columnWidth), y, { width: columnWidth, align: 'left' });
        });
        
        y += 15;
        if (y > 700) {
            doc.addPage();
            y = 50;
        }
    });
    
    // Add totals
    const totals = shortTermTx.reduce((acc, tx) => ({
        proceeds: acc.proceeds + tx.proceeds,
        costBasis: acc.costBasis + tx.costBasis,
        gainLoss: acc.gainLoss + (tx.proceeds - tx.costBasis)
    }), { proceeds: 0, costBasis: 0, gainLoss: 0 });
    
    doc.moveDown()
        .text('Totals:', { underline: true })
        .text(`Total Proceeds: $${totals.proceeds.toFixed(2)}`)
        .text(`Total Cost Basis: $${totals.costBasis.toFixed(2)}`)
        .text(`Net Gain/Loss: $${totals.gainLoss.toFixed(2)}`);
    
    // Add long-term transactions section if needed
    const longTermTx = transactions.filter(tx => 
        taxManager.isLongTerm(tx.acquiredDate, tx.soldDate));
    
    if (longTermTx.length > 0) {
        doc.addPage()
            .fontSize(12)
            .text('Part II - Long-term transactions (more than 1 year)', { underline: true })
            .moveDown();
        
        // Add similar table for long-term transactions...
    }
}

function getProfitChartData() {
    const hourlyProfits = new Map<number, number>();
    const now = Date.now();
    const last24Hours = now - 24 * 60 * 60 * 1000;
    
    // Initialize hourly buckets
    for (let i = 0; i < 24; i++) {
        const timestamp = now - i * 60 * 60 * 1000;
        hourlyProfits.set(timestamp, 0);
    }
    
    // Aggregate profits by hour
    boostOptimizer.getAllTrades().forEach(([_, trades]) => {
        trades
            .filter(trade => trade.timestamp >= last24Hours)
            .forEach(trade => {
                const hourBucket = Math.floor(trade.timestamp / (60 * 60 * 1000)) * 60 * 60 * 1000;
                hourlyProfits.set(
                    hourBucket,
                    (hourlyProfits.get(hourBucket) || 0) + trade.profit
                );
            });
    });
    
    // Sort and format data
    return Array.from(hourlyProfits.entries())
        .sort(([a], [b]) => a - b)
        .map(([timestamp, profit]) => ({
            timestamp,
            profit
        }));
}

function getBoostChartData() {
    const boostData: { timestamp: number; boost: number }[] = [];
    
    boostOptimizer.getAllTrades().forEach(([_, trades]) => {
        trades.forEach(trade => {
            boostData.push({
                timestamp: trade.timestamp,
                boost: trade.boostUsed
            });
        });
    });
    
    return boostData.sort((a, b) => a.timestamp - b.timestamp);
}

// Start server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Dashboard server running on port ${PORT}`);
}); 