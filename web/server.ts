import express, { Request, Response } from 'express';
import cors from 'cors';
import path from 'path';
import { ArbitrageBot } from '../scripts/ArbitrageBot.js';

// Initialize environment variables
import dotenv from 'dotenv';
dotenv.config({ path: '../.env' });

const app = express();
const port = process.env.PORT || 3000;

// Initialize bot with error handling
let bot: ArbitrageBot;
try {
    bot = new ArbitrageBot();
} catch (error) {
    console.error('Failed to initialize bot:', error);
    process.exit(1);
}

// Add error handling middleware
app.use((err: any, _req: Request, res: Response, next: any) => {
    console.error('Server error:', err);
    res.status(500).json({ error: 'Internal server error' });
    next();
});

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// API Routes with error handling
app.get('/api/stats', async (_req: Request, res: Response) => {
    try {
        const stats = await bot.getStats();
        res.json(stats);
    } catch (error) {
        console.error('Error getting stats:', error);
        res.status(500).json({ error: 'Failed to get bot stats' });
    }
});

app.get('/api/history', async (_req: Request, res: Response) => {
    try {
        const history = bot.getArbitrageHistory();
        res.json(history);
    } catch (error) {
        console.error('Error getting history:', error);
        res.status(500).json({ error: 'Failed to get arbitrage history' });
    }
});

app.post('/api/bot/start', async (_req: Request, res: Response) => {
    try {
        await bot.start();
        res.json({ status: 'started' });
    } catch (error) {
        console.error('Error starting bot:', error);
        res.status(500).json({ error: 'Failed to start bot' });
    }
});

app.post('/api/bot/stop', (_req: Request, res: Response) => {
    try {
        bot.stop();
        res.json({ status: 'stopped' });
    } catch (error) {
        console.error('Error stopping bot:', error);
        res.status(500).json({ error: 'Failed to stop bot' });
    }
});

// Blockchain-specific monitoring endpoints
app.get('/api/blockchain/transactions', async (_req: Request, res: Response) => {
    try {
        const transactions = await bot.getPendingTransactions();
        res.json(transactions);
    } catch (error) {
        console.error('Error getting transactions:', error);
        res.status(500).json({ error: 'Failed to get transaction data' });
    }
});

app.get('/api/blockchain/performance', async (_req: Request, res: Response) => {
    try {
        const metrics = await bot.getPerformanceMetrics();
        res.json(metrics);
    } catch (error) {
        console.error('Error getting performance metrics:', error);
        res.status(500).json({ error: 'Failed to get performance data' });
    }
});

app.get('/api/blockchain/events', async (_req: Request, res: Response) => {
    try {
        const events = await bot.getBlockchainEvents();
        res.json(events);
    } catch (error) {
        console.error('Error getting blockchain events:', error);
        res.status(500).json({ error: 'Failed to get event data' });
    }
});

// Serve React app
app.get('*', (_req: Request, res: Response) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Add graceful shutdown
process.on('SIGTERM', () => {
    console.log('Received SIGTERM. Shutting down gracefully...');
    bot.stop();
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('Received SIGINT. Shutting down gracefully...');
    bot.stop();
    process.exit(0);
});

app.listen(port, () => {
    console.log(`Bot monitoring server running on port ${port}`);
}); 