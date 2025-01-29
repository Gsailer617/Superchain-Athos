import React, { useState, useEffect } from 'react';
import {
    Container,
    Box,
    Typography,
    Paper,
    Grid,
    Button,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    CircularProgress,
    Alert
} from '@mui/material';
import { formatEther, formatUnits } from 'ethers';

interface BotStats {
    profitablePathsFound: number;
    executedTrades: number;
    totalArbitrages: number;
    totalProfitUSD: number;
    isRunning: boolean;
}

interface ArbitrageEvent {
    tokenIn: string;
    tokenOut: string;
    amountIn: bigint;
    amountOut: bigint;
    sourceDex: number;
    targetDex: number;
    profit: bigint;
    timestamp: number;
    transactionHash: string;
}

const App: React.FC = () => {
    const [stats, setStats] = useState<BotStats | null>(null);
    const [history, setHistory] = useState<ArbitrageEvent[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchStats = async () => {
        try {
            const response = await fetch('http://localhost:3000/api/stats');
            const data = await response.json();
            setStats(data);
        } catch (err) {
            setError('Failed to fetch bot stats');
        }
    };

    const fetchHistory = async () => {
        try {
            const response = await fetch('http://localhost:3000/api/history');
            const data = await response.json();
            setHistory(data);
        } catch (err) {
            setError('Failed to fetch arbitrage history');
        }
    };

    const startBot = async () => {
        try {
            await fetch('http://localhost:3000/api/bot/start', { method: 'POST' });
            fetchStats();
        } catch (err) {
            setError('Failed to start bot');
        }
    };

    const stopBot = async () => {
        try {
            await fetch('http://localhost:3000/api/bot/stop', { method: 'POST' });
            fetchStats();
        } catch (err) {
            setError('Failed to stop bot');
        }
    };

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            await Promise.all([fetchStats(), fetchHistory()]);
            setLoading(false);
        };

        fetchData();
        const interval = setInterval(fetchData, 15000); // Refresh every 15 seconds

        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
                <CircularProgress />
            </Box>
        );
    }

    const getDexName = (dex: number): string => {
        const dexNames = ['BaseSwap', 'Aerodrome', 'SushiSwap', 'PancakeSwap', 'UniswapV3'];
        return dexNames[dex] || 'Unknown';
    };

    return (
        <Container maxWidth="lg">
            <Box my={4}>
                <Typography variant="h4" component="h1" gutterBottom>
                    Arbitrage Bot Monitor
                </Typography>

                {error && (
                    <Alert severity="error" sx={{ mb: 2 }}>
                        {error}
                    </Alert>
                )}

                <Grid container spacing={3}>
                    {/* Stats Cards */}
                    <Grid item xs={12} md={3}>
                        <Paper sx={{ p: 2 }}>
                            <Typography variant="h6">Status</Typography>
                            <Typography color={stats?.isRunning ? 'success.main' : 'error.main'}>
                                {stats?.isRunning ? 'Running' : 'Stopped'}
                            </Typography>
                        </Paper>
                    </Grid>
                    <Grid item xs={12} md={3}>
                        <Paper sx={{ p: 2 }}>
                            <Typography variant="h6">Total Profit</Typography>
                            <Typography>${stats?.totalProfitUSD.toFixed(2)}</Typography>
                        </Paper>
                    </Grid>
                    <Grid item xs={12} md={3}>
                        <Paper sx={{ p: 2 }}>
                            <Typography variant="h6">Executed Trades</Typography>
                            <Typography>{stats?.executedTrades}</Typography>
                        </Paper>
                    </Grid>
                    <Grid item xs={12} md={3}>
                        <Paper sx={{ p: 2 }}>
                            <Typography variant="h6">Opportunities Found</Typography>
                            <Typography>{stats?.profitablePathsFound}</Typography>
                        </Paper>
                    </Grid>

                    {/* Control Buttons */}
                    <Grid item xs={12}>
                        <Box display="flex" gap={2}>
                            <Button
                                variant="contained"
                                color="success"
                                onClick={startBot}
                                disabled={stats?.isRunning}
                            >
                                Start Bot
                            </Button>
                            <Button
                                variant="contained"
                                color="error"
                                onClick={stopBot}
                                disabled={!stats?.isRunning}
                            >
                                Stop Bot
                            </Button>
                        </Box>
                    </Grid>

                    {/* Trade History Table */}
                    <Grid item xs={12}>
                        <TableContainer component={Paper}>
                            <Table>
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Time</TableCell>
                                        <TableCell>Path</TableCell>
                                        <TableCell align="right">Amount In</TableCell>
                                        <TableCell align="right">Amount Out</TableCell>
                                        <TableCell align="right">Profit</TableCell>
                                        <TableCell>Transaction</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {history.map((event, index) => (
                                        <TableRow key={index}>
                                            <TableCell>
                                                {new Date(event.timestamp * 1000).toLocaleString()}
                                            </TableCell>
                                            <TableCell>
                                                {`${getDexName(event.sourceDex)} → ${getDexName(event.targetDex)}`}
                                            </TableCell>
                                            <TableCell align="right">
                                                {formatEther(event.amountIn)}
                                            </TableCell>
                                            <TableCell align="right">
                                                {formatEther(event.amountOut)}
                                            </TableCell>
                                            <TableCell align="right">
                                                ${formatUnits(event.profit, 18)}
                                            </TableCell>
                                            <TableCell>
                                                <a
                                                    href={`https://basescan.org/tx/${event.transactionHash}`}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                >
                                                    {`${event.transactionHash.slice(0, 6)}...${event.transactionHash.slice(-4)}`}
                                                </a>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </Grid>
                </Grid>
            </Box>
        </Container>
    );
};

export default App; 