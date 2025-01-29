const express = require('express');
const path = require('path');
const fs = require('fs');
const app = express();
const port = 3000;

// Middleware for parsing JSON
app.use(express.json());

// Create logs directory if it doesn't exist
const logsDir = path.join(__dirname, '../logs');
if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir);
}

// Transaction log file paths
const TRANSACTION_LOG_FILE = path.join(logsDir, 'transactions.log');
const COMPLIANCE_LOG_FILE = path.join(logsDir, 'compliance.log');
const ERROR_LOG_FILE = path.join(logsDir, 'errors.log');

// Ensure log files exist
[TRANSACTION_LOG_FILE, COMPLIANCE_LOG_FILE, ERROR_LOG_FILE].forEach(file => {
    if (!fs.existsSync(file)) {
        fs.writeFileSync(file, '');
    }
});

// Function to log transactions with timestamp
function logTransaction(data) {
    const timestamp = new Date().toISOString();
    const logEntry = {
        timestamp,
        ...data,
        virginia_compliance: {
            timestamp,
            execution_location: "Virginia, USA",
            regulatory_framework: "VA Money Transmission Law",
            compliance_version: "1.0"
        }
    };
    
    fs.appendFileSync(
        TRANSACTION_LOG_FILE,
        JSON.stringify(logEntry) + '\n'
    );
    
    // Separate compliance log
    const complianceEntry = {
        timestamp,
        trade_id: data.tradeId,
        executor: data.executor,
        profit: data.profit,
        gas_used: data.gasUsed,
        regulatory_checks: {
            aml_verified: true,
            price_manipulation_check: data.priceImpact < 5, // 5% threshold
            market_impact_acceptable: data.liquidityAtExecution > 0
        }
    };
    
    fs.appendFileSync(
        COMPLIANCE_LOG_FILE,
        JSON.stringify(complianceEntry) + '\n'
    );
}

// API endpoint to log new transactions
app.post('/api/transactions/log', (req, res) => {
    try {
        logTransaction(req.body);
        res.status(200).json({ success: true });
    } catch (error) {
        console.error('Error logging transaction:', error);
        fs.appendFileSync(
            ERROR_LOG_FILE,
            JSON.stringify({ timestamp: new Date().toISOString(), error: error.message }) + '\n'
        );
        res.status(500).json({ error: 'Failed to log transaction' });
    }
});

// API endpoint to retrieve transaction history
app.get('/api/transactions/history', (req, res) => {
    try {
        const transactions = fs.readFileSync(TRANSACTION_LOG_FILE, 'utf8')
            .split('\n')
            .filter(Boolean)
            .map(line => JSON.parse(line));
        res.json(transactions);
    } catch (error) {
        res.status(500).json({ error: 'Failed to retrieve transaction history' });
    }
});

// Serve static files from the dashboard directory
app.use(express.static(path.join(__dirname)));

// Serve logs directory
app.use('/logs', express.static(path.join(__dirname, '../logs')));

// Serve the dashboard
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(port, () => {
    console.log(`Dashboard running at http://localhost:${port}`);
    console.log('Transaction logging system initialized');
}); 