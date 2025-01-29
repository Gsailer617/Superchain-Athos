class MonitoringSystem {
    constructor() {
        this.alertHistory = [];
        this.priceAlerts = new Set();
        this.volumeAlerts = new Set();
        this.patternAlerts = new Set();
        this.volatilityAlerts = new Set();
        this.customAlerts = new Map();
        this.monitoringIntervals = new Map();
        this.activeIndicators = new Set();
        this.realtimeMetrics = new Map();
        this.errorLogs = [];
        this.maxErrorLogs = 1000;
        this.alertId = 0;
        this.blockchainMetrics = {
            pendingTransactions: [],
            confirmedTransactions: [],
            gasStats: {
                average: 0,
                max: 0,
                min: 0
            },
            profitStats: {
                total: 0,
                average: 0,
                lastProfit: 0
            },
            networkStats: {
                blockNumber: 0,
                timestamp: 0
            }
        };
        
        // Start blockchain monitoring
        this.startBlockchainMonitoring();
    }

    async acknowledgeAlert(alertId) {
        const alert = this.alertHistory.find(a => a.id === alertId);
        if (!alert) return;

        try {
            // Send acknowledgment to backend
            await fetch('/api/alerts/acknowledge', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ alertId })
            });

            alert.acknowledged = true;
            this.notifyAlertUpdate(alert);
        } catch (error) {
            console.error('Failed to acknowledge alert:', error);
            throw error;
        }
    }

    logError(error) {
        const errorLog = {
            timestamp: Date.now(),
            error: error.error || error.message || 'Unknown error',
            stack: error.stack,
            component: error.component || 'Unknown',
            metadata: error.metadata || {}
        };

        this.errorLogs.unshift(errorLog);

        // Keep error logs under limit
        if (this.errorLogs.length > this.maxErrorLogs) {
            this.errorLogs.pop();
        }

        // Create alert for critical errors
        if (this.isCriticalError(error)) {
            this.createAlert({
                type: 'ERROR',
                severity: 'high',
                message: `Critical error in ${error.component}: ${error.error || error.message}`,
                metadata: error.metadata,
                actionable: true
            });
        }
    }

    isCriticalError(error) {
        const criticalKeywords = [
            'crash',
            'fatal',
            'critical',
            'connection lost',
            'database error',
            'out of memory'
        ];

        const errorText = (error.error || error.message || '').toLowerCase();
        return criticalKeywords.some(keyword => errorText.includes(keyword));
    }

    createAlert(alertData) {
        const alert = {
            id: this.generateAlertId(),
            timestamp: Date.now(),
            acknowledged: false,
            ...alertData
        };

        this.alertHistory.unshift(alert);
        this.notifyAlertUpdate(alert);

        return alert;
    }

    generateAlertId() {
        return `alert_${Date.now()}_${this.alertId++}`;
    }

    notifyAlertUpdate(alert) {
        // Dispatch event for UI updates
        window.dispatchEvent(new CustomEvent('alertUpdate', {
            detail: { alert }
        }));

        // Send to backend if needed
        this.syncWithBackend(alert);
    }

    async syncWithBackend(alert) {
        try {
            await fetch('/api/alerts/sync', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ alert })
            });
        } catch (error) {
            console.error('Failed to sync alert with backend:', error);
        }
    }

    startMonitoring(type, config) {
        if (this.monitoringIntervals.has(type)) {
            this.stopMonitoring(type);
        }

        const interval = setInterval(() => {
            this.checkConditions(type, config);
        }, config.interval || 5000);

        this.monitoringIntervals.set(type, interval);
    }

    stopMonitoring(type) {
        const interval = this.monitoringIntervals.get(type);
        if (interval) {
            clearInterval(interval);
            this.monitoringIntervals.delete(type);
        }
    }

    async checkConditions(type, config) {
        try {
            const data = await this.fetchMetrics(type);
            const conditions = this.evaluateConditions(type, data, config);

            if (conditions.length > 0) {
                conditions.forEach(condition => {
                    this.createAlert({
                        type,
                        severity: condition.severity,
                        message: condition.message,
                        metadata: condition.metadata,
                        actionable: condition.actionable
                    });
                });
            }
        } catch (error) {
            console.error(`Error checking conditions for ${type}:`, error);
            this.logError({
                error,
                component: 'Monitoring',
                metadata: { type, config }
            });
        }
    }

    async fetchMetrics(type) {
        const response = await fetch(`/api/metrics/${type}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch metrics for ${type}`);
        }
        return response.json();
    }

    evaluateConditions(type, data, config) {
        const conditions = [];

        switch (type) {
            case 'PRICE':
                this.evaluatePriceConditions(data, config, conditions);
                break;
            case 'VOLUME':
                this.evaluateVolumeConditions(data, config, conditions);
                break;
            case 'PATTERN':
                this.evaluatePatternConditions(data, config, conditions);
                break;
            case 'VOLATILITY':
                this.evaluateVolatilityConditions(data, config, conditions);
                break;
            default:
                if (this.customAlerts.has(type)) {
                    this.evaluateCustomConditions(type, data, config, conditions);
                }
        }

        return conditions;
    }

    evaluatePriceConditions(data, config, conditions) {
        const { currentPrice, previousPrice, threshold } = data;
        const priceChange = (currentPrice - previousPrice) / previousPrice * 100;

        if (Math.abs(priceChange) > threshold) {
            conditions.push({
                severity: priceChange > 0 ? 'info' : 'warning',
                message: `Price changed by ${priceChange.toFixed(2)}%`,
                metadata: { priceChange, currentPrice, previousPrice },
                actionable: Math.abs(priceChange) > threshold * 2
            });
        }
    }

    evaluateVolumeConditions(data, config, conditions) {
        const { currentVolume, averageVolume, threshold } = data;
        const volumeChange = (currentVolume - averageVolume) / averageVolume * 100;

        if (Math.abs(volumeChange) > threshold) {
            conditions.push({
                severity: 'warning',
                message: `Volume spike detected: ${volumeChange.toFixed(2)}% change`,
                metadata: { volumeChange, currentVolume, averageVolume },
                actionable: true
            });
        }
    }

    evaluatePatternConditions(data, config, conditions) {
        const { patterns } = data;
        
        patterns.forEach(pattern => {
            if (pattern.confidence > config.confidenceThreshold) {
                conditions.push({
                    severity: pattern.type === 'bullish' ? 'info' : 'warning',
                    message: `${pattern.name} pattern detected with ${pattern.confidence}% confidence`,
                    metadata: pattern,
                    actionable: pattern.confidence > 80
                });
            }
        });
    }

    evaluateVolatilityConditions(data, config, conditions) {
        const { currentVolatility, averageVolatility, threshold } = data;
        const volatilityChange = (currentVolatility - averageVolatility) / averageVolatility * 100;

        if (volatilityChange > threshold) {
            conditions.push({
                severity: 'high',
                message: `High volatility detected: ${volatilityChange.toFixed(2)}% increase`,
                metadata: { volatilityChange, currentVolatility, averageVolatility },
                actionable: true
            });
        }
    }

    evaluateCustomConditions(type, data, config, conditions) {
        const customEvaluator = this.customAlerts.get(type);
        if (customEvaluator) {
            const customConditions = customEvaluator(data, config);
            conditions.push(...customConditions);
        }
    }

    registerCustomAlert(type, evaluator) {
        this.customAlerts.set(type, evaluator);
    }

    getActiveAlerts() {
        return this.alertHistory.filter(alert => !alert.acknowledged);
    }

    getAlertsByType(type) {
        return this.alertHistory.filter(alert => alert.type === type);
    }

    getAlertsBySeverity(severity) {
        return this.alertHistory.filter(alert => alert.severity === severity);
    }

    getRecentErrors(minutes = 60) {
        const cutoff = Date.now() - minutes * 60 * 1000;
        return this.errorLogs.filter(log => log.timestamp > cutoff);
    }

    async startBlockchainMonitoring() {
        // Update blockchain metrics every 10 seconds
        setInterval(async () => {
            try {
                await this.updateBlockchainMetrics();
            } catch (error) {
                this.logError({
                    error,
                    component: 'BlockchainMonitoring',
                    metadata: { timestamp: Date.now() }
                });
            }
        }, 10000);
    }

    async updateBlockchainMetrics() {
        try {
            // Fetch blockchain data from our API endpoints
            const [transactions, performance, events] = await Promise.all([
                fetch('/api/blockchain/transactions').then(res => res.json()),
                fetch('/api/blockchain/performance').then(res => res.json()),
                fetch('/api/blockchain/events').then(res => res.json())
            ]);

            // Update metrics
            this.blockchainMetrics = {
                ...this.blockchainMetrics,
                pendingTransactions: transactions.pending || [],
                confirmedTransactions: transactions.confirmed || [],
                gasStats: performance.gas || this.blockchainMetrics.gasStats,
                profitStats: performance.profit || this.blockchainMetrics.profitStats,
                networkStats: events.network || this.blockchainMetrics.networkStats
            };

            // Notify UI of updates
            this.notifyBlockchainUpdate();

            // Create alerts for significant events
            this.checkBlockchainAlerts();
        } catch (error) {
            throw new Error(`Failed to update blockchain metrics: ${error.message}`);
        }
    }

    notifyBlockchainUpdate() {
        window.dispatchEvent(new CustomEvent('blockchainUpdate', {
            detail: { metrics: this.blockchainMetrics }
        }));
    }

    checkBlockchainAlerts() {
        const { profitStats, gasStats } = this.blockchainMetrics;

        // Alert on significant profit
        if (profitStats.lastProfit > 0) {
            this.createAlert({
                type: 'PROFIT',
                severity: 'info',
                message: `New profit recorded: ${profitStats.lastProfit} ETH`,
                metadata: { profit: profitStats.lastProfit },
                actionable: false
            });
        }

        // Alert on high gas prices
        if (gasStats.average > 100) { // Assuming 100 gwei as threshold
            this.createAlert({
                type: 'GAS',
                severity: 'warning',
                message: `High gas prices detected: ${gasStats.average} gwei`,
                metadata: gasStats,
                actionable: true
            });
        }
    }

    getBlockchainMetrics() {
        return this.blockchainMetrics;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MonitoringSystem;
} 