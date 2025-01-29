// Dashboard data management
class DashboardManager {
    constructor() {
        this.charts = {
            profit: null,
            volume: null,
            gas: null,
            performance: null,
            liquidity: null,
            slippage: null,
            profitDistribution: null,
            tokenPairCorrelation: null,
            marketDepth: null,
            trendPrediction: null,
            anomalyDetection: null,
            pairCorrelation: null,
            volatilityHeatmap: null
        };
        this.selectedTimeframe = '24h';
        this.selectedPair = 'all';
        this.selectedMetrics = new Set(['profit', 'volume', 'gas']);
        this.autoRefresh = true;
        this.refreshRate = 15000;
        
        // Add new state variables
        this.customDateRange = { start: null, end: null };
        this.selectedAnalytics = new Set(['trends', 'anomalies']);
        this.drilldownState = null;
        this.exportFormat = 'json';
        this.predictionWindow = '24h';
        
        // Add new state variables for advanced features
        this.alertSettings = {
            profitThreshold: 5.0,
            lossThreshold: -2.0,
            volatilityThreshold: 20,
            anomalyThreshold: 0.8,
            enableNotifications: true
        };
        
        this.patternSettings = {
            minPatternLength: 3,
            maxPatternLength: 10,
            similarityThreshold: 0.85,
            patterns: new Map()
        };

        this.statisticalMetrics = {
            movingAverages: new Map(),
            standardDeviations: new Map(),
            correlationMatrix: new Map(),
            regressionModels: new Map()
        };
        
        // Add technical indicators
        this.technicalIndicators = {
            macd: {
                fastPeriod: 12,
                slowPeriod: 26,
                signalPeriod: 9,
                values: new Map()
            },
            rsi: {
                period: 14,
                overbought: 70,
                oversold: 30,
                values: new Map()
            },
            bollingerBands: {
                period: 20,
                stdDev: 2,
                values: new Map()
            },
            stochastic: {
                kPeriod: 14,
                dPeriod: 3,
                values: new Map()
            },
            atr: {
                period: 14,
                values: new Map()
            },
            ichimoku: {
                conversionPeriod: 9,
                basePeriod: 26,
                spanPeriod: 52,
                displacement: 26,
                values: new Map()
            },
            keltnerChannels: {
                emaPeriod: 20,
                atrPeriod: 10,
                multiplier: 2,
                values: new Map()
            },
            vwap: {
                period: 14,
                values: new Map()
            },
            cmf: {
                period: 20,
                values: new Map()
            },
            dmi: {
                period: 14,
                values: new Map()
            },
            supertrend: {
                period: 10,
                multiplier: 3,
                values: new Map()
            },
            zigzag: {
                deviation: 5,
                values: new Map()
            },
            williamsR: {
                period: 14,
                values: new Map()
            },
            ultimateOscillator: {
                period1: 7,
                period2: 14,
                period3: 28,
                values: new Map()
            },
            aroon: {
                period: 25,
                values: new Map()
            },
            volumeProfile: {
                levels: 12,
                values: new Map()
            },
            marketProfile: {
                timeframe: '30m',
                values: new Map()
            }
        };

        // Add ML model states
        this.mlModels = {
            patternRecognition: null,
            anomalyDetection: null,
            trendPrediction: null,
            sentimentAnalysis: null,
            lstm: {
                pricePredictor: null,
                volumePredictor: null,
                volatilityPredictor: null
            },
            transformer: {
                multivariate: null,
                attention: null
            },
            ensemble: {
                models: [],
                weights: []
            },
            gru: {
                sequenceModel: null,
                attentionModel: null
            },
            wavenet: {
                dilatedModel: null,
                residualModel: null
            },
            transformerXL: {
                model: null,
                memoryLength: 512
            },
            autoencoder: {
                encoder: null,
                decoder: null,
                latentDim: 32
            },
            hybridModels: {
                lstmTransformer: null,
                cnnLstm: null,
                transformerGru: null
            }
        };

        // Real-time Monitoring
        this.monitoringSystem = new MonitoringSystem();
        this.alertHistory = [];
        this.initializeEventListeners();
        this.initializeWebSocket();
        this.updateInterval = null;
        this.retryCount = 0;
        this.maxRetries = 3;
        
        this.initializeCharts();
        this.setupWebSocket();
        this.updateInterval = setInterval(() => this.fetchUpdates(), this.refreshRate);
        this.setupEventListeners();
        
        // Initialize notification system
        this.initializeNotifications();
        
        // Start real-time monitoring
        this.startRealTimeMonitoring();
        
        // Initialize ML models
        this.initializeMLModels();

        // Initialize advanced features
        this.initializeAdvancedModels();
        this.initializeEnhancedModels();
        this.setupAdvancedMonitoring();

        this.initializeBotLearning();

        this.initializePoolMonitoring();
    }

    initializeEventListeners() {
        document.addEventListener('DOMContentLoaded', () => {
            this.setupAlertHandlers();
            this.setupChartUpdates();
            this.setupParameterControls();
            this.startPeriodicUpdates();
        });

        window.addEventListener('error', (event) => {
            this.handleError(event.error);
        });

        window.addEventListener('unhandledrejection', (event) => {
            this.handleError(event.reason);
        });
    }

    handleError(error) {
        console.error('Dashboard error:', error);
        this.showNotification('An error occurred. Please refresh the page.', 'error');
        
        // Log error to monitoring system
        this.monitoringSystem.logError({
            timestamp: Date.now(),
            error: error.message || 'Unknown error',
            stack: error.stack,
            component: 'Dashboard'
        });
    }

    async updateAlertUI() {
        try {
            const container = document.getElementById('alerts-container');
            if (!container) return;

            const activeAlerts = this.monitoringSystem.alertHistory.filter(a => !a.acknowledged);
            
            // Update alert count badge with animation
            const countBadge = document.getElementById('alert-count');
            if (countBadge) {
                const oldCount = parseInt(countBadge.textContent);
                const newCount = activeAlerts.length;
                
                countBadge.textContent = newCount;
                
                if (newCount > oldCount) {
                    countBadge.classList.add('pulse');
                    setTimeout(() => countBadge.classList.remove('pulse'), 1000);
                }
            }
            
            // Update alerts panel with smooth transitions
            const alertsList = document.getElementById('alerts-list');
            if (alertsList) {
                const newAlerts = activeAlerts
                    .map(alert => this.createAlertListItem(alert))
                    .join('');
                
                if (newAlerts !== alertsList.innerHTML) {
                    alertsList.style.opacity = '0';
                    setTimeout(() => {
                        alertsList.innerHTML = newAlerts;
                        alertsList.style.opacity = '1';
                    }, 300);
                }
            }
        } catch (error) {
            console.error('Error updating alert UI:', error);
            this.handleError(error);
        }
    }

    createAlertListItem(alert) {
        const severityClass = this.getAlertSeverityClass(alert);
        const timeAgo = this.formatTimeAgo(alert.timestamp);
        
        return `
            <div class="alert-item ${severityClass} fade-in" data-alert-id="${alert.id}">
                <div class="alert-header">
                    <span class="alert-type">${alert.type}</span>
                    <span class="alert-time">${timeAgo}</span>
                </div>
                <div class="alert-content">
                    <p>${this.escapeHtml(alert.message)}</p>
                    ${alert.recommendation ? `<p class="alert-recommendation">${this.escapeHtml(alert.recommendation)}</p>` : ''}
                </div>
                <div class="alert-actions">
                    <button class="btn-acknowledge" onclick="dashboard.acknowledgeAlert('${alert.id}')">
                        Acknowledge
                    </button>
                    ${alert.actionable ? `
                        <button class="btn-action" onclick="dashboard.handleAlertAction('${alert.id}')">
                            Take Action
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    }

    escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    formatTimeAgo(timestamp) {
        const seconds = Math.floor((Date.now() - timestamp) / 1000);
        
        if (seconds < 60) return `${seconds}s ago`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        return `${Math.floor(seconds / 86400)}d ago`;
    }

    async acknowledgeAlert(alertId) {
        try {
            const alert = this.monitoringSystem.alertHistory.find(a => a.id === alertId);
            if (!alert) return;
            
            await this.monitoringSystem.acknowledgeAlert(alertId);
            alert.acknowledged = true;
            
            // Animate alert removal
            const alertElement = document.querySelector(`[data-alert-id="${alertId}"]`);
            if (alertElement) {
                alertElement.classList.add('fade-out');
                setTimeout(() => {
                    this.updateAlertUI();
                }, 300);
            }
        } catch (error) {
            console.error('Error acknowledging alert:', error);
            this.showNotification('Failed to acknowledge alert', 'error');
        }
    }

    startPeriodicUpdates() {
        this.updateInterval = setInterval(() => {
            this.updateDashboard().catch(error => {
                this.retryCount++;
                if (this.retryCount >= this.maxRetries) {
                    clearInterval(this.updateInterval);
                    this.handleError(new Error('Dashboard updates failed repeatedly'));
                }
            });
        }, 5000);
    }

    async updateDashboard() {
        try {
            await Promise.all([
                this.updateMetrics(),
                this.updateCharts(),
                this.updateAlertUI()
            ]);
            this.retryCount = 0; // Reset retry count on successful update
        } catch (error) {
            throw error;
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Trigger animation
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Remove after delay
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }

    setupEventListeners() {
        document.getElementById('timeframe-selector')?.addEventListener('change', (e) => {
            this.updateTimeframe(e.target.value);
        });
        document.getElementById('pair-filter')?.addEventListener('change', (e) => {
            this.updatePairFilter(e.target.value);
        });
        document.getElementById('refresh-rate')?.addEventListener('change', (e) => {
            this.updateRefreshRate(parseInt(e.target.value));
        });
        document.getElementById('auto-refresh')?.addEventListener('change', (e) => {
            this.autoRefresh = e.target.checked;
            if (this.autoRefresh) {
                this.updateInterval = setInterval(() => this.fetchUpdates(), this.refreshRate);
            } else {
                clearInterval(this.updateInterval);
            }
        });
        document.getElementById('metric-selector')?.addEventListener('change', (e) => {
            const metrics = Array.from(e.target.selectedOptions).map(option => option.value);
            this.selectedMetrics = new Set(metrics);
            this.updateVisibleCharts();
        });
        document.getElementById('export-data')?.addEventListener('click', () => {
            this.exportDashboardData();
        });

        // Add date range picker listeners
        document.getElementById('date-range-start')?.addEventListener('change', (e) => {
            this.customDateRange.start = new Date(e.target.value);
            this.fetchUpdates();
        });

        document.getElementById('date-range-end')?.addEventListener('change', (e) => {
            this.customDateRange.end = new Date(e.target.value);
            this.fetchUpdates();
        });

        // Add prediction window selector
        document.getElementById('prediction-window')?.addEventListener('change', (e) => {
            this.predictionWindow = e.target.value;
            this.updatePredictions();
        });

        // Add export format selector
        document.getElementById('export-format')?.addEventListener('change', (e) => {
            this.exportFormat = e.target.value;
        });

        // Add drill-down listeners to charts
        Object.values(this.charts).forEach(chart => {
            if (chart?.canvas) {
                chart.canvas.addEventListener('click', (e) => {
                    const points = chart.getElementsAtEventForMode(e, 'nearest', { intersect: true }, true);
                    if (points.length) {
                        this.handleDrilldown(chart, points[0]);
                    }
                });
            }
        });
    }

    initializeCharts() {
        // Profit History Chart
        const profitCtx = document.getElementById('profit-chart').getContext('2d');
        this.charts.profit = new Chart(profitCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Hourly Profit (USD)',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Profit (USD)' }
                    },
                    x: {
                        title: { display: true, text: 'Time' }
                    }
                }
            }
        });

        // Volume Analysis Chart
        const volumeCtx = document.getElementById('volume-chart').getContext('2d');
        this.charts.volume = new Chart(volumeCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Trading Volume',
                    data: [],
                    backgroundColor: 'rgba(54, 162, 235, 0.5)'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Volume (USD)' }
                    }
                }
            }
        });

        // Gas Usage Chart
        const gasCtx = document.getElementById('gas-chart').getContext('2d');
        this.charts.gas = new Chart(gasCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Gas Cost (GWEI)',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    fill: false
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        title: { display: true, text: 'Gas (GWEI)' }
                    }
                }
            }
        });

        // Performance Metrics Chart
        const perfCtx = document.getElementById('performance-chart').getContext('2d');
        this.charts.performance = new Chart(perfCtx, {
            type: 'radar',
            data: {
                labels: ['Success Rate', 'ROI', 'Gas Efficiency', 'Speed', 'Risk Score'],
                datasets: [{
                    label: 'Current Performance',
                    data: [],
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgb(75, 192, 192)',
                    pointBackgroundColor: 'rgb(75, 192, 192)'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });

        // Liquidity Distribution Chart
        const liquidityCtx = document.getElementById('liquidity-chart').getContext('2d');
        this.charts.liquidity = new Chart(liquidityCtx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.5)',
                        'rgba(54, 162, 235, 0.5)',
                        'rgba(255, 206, 86, 0.5)',
                        'rgba(75, 192, 192, 0.5)'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'right'
                    }
                }
            }
        });

        // Slippage Analysis Chart
        const slippageCtx = document.getElementById('slippage-chart').getContext('2d');
        this.charts.slippage = new Chart(slippageCtx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Trade Size vs Slippage',
                    data: [],
                    backgroundColor: 'rgba(75, 192, 192, 0.5)'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: { display: true, text: 'Trade Size (USD)' }
                    },
                    y: {
                        title: { display: true, text: 'Slippage %' }
                    }
                }
            }
        });

        // Add Profit Distribution Chart
        const profitDistCtx = document.getElementById('profit-distribution-chart').getContext('2d');
        this.charts.profitDistribution = new Chart(profitDistCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Profit Distribution',
                    data: [],
                    backgroundColor: 'rgba(153, 102, 255, 0.5)'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        title: { display: true, text: 'Frequency' }
                    },
                    x: {
                        title: { display: true, text: 'Profit Range (USD)' }
                    }
                }
            }
        });

        // Add Token Pair Correlation Matrix
        const correlationCtx = document.getElementById('correlation-chart').getContext('2d');
        this.charts.tokenPairCorrelation = new Chart(correlationCtx, {
            type: 'heatmap',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: (context) => {
                        const value = context.dataset.data[context.dataIndex];
                        const alpha = (value + 1) / 2;
                        return `rgba(75, 192, 192, ${alpha})`;
                    }
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const value = context.dataset.data[context.dataIndex];
                                return `Correlation: ${value.toFixed(2)}`;
                            }
                        }
                    }
                }
            }
        });

        // Add Market Depth Chart
        const marketDepthCtx = document.getElementById('market-depth-chart').getContext('2d');
        this.charts.marketDepth = new Chart(marketDepthCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Bids',
                        data: [],
                        borderColor: 'rgba(75, 192, 192, 1)',
                        fill: true,
                        backgroundColor: 'rgba(75, 192, 192, 0.2)'
                    },
                    {
                        label: 'Asks',
                        data: [],
                        borderColor: 'rgba(255, 99, 132, 1)',
                        fill: true,
                        backgroundColor: 'rgba(255, 99, 132, 0.2)'
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: { display: true, text: 'Price' }
                    },
                    y: {
                        title: { display: true, text: 'Cumulative Volume' }
                    }
                }
            }
        });

        // Add Trend Prediction Chart
        const trendCtx = document.getElementById('trend-prediction-chart').getContext('2d');
        this.charts.trendPrediction = new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Historical',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        fill: false
                    },
                    {
                        label: 'Predicted',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        borderDash: [5, 5],
                        fill: false
                    },
                    {
                        label: 'Confidence Interval',
                        data: [],
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderColor: 'transparent',
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const confidence = context.dataset.confidenceInterval;
                                return confidence ? 
                                    `${context.parsed.y} (±${confidence}%)` : 
                                    `${context.parsed.y}`;
                            }
                        }
                    }
                }
            }
        });

        // Add Anomaly Detection Chart
        const anomalyCtx = document.getElementById('anomaly-detection-chart').getContext('2d');
        this.charts.anomalyDetection = new Chart(anomalyCtx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Normal',
                        data: [],
                        backgroundColor: 'rgba(75, 192, 192, 0.5)'
                    },
                    {
                        label: 'Anomalies',
                        data: [],
                        backgroundColor: 'rgba(255, 99, 132, 0.5)'
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const point = context.raw;
                                return `${point.x}: ${point.y} (Score: ${point.anomalyScore})`;
                            }
                        }
                    }
                }
            }
        });
    }

    setupWebSocket() {
        this.ws = new WebSocket('ws://localhost:3000/dashboard');
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleUpdate(data);
        };

        this.ws.onclose = () => {
            console.log('WebSocket connection closed. Attempting to reconnect...');
            setTimeout(() => this.setupWebSocket(), 5000);
        };
    }

    async fetchUpdates() {
        try {
            const response = await fetch('/api/dashboard/metrics');
            const data = await response.json();
            this.handleUpdate(data);
        } catch (error) {
            console.error('Failed to fetch updates:', error);
        }
    }

    handleUpdate(data) {
        this.updateOverviewCards(data.overview);
        this.updateCharts(data.charts);
        this.updatePairsTable(data.pairs);
        this.updatePerformanceMetrics(data.performance);
        this.updateRiskMetrics(data.risk);
        this.updateMEVProtection(data.mev);
    }

    updateOverviewCards(overview) {
        // Basic metrics
        document.getElementById('total-profit').textContent = 
            overview.totalProfit.toFixed(2);
        document.getElementById('success-rate').textContent = 
            (overview.successRate * 100).toFixed(1);
        document.getElementById('active-pairs').textContent = 
            overview.activePairs;
        document.getElementById('gas-efficiency').textContent = 
            overview.gasEfficiency.toFixed(3);

        // Advanced metrics
        document.getElementById('roi').textContent = 
            (overview.roi * 100).toFixed(2) + '%';
        document.getElementById('avg-execution-time').textContent = 
            overview.avgExecutionTime.toFixed(2) + 'ms';
        document.getElementById('total-volume').textContent = 
            overview.totalVolume.toFixed(2);
        document.getElementById('mev-savings').textContent = 
            overview.mevSavings.toFixed(2);
    }

    updateCharts(chartData) {
        // Update Profit Chart
        this.charts.profit.data.labels = chartData.profit.labels;
        this.charts.profit.data.datasets[0].data = chartData.profit.values;
        this.charts.profit.update();

        // Update Volume Chart
        this.charts.volume.data.labels = chartData.volume.labels;
        this.charts.volume.data.datasets[0].data = chartData.volume.values;
        this.charts.volume.update();

        // Update Gas Chart
        this.charts.gas.data.labels = chartData.gas.labels;
        this.charts.gas.data.datasets[0].data = chartData.gas.values;
        this.charts.gas.update();

        // Update Performance Chart
        this.charts.performance.data.datasets[0].data = [
            chartData.performance.successRate,
            chartData.performance.roi,
            chartData.performance.gasEfficiency,
            chartData.performance.speed,
            chartData.performance.riskScore
        ];
        this.charts.performance.update();

        // Update Liquidity Chart
        this.charts.liquidity.data.labels = chartData.liquidity.labels;
        this.charts.liquidity.data.datasets[0].data = chartData.liquidity.values;
        this.charts.liquidity.update();

        // Update Slippage Chart
        this.charts.slippage.data.datasets[0].data = chartData.slippage.points;
        this.charts.slippage.update();
    }

    updatePairsTable(pairs) {
        const tbody = document.getElementById('pairs-table-body');
        tbody.innerHTML = '';

        pairs.forEach(pair => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    ${pair.name}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    $${pair.volume.toFixed(2)}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    ${(pair.successRate * 100).toFixed(1)}%
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    $${pair.avgProfit.toFixed(2)}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    ${pair.currentBoost.toFixed(2)}x
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    ${pair.liquidityScore.toFixed(1)}/100
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    ${pair.volatility.toFixed(2)}%
                </td>
            `;
            tbody.appendChild(row);
        });
    }

    updatePerformanceMetrics(performance) {
        // Update performance indicators
        document.getElementById('execution-success').textContent = 
            (performance.executionSuccess * 100).toFixed(1) + '%';
        document.getElementById('avg-profit-per-trade').textContent = 
            '$' + performance.avgProfitPerTrade.toFixed(2);
        document.getElementById('profit-consistency').textContent = 
            performance.profitConsistency.toFixed(2);
        document.getElementById('sharpe-ratio').textContent = 
            performance.sharpeRatio.toFixed(2);
    }

    updateRiskMetrics(risk) {
        // Update risk indicators
        document.getElementById('current-risk-level').textContent = 
            risk.currentLevel;
        document.getElementById('risk-score').textContent = 
            risk.score.toFixed(1) + '/100';
        document.getElementById('max-drawdown').textContent = 
            (risk.maxDrawdown * 100).toFixed(2) + '%';
        document.getElementById('volatility-index').textContent = 
            risk.volatilityIndex.toFixed(2);
    }

    updateMEVProtection(mev) {
        // Update MEV protection metrics
        document.getElementById('mev-attempts-blocked').textContent = 
            mev.attemptsBlocked;
        document.getElementById('mev-savings-total').textContent = 
            '$' + mev.savingsTotal.toFixed(2);
        document.getElementById('current-mev-risk').textContent = 
            mev.currentRisk + ' (' + (mev.riskScore * 100).toFixed(1) + '%)';
        document.getElementById('sandwich-protection').textContent = 
            (mev.sandwichProtection * 100).toFixed(1) + '%';
    }

    updateTimeframe(timeframe) {
        this.selectedTimeframe = timeframe;
        this.fetchUpdates();
    }

    updatePairFilter(pair) {
        this.selectedPair = pair;
        this.fetchUpdates();
    }

    updateRefreshRate(rate) {
        this.refreshRate = rate;
        clearInterval(this.updateInterval);
        if (this.autoRefresh) {
            this.updateInterval = setInterval(() => this.fetchUpdates(), this.refreshRate);
        }
    }

    updateVisibleCharts() {
        Object.entries(this.charts).forEach(([chartType, chart]) => {
            const container = chart.canvas.parentElement.parentElement;
            container.style.display = this.selectedMetrics.has(chartType) ? 'block' : 'none';
        });
    }

    async handleDrilldown(chart, point) {
        const datasetIndex = point.datasetIndex;
        const index = point.index;
        
        // Get detailed data for the selected point
        const detailedData = await this.fetchDetailedData(chart, index);
        
        // Prepare drill-down content
        const content = this.prepareDrilldownContent(detailedData);
        
        // Show drill-down modal with advanced analysis
        this.showDrilldownModal(content);
    }

    async updatePredictions() {
        const predictions = await this.fetchPredictions(this.predictionWindow);
        this.updateTrendPredictionChart(predictions);
    }

    updateTrendPredictionChart(predictions) {
        const chart = this.charts.trendPrediction;
        chart.data.labels = [...predictions.historical.timestamps, ...predictions.forecast.timestamps];
        chart.data.datasets[0].data = predictions.historical.values;
        chart.data.datasets[1].data = Array(predictions.historical.values.length).fill(null).concat(predictions.forecast.values);
        chart.data.datasets[2].data = predictions.forecast.confidenceIntervals;
        chart.update();
    }

    async exportDashboardData() {
        const data = await this.gatherExportData();
        
        let exportData;
        let mimeType;
        let extension;
        
        switch (this.exportFormat) {
            case 'csv':
                exportData = this.convertToCSV(data);
                mimeType = 'text/csv';
                extension = 'csv';
                break;
            case 'excel':
                exportData = this.convertToExcel(data);
                mimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
                extension = 'xlsx';
                break;
            default:
                exportData = JSON.stringify(data, null, 2);
                mimeType = 'application/json';
                extension = 'json';
        }

        const blob = new Blob([exportData], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dashboard-export-${new Date().toISOString()}.${extension}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    async gatherExportData() {
        return {
            metadata: {
                timestamp: new Date().toISOString(),
                timeframe: this.selectedTimeframe,
                customRange: this.customDateRange,
                filters: {
                    pair: this.selectedPair,
                    metrics: Array.from(this.selectedMetrics)
                }
            },
            overview: await this.fetchOverviewData(),
            charts: await this.fetchChartData(),
            performance: await this.fetchPerformanceData(),
            risk: await this.fetchRiskData(),
            mev: await this.fetchMEVData(),
            predictions: await this.fetchPredictions(this.predictionWindow),
            anomalies: await this.fetchAnomalyData(),
            analytics: {
                trends: await this.fetchTrendAnalysis(),
                correlations: await this.fetchCorrelationData()
            }
        };
    }

    convertToCSV(data) {
        // Implementation of JSON to CSV conversion
        const rows = [];
        const headers = this.extractHeaders(data);
        
        rows.push(headers.join(','));
        this.extractRows(data).forEach(row => {
            rows.push(row.join(','));
        });
        
        return rows.join('\n');
    }

    convertToExcel(data) {
        // Implementation of JSON to Excel conversion using a library like XLSX
        // This would require adding the XLSX library to the project
        return XLSX.write(this.prepareExcelWorkbook(data), { type: 'binary' });
    }

    async initializeNotifications() {
        if (!("Notification" in window)) {
            console.warn("This browser does not support desktop notifications");
            return;
        }

        try {
            const permission = await Notification.requestPermission();
            this.alertSettings.enableNotifications = permission === "granted";
        } catch (error) {
            console.error("Failed to initialize notifications:", error);
        }
    }

    startRealTimeMonitoring() {
        // Monitor for anomalies every 5 seconds
        setInterval(() => this.checkForAnomalies(), 5000);
        
        // Update statistical metrics every minute
        setInterval(() => this.updateStatisticalMetrics(), 60000);
        
        // Scan for new patterns every 2 minutes
        setInterval(() => this.scanForNewPatterns(), 120000);
    }

    async checkForAnomalies() {
        const currentMetrics = await this.fetchLatestMetrics();
        const anomalies = this.detectAnomalies(currentMetrics);
        
        if (anomalies.length > 0) {
            this.handleAnomalies(anomalies);
        }
    }

    detectAnomalies(metrics) {
        const anomalies = [];
        
        // Check for profit/loss anomalies
        if (Math.abs(metrics.profit) > this.alertSettings.profitThreshold) {
            anomalies.push({
                type: 'profit',
                value: metrics.profit,
                timestamp: new Date(),
                severity: this.calculateSeverity(metrics.profit)
            });
        }

        // Check for volatility anomalies
        if (metrics.volatility > this.alertSettings.volatilityThreshold) {
            anomalies.push({
                type: 'volatility',
                value: metrics.volatility,
                timestamp: new Date(),
                severity: 'high'
            });
        }

        // Check for pattern-based anomalies
        const patternAnomalies = this.detectPatternAnomalies(metrics);
        anomalies.push(...patternAnomalies);

        return anomalies;
    }

    handleAnomalies(anomalies) {
        anomalies.forEach(anomaly => {
            // Update UI
            this.updateAnomalyChart(anomaly);
            
            // Show notification
            if (this.alertSettings.enableNotifications) {
                this.showNotification(anomaly);
            }

            // Log to analytics
            this.logAnomalyEvent(anomaly);
        });
    }

    showNotification(anomaly) {
        const notification = new Notification("Trading Anomaly Detected", {
            body: `${anomaly.type} anomaly detected: ${anomaly.value}`,
            icon: "/icons/alert.png",
            tag: `anomaly-${anomaly.type}`,
            data: anomaly
        });

        notification.onclick = () => {
            window.focus();
            this.showAnomalyDetails(anomaly);
        };
    }

    async updateStatisticalMetrics() {
        const data = await this.fetchHistoricalData();
        
        // Calculate moving averages
        this.statisticalMetrics.movingAverages = this.calculateMovingAverages(data, [7, 14, 30]);
        
        // Update standard deviations
        this.statisticalMetrics.standardDeviations = this.calculateVolatilityMetrics(data);
        
        // Update correlation matrix
        this.statisticalMetrics.correlationMatrix = this.calculateCorrelations(data);
        
        // Update regression models
        this.statisticalMetrics.regressionModels = this.updateRegressionModels(data);
        
        // Update statistical indicators in UI
        this.updateStatisticalIndicators();
    }

    calculateMovingAverages(data, periods) {
        const mas = new Map();
        periods.forEach(period => {
            const ma = this.calculateSMA(data, period);
            mas.set(period, ma);
        });
        return mas;
    }

    calculateSMA(data, period) {
        const values = data.map(d => d.value);
        const sma = [];
        for (let i = period - 1; i < values.length; i++) {
            const avg = values.slice(i - period + 1, i + 1).reduce((a, b) => a + b) / period;
            sma.push({ timestamp: data[i].timestamp, value: avg });
        }
        return sma;
    }

    calculateVolatilityMetrics(data) {
        const metrics = new Map();
        const returns = this.calculateReturns(data);
        
        metrics.set('daily', this.calculateStandardDeviation(returns.daily));
        metrics.set('weekly', this.calculateStandardDeviation(returns.weekly));
        metrics.set('monthly', this.calculateStandardDeviation(returns.monthly));
        
        return metrics;
    }

    async scanForNewPatterns() {
        const recentData = await this.fetchRecentTradeData();
        const newPatterns = this.identifyPatterns(recentData);
        
        newPatterns.forEach(pattern => {
            if (this.isSignificantPattern(pattern)) {
                this.patternSettings.patterns.set(pattern.id, pattern);
                this.updatePatternList(pattern);
            }
        });
    }

    identifyPatterns(data) {
        const patterns = [];
        
        // Implement various pattern recognition algorithms
        patterns.push(...this.findPricePatterns(data));
        patterns.push(...this.findVolumePatterns(data));
        patterns.push(...this.findCorrelationPatterns(data));
        
        return patterns;
    }

    findPricePatterns(data) {
        const patterns = [];
        
        // Head and Shoulders
        patterns.push(...this.detectHeadAndShoulders(data));
        
        // Double Tops/Bottoms
        patterns.push(...this.detectDoubleTopsBottoms(data));
        
        // Triangle Patterns
        patterns.push(...this.detectTrianglePatterns(data));

        // Advanced Harmonic Patterns
        patterns.push(...this.detectHarmonicPatterns(data));

        // Elliott Wave Patterns
        patterns.push(...this.detectElliottWavePatterns(data));

        // Candlestick Patterns
        patterns.push(...this.detectCandlestickPatterns(data));

        // Fibonacci Patterns
        patterns.push(...this.detectFibonacciPatterns(data));
        
        return patterns;
    }

    detectHarmonicPatterns(data) {
        const patterns = [];
        const { high, low, close } = this.extractPriceData(data);
        
        // Detect Gartley Pattern
        patterns.push(...this.findGartleyPattern(high, low, close));
        
        // Detect Butterfly Pattern
        patterns.push(...this.findButterflyPattern(high, low, close));
        
        // Detect Bat Pattern
        patterns.push(...this.findBatPattern(high, low, close));
        
        // Detect Crab Pattern
        patterns.push(...this.findCrabPattern(high, low, close));
        
        // Detect Shark Pattern
        patterns.push(...this.findSharkPattern(high, low, close));
        
        return patterns.filter(p => this.validateHarmonicPattern(p));
    }

    detectElliottWavePatterns(data) {
        const waves = [];
        const prices = data.map(d => d.close);
        
        // Detect Impulse Waves
        const impulseWaves = this.findImpulseWaves(prices);
        waves.push(...impulseWaves);
        
        // Detect Corrective Waves
        const correctiveWaves = this.findCorrectiveWaves(prices);
        waves.push(...correctiveWaves);
        
        // Validate wave counts and relationships
        return this.validateElliottWaves(waves);
    }

    detectCandlestickPatterns(data) {
        const patterns = [];
        
        // Single Candlestick Patterns
        patterns.push(...this.findDoji(data));
        patterns.push(...this.findHammer(data));
        patterns.push(...this.findShootingStar(data));
        patterns.push(...this.findMarubozu(data));
        
        // Dual Candlestick Patterns
        patterns.push(...this.findEngulfing(data));
        patterns.push(...this.findHarami(data));
        patterns.push(...this.findTweezer(data));
        
        // Triple Candlestick Patterns
        patterns.push(...this.findMorningStar(data));
        patterns.push(...this.findEveningStar(data));
        patterns.push(...this.findThreeWhiteSoldiers(data));
        patterns.push(...this.findThreeBlackCrows(data));
        
        return patterns.filter(p => this.validateCandlestickPattern(p));
    }

    detectFibonacciPatterns(data) {
        const patterns = [];
        const prices = data.map(d => d.close);
        
        // Find potential Fibonacci retracement levels
        patterns.push(...this.findFibonacciRetracements(prices));
        
        // Find potential Fibonacci extension levels
        patterns.push(...this.findFibonacciExtensions(prices));
        
        // Find potential Fibonacci time zones
        patterns.push(...this.findFibonacciTimeZones(data));
        
        // Find potential Fibonacci fans
        patterns.push(...this.findFibonacciFans(data));
        
        return patterns.filter(p => this.validateFibonacciPattern(p));
    }

    findGartleyPattern(high, low, close) {
        const patterns = [];
        const tolerance = 0.05; // 5% tolerance for ratio validation
        
        for (let i = 4; i < close.length; i++) {
            const points = this.findPotentialGartleyPoints(high, low, close, i);
            if (points && this.validateGartleyRatios(points, tolerance)) {
                patterns.push({
                    type: 'Gartley',
                    points: points,
                    confidence: this.calculatePatternConfidence(points),
                    timestamp: new Date()
                });
            }
        }
        
        return patterns;
    }

    findButterflyPattern(high, low, close) {
        const patterns = [];
        const tolerance = 0.05;
        
        for (let i = 4; i < close.length; i++) {
            const points = this.findPotentialButterflyPoints(high, low, close, i);
            if (points && this.validateButterflyRatios(points, tolerance)) {
                patterns.push({
                    type: 'Butterfly',
                    points: points,
                    confidence: this.calculatePatternConfidence(points),
                    timestamp: new Date()
                });
            }
        }
        
        return patterns;
    }

    findImpulseWaves(prices) {
        const waves = [];
        const minWaveLength = 5;
        
        for (let i = minWaveLength; i < prices.length; i++) {
            const potentialWave = this.identifyImpulseWave(prices, i - minWaveLength, i);
            if (potentialWave && this.validateImpulseWave(potentialWave)) {
                waves.push({
                    type: 'Impulse',
                    wave: potentialWave,
                    confidence: this.calculateWaveConfidence(potentialWave),
                    timestamp: new Date()
                });
            }
        }
        
        return waves;
    }

    findCorrectiveWaves(prices) {
        const waves = [];
        const minWaveLength = 3;
        
        for (let i = minWaveLength; i < prices.length; i++) {
            const potentialWave = this.identifyCorrectiveWave(prices, i - minWaveLength, i);
            if (potentialWave && this.validateCorrectiveWave(potentialWave)) {
                waves.push({
                    type: 'Corrective',
                    wave: potentialWave,
                    confidence: this.calculateWaveConfidence(potentialWave),
                    timestamp: new Date()
                });
            }
        }
        
        return waves;
    }

    findDoji(data) {
        const patterns = [];
        const tolerance = 0.1; // 10% tolerance for body size
        
        data.forEach((candle, i) => {
            const bodySize = Math.abs(candle.open - candle.close);
            const wickSize = candle.high - candle.low;
            
            if (bodySize <= wickSize * tolerance) {
                patterns.push({
                    type: 'Doji',
                    position: i,
                    confidence: this.calculateDojiConfidence(candle),
                    timestamp: new Date()
                });
            }
        });
        
        return patterns;
    }

    findEngulfing(data) {
        const patterns = [];
        
        for (let i = 1; i < data.length; i++) {
            const current = data[i];
            const previous = data[i - 1];
            
            if (this.isEngulfingPattern(previous, current)) {
                patterns.push({
                    type: current.close > current.open ? 'Bullish Engulfing' : 'Bearish Engulfing',
                    position: i,
                    confidence: this.calculateEngulfingConfidence(previous, current),
                    timestamp: new Date()
                });
            }
        }
        
        return patterns;
    }

    findFibonacciRetracements(prices) {
        const patterns = [];
        const fibLevels = [0.236, 0.382, 0.5, 0.618, 0.786];
        
        for (let i = 1; i < prices.length; i++) {
            const trend = this.identifyTrend(prices, i);
            if (trend) {
                const retracementLevels = this.calculateFibonacciLevels(trend.high, trend.low);
                if (this.validateFibonacciRetracement(prices[i], retracementLevels)) {
                    patterns.push({
                        type: 'Fibonacci Retracement',
                        levels: retracementLevels,
                        confidence: this.calculateFibonacciConfidence(prices[i], retracementLevels),
                        timestamp: new Date()
                    });
                }
            }
        }
        
        return patterns;
    }

    validateHarmonicPattern(pattern) {
        const ratios = this.calculateHarmonicRatios(pattern.points);
        const expectedRatios = this.getExpectedRatios(pattern.type);
        
        return this.compareRatios(ratios, expectedRatios, 0.05);
    }

    validateElliottWaves(waves) {
        return waves.filter(wave => {
            const rules = this.getElliottWaveRules(wave.type);
            return this.validateWaveRules(wave, rules);
        });
    }

    validateCandlestickPattern(pattern) {
        const validationRules = this.getCandlestickRules(pattern.type);
        return this.applyCandlestickRules(pattern, validationRules);
    }

    validateFibonacciPattern(pattern) {
        const validationRules = this.getFibonacciRules(pattern.type);
        return this.applyFibonacciRules(pattern, validationRules);
    }

    calculatePatternConfidence(points) {
        const ratioAccuracy = this.calculateRatioAccuracy(points);
        const priceAction = this.analyzePriceAction(points);
        const volume = this.analyzeVolume(points);
        const momentum = this.analyzeMomentum(points);
        
        return this.combineConfidenceFactors([
            ratioAccuracy,
            priceAction,
            volume,
            momentum
        ]);
    }

    calculateWaveConfidence(wave) {
        const structure = this.analyzeWaveStructure(wave);
        const momentum = this.analyzeWaveMomentum(wave);
        const volume = this.analyzeWaveVolume(wave);
        const time = this.analyzeWaveTime(wave);
        
        return this.combineConfidenceFactors([
            structure,
            momentum,
            volume,
            time
        ]);
    }

    prepareDrilldownContent(data) {
        return {
            summary: this.generateSummaryStats(data),
            trends: this.analyzeTrends(data),
            correlations: this.analyzeCorrelations(data),
            predictions: this.generatePredictions(data),
            recommendations: this.generateRecommendations(data)
        };
    }

    showDrilldownModal(content) {
        const modal = document.getElementById('drilldown-modal');
        const contentDiv = document.getElementById('drilldown-content');
        
        // Clear existing content
        contentDiv.innerHTML = '';
        
        // Add summary section
        this.addDrilldownSection(contentDiv, 'Summary Statistics', content.summary);
        
        // Add trend analysis
        this.addDrilldownSection(contentDiv, 'Trend Analysis', content.trends);
        
        // Add correlation analysis
        this.addDrilldownSection(contentDiv, 'Correlation Analysis', content.correlations);
        
        // Add predictions
        this.addDrilldownSection(contentDiv, 'Predictions', content.predictions);
        
        // Add recommendations
        this.addDrilldownSection(contentDiv, 'Recommendations', content.recommendations);
        
        // Show modal
        modal.classList.remove('hidden');
    }

    addDrilldownSection(container, title, data) {
        const section = document.createElement('div');
        section.className = 'mb-6';
        
        const heading = document.createElement('h4');
        heading.className = 'text-lg font-medium text-gray-900 mb-3';
        heading.textContent = title;
        
        const content = document.createElement('div');
        content.className = 'bg-gray-50 rounded-lg p-4';
        
        // Render section content based on data type
        if (Array.isArray(data)) {
            content.appendChild(this.createList(data));
        } else if (typeof data === 'object') {
            content.appendChild(this.createMetricsGrid(data));
        } else {
            content.textContent = data;
        }
        
        section.appendChild(heading);
        section.appendChild(content);
        container.appendChild(section);
    }

    async initializeMLModels() {
        try {
            // Load pre-trained models
            this.mlModels.patternRecognition = await tf.loadLayersModel('/models/pattern_recognition.json');
            this.mlModels.anomalyDetection = await tf.loadLayersModel('/models/anomaly_detection.json');
            this.mlModels.trendPrediction = await tf.loadLayersModel('/models/trend_prediction.json');
            
            // Initialize model parameters
            this.mlModels.patternRecognition.warmup();
            this.setupModelCallbacks();
        } catch (error) {
            console.error('Failed to initialize ML models:', error);
        }
    }

    setupModelCallbacks() {
        // Set up real-time prediction callbacks
        this.mlModels.patternRecognition.onPredict = (prediction) => {
            this.handlePatternPrediction(prediction);
        };

        this.mlModels.anomalyDetection.onPredict = (anomaly) => {
            this.handleAnomalyPrediction(anomaly);
        };
    }

    calculateTechnicalIndicators(data) {
        // Calculate MACD
        const macdResult = this.calculateMACD(data);
        this.technicalIndicators.macd.values = macdResult;

        // Calculate RSI
        const rsiResult = this.calculateRSI(data);
        this.technicalIndicators.rsi.values = rsiResult;

        // Calculate Bollinger Bands
        const bbResult = this.calculateBollingerBands(data);
        this.technicalIndicators.bollingerBands.values = bbResult;

        // Calculate Stochastic Oscillator
        const stochResult = this.calculateStochastic(data);
        this.technicalIndicators.stochastic.values = stochResult;

        // Calculate ATR
        const atrResult = this.calculateATR(data);
        this.technicalIndicators.atr.values = atrResult;

        return {
            macd: macdResult,
            rsi: rsiResult,
            bollingerBands: bbResult,
            stochastic: stochResult,
            atr: atrResult
        };
    }

    calculateMACD(data) {
        const { fastPeriod, slowPeriod, signalPeriod } = this.technicalIndicators.macd;
        const prices = data.map(d => d.price);
        
        // Calculate EMAs
        const fastEMA = this.calculateEMA(prices, fastPeriod);
        const slowEMA = this.calculateEMA(prices, slowPeriod);
        
        // Calculate MACD line
        const macdLine = fastEMA.map((fast, i) => fast - slowEMA[i]);
        
        // Calculate signal line
        const signalLine = this.calculateEMA(macdLine, signalPeriod);
        
        // Calculate histogram
        const histogram = macdLine.map((macd, i) => macd - signalLine[i]);
        
        return {
            macdLine,
            signalLine,
            histogram,
            timestamps: data.map(d => d.timestamp)
        };
    }

    calculateRSI(data) {
        const period = this.technicalIndicators.rsi.period;
        const prices = data.map(d => d.price);
        const gains = [];
        const losses = [];
        
        // Calculate price changes
        for (let i = 1; i < prices.length; i++) {
            const change = prices[i] - prices[i - 1];
            gains.push(Math.max(0, change));
            losses.push(Math.max(0, -change));
        }
        
        // Calculate average gains and losses
        const avgGain = this.calculateSMA(gains, period);
        const avgLoss = this.calculateSMA(losses, period);
        
        // Calculate RSI
        const rsi = avgGain.map((gain, i) => {
            const rs = gain / avgLoss[i];
            return 100 - (100 / (1 + rs));
        });
        
        return {
            values: rsi,
            timestamps: data.slice(period).map(d => d.timestamp)
        };
    }

    calculateBollingerBands(data) {
        const { period, stdDev } = this.technicalIndicators.bollingerBands;
        const prices = data.map(d => d.price);
        
        // Calculate middle band (SMA)
        const middleBand = this.calculateSMA(prices, period);
        
        // Calculate standard deviation
        const standardDeviation = this.calculateRollingStdDev(prices, period);
        
        // Calculate upper and lower bands
        const upperBand = middleBand.map((mid, i) => mid + (standardDeviation[i] * stdDev));
        const lowerBand = middleBand.map((mid, i) => mid - (standardDeviation[i] * stdDev));
        
        return {
            upper: upperBand,
            middle: middleBand,
            lower: lowerBand,
            timestamps: data.slice(period - 1).map(d => d.timestamp)
        };
    }

    async predictPatterns(data) {
        const features = this.prepareFeatures(data);
        const tensorData = tf.tensor2d(features);
        
        // Make prediction
        const prediction = await this.mlModels.patternRecognition.predict(tensorData);
        const patterns = this.interpretPatternPrediction(prediction);
        
        tensorData.dispose();
        prediction.dispose();
        
        return patterns;
    }

    prepareFeatures(data) {
        // Calculate technical indicators for feature engineering
        const indicators = this.calculateTechnicalIndicators(data);
        
        // Combine features
        return data.map((d, i) => [
            d.price,
            d.volume,
            indicators.macd.macdLine[i] || 0,
            indicators.rsi.values[i] || 0,
            indicators.bollingerBands.upper[i] || 0,
            indicators.bollingerBands.lower[i] || 0,
            indicators.stochastic.values[i]?.k || 0,
            indicators.atr.values[i] || 0
        ]);
    }

    interpretPatternPrediction(prediction) {
        const patterns = [];
        const confidenceThreshold = 0.75;
        
        // Get prediction array
        const predictionArray = prediction.arraySync()[0];
        
        // Map predictions to pattern types
        const patternTypes = [
            'Head and Shoulders',
            'Double Top',
            'Double Bottom',
            'Triangle',
            'Flag',
            'Channel',
            'Cup and Handle'
        ];
        
        patternTypes.forEach((pattern, i) => {
            if (predictionArray[i] > confidenceThreshold) {
                patterns.push({
                    type: pattern,
                    confidence: predictionArray[i],
                    timestamp: new Date()
                });
            }
        });
        
        return patterns;
    }

    async updateDrilldownAnalysis(data) {
        const analysis = {
            technical: await this.calculateTechnicalAnalysis(data),
            patterns: await this.predictPatterns(data),
            statistics: this.calculateAdvancedStatistics(data),
            correlations: await this.calculateCorrelations(data),
            predictions: await this.generatePredictions(data)
        };
        
        return this.formatDrilldownAnalysis(analysis);
    }

    calculateAdvancedStatistics(data) {
        return {
            basic: {
                mean: this.calculateMean(data),
                median: this.calculateMedian(data),
                stdDev: this.calculateStdDev(data),
                skewness: this.calculateSkewness(data),
                kurtosis: this.calculateKurtosis(data)
            },
            momentum: {
                roc: this.calculateRateOfChange(data),
                momentum: this.calculateMomentum(data),
                trix: this.calculateTRIX(data)
            },
            volatility: {
                atr: this.calculateATR(data),
                standardDeviation: this.calculateVolatilityMetrics(data),
                bollingerWidth: this.calculateBollingerBandWidth(data)
            },
            volume: {
                obv: this.calculateOBV(data),
                vwap: this.calculateVWAP(data),
                volumeForce: this.calculateForceIndex(data)
            }
        };
    }

    formatDrilldownAnalysis(analysis) {
        return {
            summary: this.formatSummarySection(analysis),
            technicalIndicators: this.formatTechnicalSection(analysis),
            patterns: this.formatPatternsSection(analysis),
            predictions: this.formatPredictionsSection(analysis),
            statistics: this.formatStatisticsSection(analysis),
            recommendations: this.generateRecommendations(analysis)
        };
    }

    async initializeAdvancedModels() {
        try {
            // Initialize LSTM models
            this.mlModels.lstm.pricePredictor = await tf.loadLayersModel('/models/lstm_price.json');
            this.mlModels.lstm.volumePredictor = await tf.loadLayersModel('/models/lstm_volume.json');
            this.mlModels.lstm.volatilityPredictor = await tf.loadLayersModel('/models/lstm_volatility.json');

            // Initialize Transformer models
            this.mlModels.transformer.multivariate = await tf.loadLayersModel('/models/transformer_multivariate.json');
            this.mlModels.transformer.attention = await tf.loadLayersModel('/models/transformer_attention.json');

            // Initialize ensemble models
            const ensembleModels = await Promise.all([
                tf.loadLayersModel('/models/ensemble_model1.json'),
                tf.loadLayersModel('/models/ensemble_model2.json'),
                tf.loadLayersModel('/models/ensemble_model3.json')
            ]);
            this.mlModels.ensemble.models = ensembleModels;
            this.mlModels.ensemble.weights = [0.4, 0.3, 0.3];

            // Warm up models
            await this.warmupModels();
        } catch (error) {
            console.error('Failed to initialize advanced models:', error);
        }
    }

    async warmupModels() {
        const warmupData = tf.zeros([1, 100, 8]); // Example shape
        
        // Warm up LSTM models
        await this.mlModels.lstm.pricePredictor.predict(warmupData).data();
        await this.mlModels.lstm.volumePredictor.predict(warmupData).data();
        await this.mlModels.lstm.volatilityPredictor.predict(warmupData).data();
        
        // Warm up Transformer models
        await this.mlModels.transformer.multivariate.predict(warmupData).data();
        await this.mlModels.transformer.attention.predict(warmupData).data();
        
        warmupData.dispose();
    }

    calculateAdvancedIndicators(data) {
        const indicators = {
            ichimoku: this.calculateIchimoku(data),
            keltner: this.calculateKeltnerChannels(data),
            vwap: this.calculateVWAP(data),
            cmf: this.calculateCMF(data),
            dmi: this.calculateDMI(data),
            supertrend: this.calculateSupertrend(data)
        };

        // Update state
        Object.entries(indicators).forEach(([key, value]) => {
            this.technicalIndicators[key].values = value;
        });

        return indicators;
    }

    calculateIchimoku(data) {
        const { conversionPeriod, basePeriod, spanPeriod, displacement } = this.technicalIndicators.ichimoku;
        const prices = data.map(d => d.price);
        
        const conversionLine = this.calculateIchimokuLine(prices, conversionPeriod);
        const baseLine = this.calculateIchimokuLine(prices, basePeriod);
        const leadingSpanA = this.calculateLeadingSpanA(conversionLine, baseLine);
        const leadingSpanB = this.calculateLeadingSpanB(prices, spanPeriod);
        
        return {
            conversion: conversionLine,
            base: baseLine,
            spanA: leadingSpanA,
            spanB: leadingSpanB,
            laggingSpan: prices.slice(0, -displacement)
        };
    }

    calculateKeltnerChannels(data) {
        const { emaPeriod, atrPeriod, multiplier } = this.technicalIndicators.keltnerChannels;
        const prices = data.map(d => d.price);
        
        const middleLine = this.calculateEMA(prices, emaPeriod);
        const atr = this.calculateATR(data, atrPeriod);
        
        const upperBand = middleLine.map((value, i) => value + (multiplier * atr[i]));
        const lowerBand = middleLine.map((value, i) => value - (multiplier * atr[i]));
        
        return { middle: middleLine, upper: upperBand, lower: lowerBand };
    }

    async generateAdvancedPredictions(data) {
        const features = this.prepareAdvancedFeatures(data);
        const tensorData = tf.tensor3d(features, [1, features.length, features[0].length]);
        
        try {
            // Generate predictions from different models
            const lstmPrediction = await this.mlModels.lstm.pricePredictor.predict(tensorData).data();
            const transformerPrediction = await this.mlModels.transformer.multivariate.predict(tensorData).data();
            const ensemblePredictions = await Promise.all(
                this.mlModels.ensemble.models.map(model => model.predict(tensorData).data())
            );
            
            // Combine predictions using ensemble weights
            const ensemblePrediction = this.combineEnsemblePredictions(ensemblePredictions);
            
            // Calculate confidence intervals
            const predictions = {
                lstm: this.processLSTMPrediction(lstmPrediction),
                transformer: this.processTransformerPrediction(transformerPrediction),
                ensemble: this.processEnsemblePrediction(ensemblePrediction)
            };
            
            tensorData.dispose();
            return predictions;
        } catch (error) {
            console.error('Prediction error:', error);
            tensorData.dispose();
            return null;
        }
    }

    prepareAdvancedFeatures(data) {
        const technicalFeatures = this.calculateAdvancedIndicators(data);
        const marketFeatures = this.calculateAdvancedMarketFeatures(data);
        const sentimentFeatures = this.calculateSentimentFeatures(data);
        
        return data.map((d, i) => [
            d.price,
            d.volume,
            technicalFeatures.ichimoku.conversion[i],
            technicalFeatures.ichimoku.base[i],
            technicalFeatures.keltner.middle[i],
            technicalFeatures.vwap[i],
            technicalFeatures.cmf[i],
            technicalFeatures.dmi.plusDI[i],
            technicalFeatures.dmi.minusDI[i],
            technicalFeatures.supertrend.trend[i],
            marketFeatures.liquidity[i],
            marketFeatures.volatility[i],
            sentimentFeatures.score[i]
        ]);
    }

    async updateDrilldownAnalysis(data) {
        const analysis = {
            technical: await this.calculateTechnicalAnalysis(data),
            patterns: await this.predictPatterns(data),
            statistics: this.calculateAdvancedStatistics(data),
            correlations: await this.calculateCorrelations(data),
            predictions: await this.generateAdvancedPredictions(data),
            marketStructure: this.analyzeMarketStructure(data),
            orderFlow: this.analyzeOrderFlow(data),
            liquidityProfile: this.analyzeLiquidityProfile(data)
        };
        
        return this.formatAdvancedDrilldownAnalysis(analysis);
    }

    formatAdvancedDrilldownAnalysis(analysis) {
        return {
            summary: this.formatSummarySection(analysis),
            technicalAnalysis: this.formatTechnicalSection(analysis),
            patterns: this.formatPatternsSection(analysis),
            predictions: this.formatPredictionsSection(analysis),
            statistics: this.formatStatisticsSection(analysis),
            marketStructure: this.formatMarketStructureSection(analysis),
            orderFlow: this.formatOrderFlowSection(analysis),
            liquidity: this.formatLiquiditySection(analysis),
            recommendations: this.generateAdvancedRecommendations(analysis)
        };
    }

    showAdvancedDrilldownModal(content) {
        const modal = document.getElementById('drilldown-modal');
        const contentDiv = document.getElementById('drilldown-content');
        
        contentDiv.innerHTML = '';
        
        // Add interactive sections
        this.addInteractiveTechnicalSection(contentDiv, content.technicalAnalysis);
        this.addInteractivePatternsSection(contentDiv, content.patterns);
        this.addInteractivePredictionsSection(contentDiv, content.predictions);
        this.addInteractiveMarketStructureSection(contentDiv, content.marketStructure);
        this.addInteractiveOrderFlowSection(contentDiv, content.orderFlow);
        this.addInteractiveLiquiditySection(contentDiv, content.liquidity);
        
        modal.classList.remove('hidden');
    }

    addInteractiveTechnicalSection(container, data) {
        const section = document.createElement('div');
        section.className = 'mb-6';
        
        // Add interactive controls
        const controls = this.createTechnicalControls(data);
        section.appendChild(controls);
        
        // Add interactive chart
        const chart = this.createInteractiveChart(data);
        section.appendChild(chart);
        
        container.appendChild(section);
    }

    createInteractiveChart(data) {
        const chartContainer = document.createElement('div');
        chartContainer.className = 'mt-4 relative';
        
        // Add zoom controls
        const zoomControls = this.createZoomControls();
        chartContainer.appendChild(zoomControls);
        
        // Add interactive tooltips
        const tooltips = this.createInteractiveTooltips(data);
        chartContainer.appendChild(tooltips);
        
        // Add crosshair
        const crosshair = this.createCrosshair();
        chartContainer.appendChild(crosshair);
        
        return chartContainer;
    }

    async initializeEnhancedModels() {
        try {
            // Initialize GRU models
            this.mlModels.gru.sequenceModel = await tf.loadLayersModel('/models/gru_sequence.json');
            this.mlModels.gru.attentionModel = await tf.loadLayersModel('/models/gru_attention.json');

            // Initialize WaveNet models
            this.mlModels.wavenet.dilatedModel = await tf.loadLayersModel('/models/wavenet_dilated.json');
            this.mlModels.wavenet.residualModel = await tf.loadLayersModel('/models/wavenet_residual.json');

            // Initialize Transformer-XL
            this.mlModels.transformerXL.model = await tf.loadLayersModel('/models/transformer_xl.json');

            // Initialize Autoencoder
            this.mlModels.autoencoder.encoder = await tf.loadLayersModel('/models/autoencoder_encoder.json');
            this.mlModels.autoencoder.decoder = await tf.loadLayersModel('/models/autoencoder_decoder.json');

            // Initialize Hybrid Models
            this.mlModels.hybridModels.lstmTransformer = await tf.loadLayersModel('/models/lstm_transformer.json');
            this.mlModels.hybridModels.cnnLstm = await tf.loadLayersModel('/models/cnn_lstm.json');
            this.mlModels.hybridModels.transformerGru = await tf.loadLayersModel('/models/transformer_gru.json');

            await this.warmupEnhancedModels();
        } catch (error) {
            console.error('Failed to initialize enhanced models:', error);
        }
    }

    async warmupEnhancedModels() {
        const warmupData = tf.zeros([1, 100, 16]); // Enhanced feature dimension
        
        // Warm up GRU models
        await this.mlModels.gru.sequenceModel.predict(warmupData).data();
        await this.mlModels.gru.attentionModel.predict(warmupData).data();
        
        // Warm up WaveNet models
        await this.mlModels.wavenet.dilatedModel.predict(warmupData).data();
        await this.mlModels.wavenet.residualModel.predict(warmupData).data();
        
        // Warm up Transformer-XL
        await this.mlModels.transformerXL.model.predict(warmupData).data();
        
        // Warm up Autoencoder
        const encoded = await this.mlModels.autoencoder.encoder.predict(warmupData).data();
        await this.mlModels.autoencoder.decoder.predict(tf.tensor(encoded)).data();
        
        // Warm up Hybrid Models
        await this.mlModels.hybridModels.lstmTransformer.predict(warmupData).data();
        await this.mlModels.hybridModels.cnnLstm.predict(warmupData).data();
        await this.mlModels.hybridModels.transformerGru.predict(warmupData).data();
        
        warmupData.dispose();
    }

    setupAdvancedMonitoring() {
        // Setup real-time price monitoring
        this.monitoringSystem.monitoringIntervals.set('price', 
            setInterval(() => this.monitorPriceMovements(), 1000));

        // Setup volume analysis
        this.monitoringSystem.monitoringIntervals.set('volume',
            setInterval(() => this.monitorVolumePatterns(), 5000));

        // Setup pattern recognition
        this.monitoringSystem.monitoringIntervals.set('patterns',
            setInterval(() => this.monitorTradingPatterns(), 10000));

        // Setup volatility monitoring
        this.monitoringSystem.monitoringIntervals.set('volatility',
            setInterval(() => this.monitorVolatilityChanges(), 3000));

        // Setup custom alert monitoring
        this.monitoringSystem.monitoringIntervals.set('custom',
            setInterval(() => this.checkCustomAlerts(), 2000));
    }

    async monitorPriceMovements() {
        const currentPrice = await this.fetchLatestPrice();
        const predictions = await this.generateEnhancedPredictions(currentPrice);
        
        if (this.detectPriceAnomaly(currentPrice, predictions)) {
            this.triggerPriceAlert({
                type: 'PRICE_ANOMALY',
                price: currentPrice,
                prediction: predictions,
                timestamp: new Date()
            });
        }
    }

    async monitorVolumePatterns() {
        const volumeData = await this.fetchVolumeData();
        const analysis = this.analyzeVolumeProfile(volumeData);
        
        if (analysis.anomalies.length > 0) {
            this.triggerVolumeAlert({
                type: 'VOLUME_ANOMALY',
                anomalies: analysis.anomalies,
                timestamp: new Date()
            });
        }
    }

    async generateEnhancedPredictions(data) {
        const features = this.prepareEnhancedFeatures(data);
        const tensorData = tf.tensor3d(features, [1, features.length, features[0].length]);
        
        try {
            // Generate predictions from different model architectures
            const gruPrediction = await this.mlModels.gru.sequenceModel.predict(tensorData).data();
            const wavenetPrediction = await this.mlModels.wavenet.dilatedModel.predict(tensorData).data();
            const transformerXLPrediction = await this.mlModels.transformerXL.model.predict(tensorData).data();
            
            // Generate hybrid predictions
            const hybridPredictions = await this.generateHybridPredictions(tensorData);
            
            // Combine predictions using advanced ensemble technique
            const enhancedPrediction = this.combineEnhancedPredictions([
                gruPrediction,
                wavenetPrediction,
                transformerXLPrediction,
                ...hybridPredictions
            ]);
            
            tensorData.dispose();
            return enhancedPrediction;
        } catch (error) {
            console.error('Enhanced prediction error:', error);
            tensorData.dispose();
            return null;
        }
    }

    prepareEnhancedFeatures(data) {
        const technicalFeatures = this.calculateEnhancedIndicators(data);
        const marketFeatures = this.calculateAdvancedMarketFeatures(data);
        const volumeFeatures = this.calculateVolumeProfile(data);
        
        return data.map((d, i) => [
            ...this.getBasicFeatures(d),
            ...this.getTechnicalFeatures(technicalFeatures, i),
            ...this.getMarketFeatures(marketFeatures, i),
            ...this.getVolumeFeatures(volumeFeatures, i)
        ]);
    }

    calculateEnhancedIndicators(data) {
        return {
            zigzag: this.calculateZigZag(data),
            williamsR: this.calculateWilliamsR(data),
            ultimateOscillator: this.calculateUltimateOscillator(data),
            aroon: this.calculateAroon(data),
            volumeProfile: this.calculateVolumeProfile(data),
            marketProfile: this.calculateMarketProfile(data)
        };
    }

    calculateZigZag(data) {
        const { deviation } = this.technicalIndicators.zigzag;
        const prices = data.map(d => d.price);
        const zigzag = [];
        let pivot = prices[0];
        let trend = 0;
        
        for (let i = 1; i < prices.length; i++) {
            const change = ((prices[i] - pivot) / pivot) * 100;
            
            if (Math.abs(change) >= deviation) {
                zigzag.push({
                    price: prices[i],
                    index: i,
                    trend: change > 0 ? 1 : -1
                });
                pivot = prices[i];
                trend = change > 0 ? 1 : -1;
            }
        }
        
        return zigzag;
    }

    calculateWilliamsR(data) {
        const period = this.technicalIndicators.williamsR.period;
        const values = [];
        
        for (let i = period - 1; i < data.length; i++) {
            const slice = data.slice(i - period + 1, i + 1);
            const high = Math.max(...slice.map(d => d.high));
            const low = Math.min(...slice.map(d => d.low));
            const close = slice[slice.length - 1].close;
            
            const williamsR = ((high - close) / (high - low)) * -100;
            values.push(williamsR);
        }
        
        return values;
    }

    calculateUltimateOscillator(data) {
        const { period1, period2, period3 } = this.technicalIndicators.ultimateOscillator;
        const values = [];
        
        for (let i = Math.max(period1, period2, period3) - 1; i < data.length; i++) {
            const bp1 = this.calculateBuyingPressure(data, i, period1);
            const bp2 = this.calculateBuyingPressure(data, i, period2);
            const bp3 = this.calculateBuyingPressure(data, i, period3);
            
            const uo = (4 * bp1 + 2 * bp2 + bp3) / 7;
            values.push(uo);
        }
        
        return values;
    }

    calculateAroon(data) {
        const period = this.technicalIndicators.aroon.period;
        const values = [];
        
        for (let i = period - 1; i < data.length; i++) {
            const slice = data.slice(i - period + 1, i + 1);
            const highIndex = slice.indexOf(Math.max(...slice.map(d => d.high)));
            const lowIndex = slice.indexOf(Math.min(...slice.map(d => d.low)));
            
            const aroonUp = ((period - highIndex) / period) * 100;
            const aroonDown = ((period - lowIndex) / period) * 100;
            
            values.push({
                up: aroonUp,
                down: aroonDown,
                oscillator: aroonUp - aroonDown
            });
        }
        
        return values;
    }

    calculateVolumeProfile(data) {
        const levels = this.technicalIndicators.volumeProfile.levels;
        const profile = new Map();
        
        // Calculate price range
        const prices = data.map(d => d.price);
        const min = Math.min(...prices);
        const max = Math.max(...prices);
        const step = (max - min) / levels;
        
        // Initialize price levels
        for (let i = 0; i < levels; i++) {
            profile.set(min + (i * step), 0);
        }
        
        // Aggregate volume at each price level
        data.forEach(d => {
            const level = Math.floor((d.price - min) / step) * step + min;
            profile.set(level, (profile.get(level) || 0) + d.volume);
        });
        
        return Array.from(profile.entries()).map(([price, volume]) => ({
            price,
            volume,
            poc: false // Point of Control flag
        }));
    }

    calculateMarketProfile(data) {
        const timeframe = this.technicalIndicators.marketProfile.timeframe;
        const profile = new Map();
        
        // Group data by time periods
        const periods = this.groupDataByTimeframe(data, timeframe);
        
        periods.forEach(period => {
            const tpo = this.calculateTimeAndPriceOpportunity(period);
            this.mergeTPOIntoProfile(profile, tpo);
        });
        
        return Array.from(profile.entries()).map(([price, value]) => ({
            price,
            letters: value.letters,
            volume: value.volume,
            valueArea: value.valueArea
        }));
    }

    createInteractiveVisualization(container, data, options = {}) {
        const {
            type = 'candlestick',
            indicators = [],
            overlays = [],
            interactions = ['zoom', 'pan', 'crosshair']
        } = options;

        // Create main chart
        const mainChart = this.createMainChart(container, data, type);
        
        // Add technical indicators
        indicators.forEach(indicator => {
            this.addIndicatorToChart(mainChart, data, indicator);
        });
        
        // Add overlay charts
        overlays.forEach(overlay => {
            this.addChartOverlay(mainChart, data, overlay);
        });
        
        // Add interactive features
        if (interactions.includes('zoom')) {
            this.addZoomCapability(mainChart);
        }
        if (interactions.includes('pan')) {
            this.addPanCapability(mainChart);
        }
        if (interactions.includes('crosshair')) {
            this.addCrosshair(mainChart);
        }
        
        // Add tooltips and legends
        this.addEnhancedTooltips(mainChart);
        this.addInteractiveLegend(mainChart);
        
        return mainChart;
    }

    addZoomCapability(chart) {
        const zoomOptions = {
            mode: 'xy',
            rangeMin: {
                x: null,
                y: null
            },
            rangeMax: {
                x: null,
                y: null
            },
            onZoom: (event) => this.handleZoomEvent(event, chart)
        };
        
        chart.options.plugins.zoom = zoomOptions;
        chart.update();
    }

    addPanCapability(chart) {
        const panOptions = {
            enabled: true,
            mode: 'xy',
            speed: 20,
            threshold: 10,
            onPan: (event) => this.handlePanEvent(event, chart)
        };
        
        chart.options.plugins.pan = panOptions;
        chart.update();
    }

    addCrosshair(chart) {
        const crosshairPlugin = {
            id: 'crosshair',
            afterDraw: (chart, args, options) => {
                const { ctx, chartArea, scales } = chart;
                
                if (!chart.crosshairPosition) return;
                
                const { x, y } = chart.crosshairPosition;
                
                // Draw vertical line
                ctx.beginPath();
                ctx.moveTo(x, chartArea.top);
                ctx.lineTo(x, chartArea.bottom);
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
                ctx.stroke();
                
                // Draw horizontal line
                ctx.beginPath();
                ctx.moveTo(chartArea.left, y);
                ctx.lineTo(chartArea.right, y);
                ctx.stroke();
                
                // Show values
                this.showCrosshairValues(chart, x, y);
            }
        };
        
        chart.options.plugins.push(crosshairPlugin);
        chart.update();
    }

    addEnhancedTooltips(chart) {
        chart.options.plugins.tooltip = {
            enabled: true,
            mode: 'index',
            intersect: false,
            position: 'nearest',
            callbacks: {
                label: (context) => this.formatTooltipLabel(context),
                title: (tooltipItems) => this.formatTooltipTitle(tooltipItems)
            },
            external: (context) => this.customTooltipRenderer(context)
        };
        
        chart.update();
    }

    addInteractiveLegend(chart) {
        const legendPlugin = {
            id: 'interactiveLegend',
            afterDraw: (chart) => {
                const legendItems = chart.legend.legendItems;
                
                legendItems.forEach((item, index) => {
                    item.clickHandler = () => this.handleLegendClick(chart, index);
                    item.hoverHandler = () => this.handleLegendHover(chart, index);
                });
            }
        };
        
        chart.options.plugins.push(legendPlugin);
        chart.update();
    }

    handleZoomEvent(event, chart) {
        const { chart: chartInstance } = event;
        const { min, max } = chartInstance.scales.x;
        
        // Update data range based on zoom
        this.updateDataRange(min, max);
        
        // Recalculate indicators for visible range
        this.recalculateIndicators(chart, min, max);
        
        // Update overlays
        this.updateChartOverlays(chart);
    }

    handlePanEvent(event, chart) {
        const { chart: chartInstance } = event;
        const { min, max } = chartInstance.scales.x;
        
        // Check if we need to load more data
        if (this.shouldLoadMoreData(min, max)) {
            this.loadAdditionalData(chart, min, max);
        }
        
        // Update visible indicators
        this.updateVisibleIndicators(chart, min, max);
    }

    showCrosshairValues(chart, x, y) {
        const { ctx, scales } = chart;
        const xValue = scales.x.getValueForPixel(x);
        const yValue = scales.y.getValueForPixel(y);
        
        // Format values
        const formattedX = this.formatDateValue(xValue);
        const formattedY = this.formatPriceValue(yValue);
        
        // Draw value boxes
        this.drawValueBox(ctx, x, chart.chartArea.bottom, formattedX);
        this.drawValueBox(ctx, chart.chartArea.left, y, formattedY);
    }

    customTooltipRenderer(context) {
        const tooltipEl = document.getElementById('chartjs-tooltip');
        
        if (!tooltipEl) {
            const newTooltip = document.createElement('div');
            newTooltip.id = 'chartjs-tooltip';
            newTooltip.innerHTML = '<table></table>';
            document.body.appendChild(newTooltip);
        }
        
        const tooltipRoot = document.getElementById('chartjs-tooltip');
        const position = context.chart.canvas.getBoundingClientRect();
        
        // Show tooltip
        if (context.tooltip.opacity === 0) {
            tooltipRoot.style.opacity = 0;
            return;
        }
        
        // Set tooltip content
        if (context.tooltip.body) {
            this.updateTooltipContent(tooltipRoot, context);
        }
        
        // Position tooltip
        const { offsetLeft: positionX, offsetTop: positionY } = position;
        
        tooltipRoot.style.opacity = 1;
        tooltipRoot.style.left = positionX + context.tooltip.caretX + 'px';
        tooltipRoot.style.top = positionY + context.tooltip.caretY + 'px';
    }

    updateTooltipContent(tooltipRoot, context) {
        const tableRoot = tooltipRoot.querySelector('table');
        const titleLines = context.tooltip.title;
        const bodyLines = context.tooltip.body.map(b => b.lines);
        
        let innerHTML = '<thead>';
        
        titleLines.forEach(title => {
            innerHTML += `<tr><th class="tooltip-title">${title}</th></tr>`;
        });
        
        innerHTML += '</thead><tbody>';
        
        bodyLines.forEach((body, i) => {
            const colors = context.tooltip.labelColors[i];
            const style = `background:${colors.backgroundColor};border-color:${colors.borderColor}`;
            
            innerHTML += `
                <tr>
                    <td class="tooltip-marker" style="${style}"></td>
                    <td class="tooltip-value">${body}</td>
                </tr>
            `;
        });
        
        innerHTML += '</tbody>';
        tableRoot.innerHTML = innerHTML;
    }

    handleLegendClick(chart, index) {
        const dataset = chart.data.datasets[index];
        dataset.hidden = !dataset.hidden;
        
        // Update visibility state
        this.updateIndicatorVisibility(chart, index, !dataset.hidden);
        
        chart.update();
    }

    handleLegendHover(chart, index) {
        const dataset = chart.data.datasets[index];
        
        // Highlight the hovered dataset
        chart.data.datasets.forEach((ds, i) => {
            if (i !== index) {
                ds.borderWidth = 1;
                ds.borderColor = this.fadeColor(ds.originalBorderColor || ds.borderColor);
            }
        });
        
        dataset.borderWidth = 2;
        dataset.borderColor = dataset.originalBorderColor || dataset.borderColor;
        
        chart.update();
    }

    monitorTradingPatterns() {
        const patterns = this.scanForPatterns();
        
        patterns.forEach(pattern => {
            if (this.isSignificantPattern(pattern)) {
                this.triggerPatternAlert(pattern);
            }
        });
    }

    monitorVolatilityChanges() {
        const volatility = this.calculateCurrentVolatility();
        const threshold = this.monitoringSystem.volatilityAlerts.threshold;
        
        if (volatility > threshold) {
            this.triggerVolatilityAlert({
                type: 'HIGH_VOLATILITY',
                value: volatility,
                threshold: threshold,
                timestamp: new Date()
            });
        }
    }

    checkCustomAlerts() {
        this.monitoringSystem.customAlerts.forEach((alert, id) => {
            if (this.checkAlertCondition(alert)) {
                this.triggerCustomAlert(id, alert);
            }
        });
    }

    triggerPriceAlert(alert) {
        // Add to alert history
        this.monitoringSystem.alertHistory.push({
            ...alert,
            id: crypto.randomUUID(),
            acknowledged: false
        });
        
        // Show notification
        this.showAlertNotification(alert);
        
        // Update UI
        this.updateAlertUI();
        
        // Trigger any registered callbacks
        this.notifyAlertSubscribers('price', alert);
    }

    triggerVolumeAlert(alert) {
        this.monitoringSystem.alertHistory.push({
            ...alert,
            id: crypto.randomUUID(),
            acknowledged: false
        });
        
        this.showAlertNotification(alert);
        this.updateAlertUI();
        this.notifyAlertSubscribers('volume', alert);
    }

    triggerPatternAlert(pattern) {
        const alert = {
            type: 'PATTERN_DETECTED',
            pattern: pattern,
            confidence: pattern.confidence,
            timestamp: new Date()
        };
        
        this.monitoringSystem.alertHistory.push({
            ...alert,
            id: crypto.randomUUID(),
            acknowledged: false
        });
        
        this.showAlertNotification(alert);
        this.updateAlertUI();
        this.notifyAlertSubscribers('pattern', alert);
    }

    triggerVolatilityAlert(alert) {
        this.monitoringSystem.alertHistory.push({
            ...alert,
            id: crypto.randomUUID(),
            acknowledged: false
        });
        
        this.showAlertNotification(alert);
        this.updateAlertUI();
        this.notifyAlertSubscribers('volatility', alert);
    }

    triggerCustomAlert(id, alert) {
        const alertData = {
            ...alert,
            id: crypto.randomUUID(),
            triggerId: id,
            timestamp: new Date(),
            acknowledged: false
        };
        
        this.monitoringSystem.alertHistory.push(alertData);
        this.showAlertNotification(alertData);
        this.updateAlertUI();
        this.notifyAlertSubscribers('custom', alertData);
    }

    showAlertNotification(alert) {
        const container = document.getElementById('alerts-container');
        const alertElement = document.createElement('div');
        
        alertElement.className = `alert alert-${this.getAlertSeverityClass(alert)} slide-in`;
        alertElement.innerHTML = this.formatAlertContent(alert);
        
        // Add close button
        const closeButton = document.createElement('button');
        closeButton.className = 'alert-close';
        closeButton.innerHTML = '×';
        closeButton.onclick = () => this.acknowledgeAlert(alert.id);
        
        alertElement.appendChild(closeButton);
        container.appendChild(alertElement);
        
        // Auto-remove after delay
        setTimeout(() => {
            alertElement.classList.add('slide-out');
            setTimeout(() => alertElement.remove(), 300);
        }, 5000);
    }

    updateAlertUI() {
        const container = document.getElementById('alerts-container');
        const activeAlerts = this.monitoringSystem.alertHistory.filter(a => !a.acknowledged);
        
        // Update alert count badge
        document.getElementById('alert-count').textContent = activeAlerts.length;
        
        // Update alerts panel
        const alertsList = document.getElementById('alerts-list');
        if (alertsList) {
            alertsList.innerHTML = activeAlerts
                .map(alert => this.createAlertListItem(alert))
                .join('');
        }
    }

    acknowledgeAlert(alertId) {
        const alert = this.monitoringSystem.alertHistory.find(a => a.id === alertId);
        if (alert) {
            alert.acknowledged = true;
            this.updateAlertUI();
        }
    }

    getAlertSeverityClass(alert) {
        switch (alert.type) {
            case 'PRICE_ANOMALY':
                return 'warning';
            case 'VOLUME_ANOMALY':
                return 'info';
            case 'PATTERN_DETECTED':
                return 'success';
            case 'HIGH_VOLATILITY':
                return 'danger';
            default:
                return 'primary';
        }
    }

    formatAlertContent(alert) {
        switch (alert.type) {
            case 'PRICE_ANOMALY':
                return `Price anomaly detected: ${alert.price} (${alert.prediction.confidence.toFixed(2)}% confidence)`;
            case 'VOLUME_ANOMALY':
                return `Unusual volume detected: ${alert.anomalies.length} anomalies found`;
            case 'PATTERN_DETECTED':
                return `Pattern detected: ${alert.pattern.type} (${alert.confidence.toFixed(2)}% confidence)`;
            case 'HIGH_VOLATILITY':
                return `High volatility alert: ${alert.value.toFixed(2)}% (threshold: ${alert.threshold}%)`;
            default:
                return `Custom alert: ${alert.message}`;
        }
    }

    createAlertListItem(alert) {
        return `
            <div class="alert-item ${alert.acknowledged ? 'acknowledged' : ''}" data-alert-id="${alert.id}">
                <div class="alert-content">
                    <div class="alert-header">
                        <span class="alert-type">${alert.type}</span>
                        <span class="alert-time">${this.formatTimestamp(alert.timestamp)}</span>
                    </div>
                    <div class="alert-message">${this.formatAlertContent(alert)}</div>
                </div>
                <div class="alert-actions">
                    <button onclick="window.dashboardManager.acknowledgeAlert('${alert.id}')">
                        Acknowledge
                    </button>
                    <button onclick="window.dashboardManager.showAlertDetails('${alert.id}')">
                        Details
                    </button>
                </div>
            </div>
        `;
    }

    showAlertDetails(alertId) {
        const alert = this.monitoringSystem.alertHistory.find(a => a.id === alertId);
        if (!alert) return;
        
        const modal = document.getElementById('alert-details-modal');
        const content = document.getElementById('alert-details-content');
        
        content.innerHTML = this.formatAlertDetails(alert);
        modal.classList.remove('hidden');
    }

    formatAlertDetails(alert) {
        let details = `
            <div class="alert-details">
                <h3>Alert Details</h3>
                <div class="detail-row">
                    <span class="label">Type:</span>
                    <span class="value">${alert.type}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Time:</span>
                    <span class="value">${this.formatTimestamp(alert.timestamp)}</span>
                </div>
        `;
        
        // Add type-specific details
        switch (alert.type) {
            case 'PRICE_ANOMALY':
                details += this.formatPriceAnomalyDetails(alert);
                break;
            case 'VOLUME_ANOMALY':
                details += this.formatVolumeAnomalyDetails(alert);
                break;
            case 'PATTERN_DETECTED':
                details += this.formatPatternDetails(alert);
                break;
            case 'HIGH_VOLATILITY':
                details += this.formatVolatilityDetails(alert);
                break;
        }
        
        details += '</div>';
        return details;
    }

    formatTimestamp(timestamp) {
        return new Date(timestamp).toLocaleString();
    }

    notifyAlertSubscribers(type, alert) {
        const subscribers = this.monitoringSystem.subscribers?.[type] || [];
        subscribers.forEach(callback => {
            try {
                callback(alert);
            } catch (error) {
                console.error('Error in alert subscriber callback:', error);
            }
        });
    }

    findHammer(data) {
        const patterns = [];
        const shadowMultiplier = 2; // Lower shadow should be at least 2x the body
        
        data.forEach((candle, i) => {
            const bodySize = Math.abs(candle.close - candle.open);
            const upperShadow = candle.high - Math.max(candle.open, candle.close);
            const lowerShadow = Math.min(candle.open, candle.close) - candle.low;
            
            if (lowerShadow > bodySize * shadowMultiplier && upperShadow < bodySize) {
                patterns.push({
                    type: 'Hammer',
                    position: i,
                    confidence: this.calculateHammerConfidence(candle),
                    timestamp: new Date()
                });
            }
        });
        
        return patterns;
    }

    findShootingStar(data) {
        const patterns = [];
        const shadowMultiplier = 2;
        
        data.forEach((candle, i) => {
            const bodySize = Math.abs(candle.close - candle.open);
            const upperShadow = candle.high - Math.max(candle.open, candle.close);
            const lowerShadow = Math.min(candle.open, candle.close) - candle.low;
            
            if (upperShadow > bodySize * shadowMultiplier && lowerShadow < bodySize) {
                patterns.push({
                    type: 'Shooting Star',
                    position: i,
                    confidence: this.calculateShootingStarConfidence(candle),
                    timestamp: new Date()
                });
            }
        });
        
        return patterns;
    }

    findMarubozu(data) {
        const patterns = [];
        const shadowTolerance = 0.1; // 10% of body size tolerance for shadows
        
        data.forEach((candle, i) => {
            const bodySize = Math.abs(candle.close - candle.open);
            const upperShadow = candle.high - Math.max(candle.open, candle.close);
            const lowerShadow = Math.min(candle.open, candle.close) - candle.low;
            
            if (upperShadow <= bodySize * shadowTolerance && 
                lowerShadow <= bodySize * shadowTolerance) {
                patterns.push({
                    type: candle.close > candle.open ? 'Bullish Marubozu' : 'Bearish Marubozu',
                    position: i,
                    confidence: this.calculateMarubozuConfidence(candle),
                    timestamp: new Date()
                });
            }
        });
        
        return patterns;
    }

    findHarami(data) {
        const patterns = [];
        
        for (let i = 1; i < data.length; i++) {
            const current = data[i];
            const previous = data[i - 1];
            
            if (this.isHaramiPattern(previous, current)) {
                patterns.push({
                    type: current.close > current.open ? 'Bullish Harami' : 'Bearish Harami',
                    position: i,
                    confidence: this.calculateHaramiConfidence(previous, current),
                    timestamp: new Date()
                });
            }
        }
        
        return patterns;
    }

    findTweezer(data) {
        const patterns = [];
        const priceTolerance = 0.001; // 0.1% tolerance for price levels
        
        for (let i = 1; i < data.length; i++) {
            const current = data[i];
            const previous = data[i - 1];
            
            if (this.isTweezerPattern(previous, current, priceTolerance)) {
                patterns.push({
                    type: current.close > current.open ? 'Tweezer Bottom' : 'Tweezer Top',
                    position: i,
                    confidence: this.calculateTweezerConfidence(previous, current),
                    timestamp: new Date()
                });
            }
        }
        
        return patterns;
    }

    findMorningStar(data) {
        const patterns = [];
        
        for (let i = 2; i < data.length; i++) {
            const first = data[i - 2];
            const second = data[i - 1];
            const third = data[i];
            
            if (this.isMorningStarPattern(first, second, third)) {
                patterns.push({
                    type: 'Morning Star',
                    position: i,
                    confidence: this.calculateMorningStarConfidence(first, second, third),
                    timestamp: new Date()
                });
            }
        }
        
        return patterns;
    }

    findEveningStar(data) {
        const patterns = [];
        
        for (let i = 2; i < data.length; i++) {
            const first = data[i - 2];
            const second = data[i - 1];
            const third = data[i];
            
            if (this.isEveningStarPattern(first, second, third)) {
                patterns.push({
                    type: 'Evening Star',
                    position: i,
                    confidence: this.calculateEveningStarConfidence(first, second, third),
                    timestamp: new Date()
                });
            }
        }
        
        return patterns;
    }

    findThreeWhiteSoldiers(data) {
        const patterns = [];
        
        for (let i = 2; i < data.length; i++) {
            const first = data[i - 2];
            const second = data[i - 1];
            const third = data[i];
            
            if (this.isThreeWhiteSoldiersPattern(first, second, third)) {
                patterns.push({
                    type: 'Three White Soldiers',
                    position: i,
                    confidence: this.calculateThreeWhiteSoldiersConfidence(first, second, third),
                    timestamp: new Date()
                });
            }
        }
        
        return patterns;
    }

    findThreeBlackCrows(data) {
        const patterns = [];
        
        for (let i = 2; i < data.length; i++) {
            const first = data[i - 2];
            const second = data[i - 1];
            const third = data[i];
            
            if (this.isThreeBlackCrowsPattern(first, second, third)) {
                patterns.push({
                    type: 'Three Black Crows',
                    position: i,
                    confidence: this.calculateThreeBlackCrowsConfidence(first, second, third),
                    timestamp: new Date()
                });
            }
        }
        
        return patterns;
    }

    findFibonacciExtensions(prices) {
        const patterns = [];
        const fibLevels = [1.618, 2.618, 3.618, 4.236];
        
        for (let i = 2; i < prices.length; i++) {
            const swing = this.identifySwing(prices, i);
            if (swing) {
                const extensionLevels = this.calculateFibonacciExtensionLevels(
                    swing.start,
                    swing.high,
                    swing.low
                );
                
                if (this.validateFibonacciExtension(prices[i], extensionLevels)) {
                    patterns.push({
                        type: 'Fibonacci Extension',
                        levels: extensionLevels,
                        confidence: this.calculateFibonacciConfidence(prices[i], extensionLevels),
                        timestamp: new Date()
                    });
                }
            }
        }
        
        return patterns;
    }

    findFibonacciTimeZones(data) {
        const patterns = [];
        const fibSequence = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
        
        const pivots = this.findSignificantPivots(data);
        if (pivots.length > 0) {
            const timeZones = this.calculateFibonacciTimeZones(pivots[0].timestamp, fibSequence);
            
            patterns.push({
                type: 'Fibonacci Time Zones',
                zones: timeZones,
                confidence: this.calculateTimeZoneConfidence(data, timeZones),
                timestamp: new Date()
            });
        }
        
        return patterns;
    }

    findFibonacciFans(data) {
        const patterns = [];
        const fanLevels = [0.382, 0.5, 0.618];
        
        const pivots = this.findSignificantPivots(data);
        for (let i = 1; i < pivots.length; i++) {
            const fanLines = this.calculateFibonacciFanLines(
                pivots[i - 1],
                pivots[i],
                fanLevels
            );
            
            if (this.validateFibonacciFan(data, fanLines)) {
                patterns.push({
                    type: 'Fibonacci Fan',
                    lines: fanLines,
                    confidence: this.calculateFanConfidence(data, fanLines),
                    timestamp: new Date()
                });
            }
        }
        
        return patterns;
    }

    findSignificantPivots(data) {
        const pivots = [];
        const windowSize = 5;
        
        for (let i = windowSize; i < data.length - windowSize; i++) {
            const window = data.slice(i - windowSize, i + windowSize + 1);
            if (this.isPivotHigh(window) || this.isPivotLow(window)) {
                pivots.push({
                    price: data[i].close,
                    timestamp: data[i].timestamp,
                    type: this.isPivotHigh(window) ? 'high' : 'low'
                });
            }
        }
        
        return pivots;
    }

    isPivotHigh(window) {
        const middle = Math.floor(window.length / 2);
        const centerPrice = window[middle].high;
        
        return window.every((candle, i) => 
            i === middle || candle.high <= centerPrice
        );
    }

    isPivotLow(window) {
        const middle = Math.floor(window.length / 2);
        const centerPrice = window[middle].low;
        
        return window.every((candle, i) => 
            i === middle || candle.low >= centerPrice
        );
    }

    calculateFibonacciLevels(high, low) {
        const diff = high - low;
        const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
        
        return levels.map(level => ({
            level,
            price: high - (diff * level)
        }));
    }

    calculateFibonacciExtensionLevels(start, high, low) {
        const diff = high - low;
        const levels = [1, 1.618, 2.618, 3.618, 4.236];
        
        return levels.map(level => ({
            level,
            price: start + (diff * level)
        }));
    }

    calculateFibonacciTimeZones(startTime, sequence) {
        const baseInterval = 24 * 60 * 60 * 1000; // 1 day in milliseconds
        
        return sequence.map(multiplier => ({
            multiplier,
            timestamp: new Date(startTime.getTime() + (baseInterval * multiplier))
        }));
    }

    calculateFibonacciFanLines(startPivot, endPivot, levels) {
        const timeRange = endPivot.timestamp - startPivot.timestamp;
        const priceRange = endPivot.price - startPivot.price;
        
        return levels.map(level => ({
            level,
            slope: (priceRange * level) / timeRange,
            startPoint: startPivot
        }));
    }

    isHaramiPattern(previous, current) {
        const prevBodySize = Math.abs(previous.close - previous.open);
        const currBodySize = Math.abs(current.close - current.open);
        
        const isPrevBullish = previous.close > previous.open;
        const isCurrBullish = current.close > current.open;
        
        // Previous candle should be larger than current
        if (currBodySize >= prevBodySize) return false;
        
        // Opposite colors
        if (isPrevBullish === isCurrBullish) return false;
        
        // Current candle body should be inside previous candle body
        const prevMax = Math.max(previous.open, previous.close);
        const prevMin = Math.min(previous.open, previous.close);
        const currMax = Math.max(current.open, current.close);
        const currMin = Math.min(current.open, current.close);
        
        return currMax <= prevMax && currMin >= prevMin;
    }

    isTweezerPattern(previous, current, tolerance) {
        const isPrevBullish = previous.close > previous.open;
        const isCurrBullish = current.close > current.open;
        
        // Opposite colors
        if (isPrevBullish === isCurrBullish) return false;
        
        // Check for similar highs or lows within tolerance
        const highDiff = Math.abs(previous.high - current.high) / previous.high;
        const lowDiff = Math.abs(previous.low - current.low) / previous.low;
        
        return (highDiff <= tolerance && isPrevBullish) || 
               (lowDiff <= tolerance && !isPrevBullish);
    }

    isMorningStarPattern(first, second, third) {
        // First candle should be bearish
        if (first.close >= first.open) return false;
        
        // Third candle should be bullish
        if (third.close <= third.open) return false;
        
        // Second candle should have a small body
        const secondBodySize = Math.abs(second.close - second.open);
        const firstBodySize = Math.abs(first.close - first.open);
        if (secondBodySize >= firstBodySize * 0.3) return false;
        
        // Gap down between first and second
        const firstLow = Math.min(first.open, first.close);
        const secondHigh = Math.max(second.open, second.close);
        if (secondHigh >= firstLow) return false;
        
        // Gap up between second and third
        const secondLow = Math.min(second.open, second.close);
        const thirdHigh = Math.max(third.open, third.close);
        if (thirdHigh <= secondLow) return false;
        
        return true;
    }

    isEveningStarPattern(first, second, third) {
        // First candle should be bullish
        if (first.close <= first.open) return false;
        
        // Third candle should be bearish
        if (third.close >= third.open) return false;
        
        // Second candle should have a small body
        const secondBodySize = Math.abs(second.close - second.open);
        const firstBodySize = Math.abs(first.close - first.open);
        if (secondBodySize >= firstBodySize * 0.3) return false;
        
        // Gap up between first and second
        const firstHigh = Math.max(first.open, first.close);
        const secondLow = Math.min(second.open, second.close);
        if (secondLow <= firstHigh) return false;
        
        // Gap down between second and third
        const secondHigh = Math.max(second.open, second.close);
        const thirdLow = Math.min(third.open, third.close);
        if (thirdLow >= secondHigh) return false;
        
        return true;
    }

    isThreeWhiteSoldiersPattern(first, second, third) {
        // All candles should be bullish
        if (first.close <= first.open || 
            second.close <= second.open || 
            third.close <= third.open) return false;
        
        // Each candle should open within previous candle's body
        if (second.open < first.open || third.open < second.open) return false;
        
        // Each candle should close higher than previous
        if (second.close <= first.close || third.close <= second.close) return false;
        
        // Small upper shadows
        const maxShadowRatio = 0.1;
        if (!this.hasSmallUpperShadow(first, maxShadowRatio) ||
            !this.hasSmallUpperShadow(second, maxShadowRatio) ||
            !this.hasSmallUpperShadow(third, maxShadowRatio)) return false;
        
        return true;
    }

    isThreeBlackCrowsPattern(first, second, third) {
        // All candles should be bearish
        if (first.close >= first.open || 
            second.close >= second.open || 
            third.close >= third.open) return false;
        
        // Each candle should open within previous candle's body
        if (second.open > first.open || third.open > second.open) return false;
        
        // Each candle should close lower than previous
        if (second.close >= first.close || third.close >= second.close) return false;
        
        // Small lower shadows
        const maxShadowRatio = 0.1;
        if (!this.hasSmallLowerShadow(first, maxShadowRatio) ||
            !this.hasSmallLowerShadow(second, maxShadowRatio) ||
            !this.hasSmallLowerShadow(third, maxShadowRatio)) return false;
        
        return true;
    }

    hasSmallUpperShadow(candle, maxRatio) {
        const bodySize = Math.abs(candle.close - candle.open);
        const upperShadow = candle.high - Math.max(candle.open, candle.close);
        return upperShadow <= bodySize * maxRatio;
    }

    hasSmallLowerShadow(candle, maxRatio) {
        const bodySize = Math.abs(candle.close - candle.open);
        const lowerShadow = Math.min(candle.open, candle.close) - candle.low;
        return lowerShadow <= bodySize * maxRatio;
    }

    calculateHammerConfidence(candle) {
        const bodySize = Math.abs(candle.close - candle.open);
        const upperShadow = candle.high - Math.max(candle.open, candle.close);
        const lowerShadow = Math.min(candle.open, candle.close) - candle.low;
        
        // Calculate ratios
        const shadowRatio = lowerShadow / bodySize;
        const upperShadowRatio = upperShadow / bodySize;
        
        // Perfect hammer has large lower shadow and minimal upper shadow
        let confidence = 0;
        
        // Lower shadow should be at least 2x body
        confidence += Math.min(1, shadowRatio / 2) * 0.4;
        
        // Upper shadow should be minimal
        confidence += (1 - Math.min(1, upperShadowRatio)) * 0.3;
        
        // Body size relative to average
        confidence += this.calculateRelativeBodySize(bodySize) * 0.3;
        
        return confidence;
    }

    calculateShootingStarConfidence(candle) {
        const bodySize = Math.abs(candle.close - candle.open);
        const upperShadow = candle.high - Math.max(candle.open, candle.close);
        const lowerShadow = Math.min(candle.open, candle.close) - candle.low;
        
        // Calculate ratios
        const shadowRatio = upperShadow / bodySize;
        const lowerShadowRatio = lowerShadow / bodySize;
        
        let confidence = 0;
        
        // Upper shadow should be at least 2x body
        confidence += Math.min(1, shadowRatio / 2) * 0.4;
        
        // Lower shadow should be minimal
        confidence += (1 - Math.min(1, lowerShadowRatio)) * 0.3;
        
        // Body size relative to average
        confidence += this.calculateRelativeBodySize(bodySize) * 0.3;
        
        return confidence;
    }

    calculateMarubozuConfidence(candle) {
        const bodySize = Math.abs(candle.close - candle.open);
        const upperShadow = candle.high - Math.max(candle.open, candle.close);
        const lowerShadow = Math.min(candle.open, candle.close) - candle.low;
        
        // Calculate shadow ratios
        const upperShadowRatio = upperShadow / bodySize;
        const lowerShadowRatio = lowerShadow / bodySize;
        
        let confidence = 0;
        
        // Minimal shadows
        confidence += (1 - Math.min(1, upperShadowRatio)) * 0.4;
        confidence += (1 - Math.min(1, lowerShadowRatio)) * 0.4;
        
        // Body size relative to average
        confidence += this.calculateRelativeBodySize(bodySize) * 0.2;
        
        return confidence;
    }

    calculateHaramiConfidence(previous, current) {
        const prevBodySize = Math.abs(previous.close - previous.open);
        const currBodySize = Math.abs(current.close - current.open);
        
        let confidence = 0;
        
        // Size ratio between bodies
        confidence += Math.min(1, (prevBodySize / currBodySize) / 2) * 0.4;
        
        // Centrality of current candle within previous
        confidence += this.calculateCentrality(previous, current) * 0.4;
        
        // Body sizes relative to average
        confidence += this.calculateRelativeBodySize(prevBodySize) * 0.1;
        confidence += this.calculateRelativeBodySize(currBodySize) * 0.1;
        
        return confidence;
    }

    calculateCentrality(previous, current) {
        const prevMax = Math.max(previous.open, previous.close);
        const prevMin = Math.min(previous.open, previous.close);
        const currMax = Math.max(current.open, current.close);
        const currMin = Math.min(current.open, current.close);
        
        const prevRange = prevMax - prevMin;
        const currRange = currMax - currMin;
        
        const idealCenter = prevMin + (prevRange / 2);
        const actualCenter = currMin + (currRange / 2);
        
        const deviation = Math.abs(idealCenter - actualCenter) / prevRange;
        return 1 - Math.min(1, deviation * 2);
    }

    calculateRelativeBodySize(bodySize) {
        // Compare to average body size of recent candles
        const avgBodySize = this.getAverageBodySize();
        const ratio = bodySize / avgBodySize;
        
        // Normalize between 0 and 1
        return Math.min(1, ratio);
    }

    getAverageBodySize() {
        // Implementation depends on how historical data is stored
        // Return a reasonable default if not available
        return 100;
    }

    scanForPatterns(data) {
        const patterns = [];
        const minConfidence = 0.7; // Minimum confidence threshold
        
        // Scan for single candlestick patterns
        for (let i = 0; i < data.length; i++) {
            const candle = data[i];
            
            // Check for Hammer
            const hammerConfidence = this.calculateHammerConfidence(candle);
            if (hammerConfidence >= minConfidence) {
                patterns.push({
                    type: 'Hammer',
                    confidence: hammerConfidence,
                    index: i,
                    timestamp: candle.timestamp
                });
            }
            
            // Check for Shooting Star
            const shootingStarConfidence = this.calculateShootingStarConfidence(candle);
            if (shootingStarConfidence >= minConfidence) {
                patterns.push({
                    type: 'Shooting Star',
                    confidence: shootingStarConfidence,
                    index: i,
                    timestamp: candle.timestamp
                });
            }
            
            // Check for Marubozu
            const marubozuConfidence = this.calculateMarubozuConfidence(candle);
            if (marubozuConfidence >= minConfidence) {
                patterns.push({
                    type: 'Marubozu',
                    confidence: marubozuConfidence,
                    index: i,
                    timestamp: candle.timestamp
                });
            }
        }
        
        // Scan for two candlestick patterns
        for (let i = 1; i < data.length; i++) {
            const current = data[i];
            const previous = data[i-1];
            
            // Check for Harami
            if (this.isHaramiPattern(previous, current)) {
                const confidence = this.calculateHaramiConfidence(previous, current);
                if (confidence >= minConfidence) {
                    patterns.push({
                        type: 'Harami',
                        confidence: confidence,
                        index: i,
                        timestamp: current.timestamp
                    });
                }
            }
            
            // Check for Tweezer
            if (this.isTweezerPattern(previous, current, 0.001)) {
                patterns.push({
                    type: 'Tweezer',
                    confidence: 0.8, // Fixed confidence for Tweezers
                    index: i,
                    timestamp: current.timestamp
                });
            }
        }
        
        // Scan for three candlestick patterns
        for (let i = 2; i < data.length; i++) {
            const first = data[i-2];
            const second = data[i-1];
            const third = data[i];
            
            // Check for Morning Star
            if (this.isMorningStarPattern(first, second, third)) {
                patterns.push({
                    type: 'Morning Star',
                    confidence: 0.9, // High confidence for Morning Star
                    index: i,
                    timestamp: third.timestamp
                });
            }
            
            // Check for Evening Star
            if (this.isEveningStarPattern(first, second, third)) {
                patterns.push({
                    type: 'Evening Star',
                    confidence: 0.9, // High confidence for Evening Star
                    index: i,
                    timestamp: third.timestamp
                });
            }
            
            // Check for Three White Soldiers
            if (this.isThreeWhiteSoldiersPattern(first, second, third)) {
                patterns.push({
                    type: 'Three White Soldiers',
                    confidence: 0.95, // Very high confidence
                    index: i,
                    timestamp: third.timestamp
                });
            }
            
            // Check for Three Black Crows
            if (this.isThreeBlackCrowsPattern(first, second, third)) {
                patterns.push({
                    type: 'Three Black Crows',
                    confidence: 0.95, // Very high confidence
                    index: i,
                    timestamp: third.timestamp
                });
            }
        }
        
        return patterns;
    }

    updatePatternDisplay(patterns) {
        const container = document.getElementById('pattern-container');
        if (!container) return;
        
        // Clear existing patterns
        container.innerHTML = '';
        
        // Sort patterns by timestamp (most recent first)
        patterns.sort((a, b) => b.timestamp - a.timestamp);
        
        // Create pattern elements
        patterns.forEach(pattern => {
            const patternElement = document.createElement('div');
            patternElement.className = 'pattern-item';
            
            const confidencePercentage = Math.round(pattern.confidence * 100);
            const timestamp = new Date(pattern.timestamp).toLocaleString();
            
            patternElement.innerHTML = `
                <div class="pattern-header">
                    <span class="pattern-type">${pattern.type}</span>
                    <span class="pattern-confidence">${confidencePercentage}%</span>
                </div>
                <div class="pattern-details">
                    <span class="pattern-timestamp">${timestamp}</span>
                    <button class="pattern-analyze-btn" data-index="${pattern.index}">
                        Analyze
                    </button>
                </div>
            `;
            
            // Add click handler for the analyze button
            const analyzeBtn = patternElement.querySelector('.pattern-analyze-btn');
            analyzeBtn.addEventListener('click', () => {
                this.showPatternAnalysis(pattern);
            });
            
            container.appendChild(patternElement);
        });
        
        // Update pattern count
        const countElement = document.getElementById('pattern-count');
        if (countElement) {
            countElement.textContent = patterns.length;
        }
    }

    showPatternAnalysis(pattern) {
        const modal = document.getElementById('pattern-analysis-modal');
        const content = document.getElementById('pattern-analysis-content');
        if (!modal || !content) return;
        
        // Get pattern description and trading implications
        const analysis = this.getPatternAnalysis(pattern);
        
        content.innerHTML = `
            <h3>${pattern.type} Pattern Analysis</h3>
            <div class="analysis-section">
                <h4>Description</h4>
                <p>${analysis.description}</p>
            </div>
            <div class="analysis-section">
                <h4>Trading Implications</h4>
                <p>${analysis.implications}</p>
            </div>
            <div class="analysis-section">
                <h4>Confidence Level</h4>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${pattern.confidence * 100}%"></div>
                    <span>${Math.round(pattern.confidence * 100)}%</span>
                </div>
            </div>
            <div class="analysis-section">
                <h4>Supporting Metrics</h4>
                <ul>
                    ${analysis.metrics.map(metric => `
                        <li>
                            <span class="metric-name">${metric.name}:</span>
                            <span class="metric-value">${metric.value}</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
        
        modal.style.display = 'block';
    }

    getPatternAnalysis(pattern) {
        // Pattern descriptions and implications
        const analyses = {
            'Hammer': {
                description: 'A single candlestick pattern with a small body and a long lower shadow, typically indicating a potential bullish reversal.',
                implications: 'Suggests buying pressure emerging after a downtrend, often leading to a trend reversal.',
                metrics: [
                    { name: 'Shadow Ratio', value: '2.5x body' },
                    { name: 'Trend Direction', value: 'Downtrend' },
                    { name: 'Volume', value: '1.5x average' }
                ]
            },
            'Shooting Star': {
                description: 'A single candlestick pattern with a small body and a long upper shadow, typically indicating a potential bearish reversal.',
                implications: 'Suggests selling pressure emerging after an uptrend, often leading to a trend reversal.',
                metrics: [
                    { name: 'Shadow Ratio', value: '2.5x body' },
                    { name: 'Trend Direction', value: 'Uptrend' },
                    { name: 'Volume', value: '1.3x average' }
                ]
            },
            // Add analyses for other patterns...
        };
        
        return analyses[pattern.type] || {
            description: 'Pattern analysis not available.',
            implications: 'Trading implications not available.',
            metrics: []
        };
    }

    initializeBotLearning() {
        this.botLearning = {
            successfulTrades: [],
            tradeParameters: new Map(),
            performanceMetrics: new Map(),
            optimizationSuggestions: [],
            learningThresholds: {
                minProfitUSD: 5.0,
                minROI: 0.02, // 2%
                maxGasUsage: 500000,
                maxSlippage: 0.005, // 0.5%
                minLiquidity: 50000 // $50k minimum liquidity
            }
        };

        // Initialize learning system
        this.setupLearningSystem();
    }

    setupLearningSystem() {
        // Monitor successful trades
        setInterval(() => this.analyzeTrades(), 60000); // Every minute
        
        // Generate optimization suggestions
        setInterval(() => this.generateOptimizations(), 300000); // Every 5 minutes
        
        // Update learning thresholds based on performance
        setInterval(() => this.updateLearningThresholds(), 3600000); // Every hour
    }

    analyzeTrades() {
        const recentTrades = this.getRecentTrades();
        const successfulTrades = recentTrades.filter(trade => 
            trade.profit >= this.botLearning.learningThresholds.minProfitUSD &&
            trade.roi >= this.botLearning.learningThresholds.minROI &&
            trade.gasUsed <= this.botLearning.learningThresholds.maxGasUsage &&
            trade.slippage <= this.botLearning.learningThresholds.maxSlippage
        );

        successfulTrades.forEach(trade => {
            this.analyzeTradingParameters(trade);
            this.updatePerformanceMetrics(trade);
            this.identifySuccessPatterns(trade);
        });
    }

    analyzeTradingParameters(trade) {
        const parameters = {
            tokenPair: trade.tokenPair,
            exchangePair: [trade.sourceExchange, trade.targetExchange],
            tradeSize: trade.amount,
            gasPrice: trade.gasPrice,
            slippageTolerance: trade.slippageTolerance,
            routingStrategy: trade.routingStrategy,
            timing: {
                timeOfDay: new Date(trade.timestamp).getHours(),
                dayOfWeek: new Date(trade.timestamp).getDay()
            }
        };

        // Update successful parameters map
        this.updateParameterSuccess(parameters);
    }

    updateParameterSuccess(parameters) {
        const key = JSON.stringify(parameters);
        const current = this.botLearning.tradeParameters.get(key) || {
            count: 0,
            totalProfit: 0,
            averageROI: 0,
            successRate: 0
        };

        current.count++;
        current.totalProfit += parameters.profit;
        current.averageROI = current.totalProfit / (current.count * parameters.tradeSize);
        current.successRate = current.count / this.getTotalTradesForParameters(parameters);

        this.botLearning.tradeParameters.set(key, current);
    }

    generateOptimizations() {
        const suggestions = [];
        
        // Analyze token pair performance
        this.analyzeTokenPairPerformance(suggestions);
        
        // Analyze exchange pair performance
        this.analyzeExchangePairPerformance(suggestions);
        
        // Analyze trade size optimization
        this.analyzeTradeSize(suggestions);
        
        // Analyze gas and timing optimization
        this.analyzeGasAndTiming(suggestions);
        
        // Update optimization suggestions
        this.botLearning.optimizationSuggestions = suggestions;
        
        // Update UI with suggestions
        this.updateOptimizationDisplay();
    }

    analyzeTokenPairPerformance(suggestions) {
        const pairStats = new Map();
        
        // Aggregate statistics by token pair
        this.botLearning.successfulTrades.forEach(trade => {
            const pair = trade.tokenPair;
            const stats = pairStats.get(pair) || {
                trades: 0,
                totalProfit: 0,
                averageROI: 0,
                failureRate: 0
            };
            
            stats.trades++;
            stats.totalProfit += trade.profit;
            stats.averageROI = stats.totalProfit / (stats.trades * trade.amount);
            stats.failureRate = this.calculateFailureRate(pair);
            
            pairStats.set(pair, stats);
        });
        
        // Generate suggestions based on pair performance
        pairStats.forEach((stats, pair) => {
            if (stats.failureRate > 0.3) { // More than 30% failure rate
                suggestions.push({
                    type: 'TOKEN_PAIR',
                    severity: 'high',
                    message: `Consider increasing minimum liquidity threshold for ${pair}. Current failure rate: ${(stats.failureRate * 100).toFixed(1)}%`,
                    action: 'ADJUST_LIQUIDITY_THRESHOLD',
                    parameters: { pair, suggestedThreshold: this.calculateSuggestedLiquidity(stats) }
                });
            }
            
            if (stats.averageROI < this.botLearning.learningThresholds.minROI) {
                suggestions.push({
                    type: 'TOKEN_PAIR',
                    severity: 'medium',
                    message: `${pair} showing low ROI (${(stats.averageROI * 100).toFixed(2)}%). Consider adjusting minimum profit threshold.`,
                    action: 'ADJUST_PROFIT_THRESHOLD',
                    parameters: { pair, suggestedThreshold: this.calculateSuggestedProfitThreshold(stats) }
                });
            }
        });
    }

    analyzeExchangePairPerformance(suggestions) {
        const exchangeStats = new Map();
        
        // Analyze performance by exchange pair
        this.botLearning.successfulTrades.forEach(trade => {
            const exchangePair = `${trade.sourceExchange}-${trade.targetExchange}`;
            const stats = exchangeStats.get(exchangePair) || {
                trades: 0,
                totalProfit: 0,
                averageGas: 0,
                failureRate: 0
            };
            
            stats.trades++;
            stats.totalProfit += trade.profit;
            stats.averageGas = ((stats.averageGas * (stats.trades - 1)) + trade.gasUsed) / stats.trades;
            stats.failureRate = this.calculateExchangeFailureRate(trade.sourceExchange, trade.targetExchange);
            
            exchangeStats.set(exchangePair, stats);
        });
        
        // Generate exchange-specific suggestions
        exchangeStats.forEach((stats, exchangePair) => {
            if (stats.averageGas > this.botLearning.learningThresholds.maxGasUsage * 0.8) {
                suggestions.push({
                    type: 'EXCHANGE_PAIR',
                    severity: 'medium',
                    message: `High gas usage on ${exchangePair}. Consider optimizing route or increasing profit threshold to compensate.`,
                    action: 'OPTIMIZE_ROUTE',
                    parameters: { exchangePair, currentGas: stats.averageGas }
                });
            }
            
            if (stats.failureRate > 0.2) { // More than 20% failure rate
                suggestions.push({
                    type: 'EXCHANGE_PAIR',
                    severity: 'high',
                    message: `High failure rate on ${exchangePair}. Consider implementing additional safety checks.`,
                    action: 'IMPLEMENT_SAFETY_CHECKS',
                    parameters: { exchangePair, failureRate: stats.failureRate }
                });
            }
        });
    }

    analyzeTradeSize(suggestions) {
        const sizeStats = new Map();
        const sizeBuckets = [1000, 5000, 10000, 50000, 100000]; // USD trade size buckets
        
        // Analyze performance by trade size
        this.botLearning.successfulTrades.forEach(trade => {
            const bucket = this.getTradeSize(trade.amount, sizeBuckets);
            const stats = sizeStats.get(bucket) || {
                trades: 0,
                totalProfit: 0,
                averageROI: 0,
                successRate: 0
            };
            
            stats.trades++;
            stats.totalProfit += trade.profit;
            stats.averageROI = stats.totalProfit / (stats.trades * trade.amount);
            stats.successRate = this.calculateSuccessRate(bucket);
            
            sizeStats.set(bucket, stats);
        });
        
        // Generate trade size optimization suggestions
        sizeStats.forEach((stats, bucket) => {
            if (stats.averageROI > this.botLearning.learningThresholds.minROI * 1.5) {
                suggestions.push({
                    type: 'TRADE_SIZE',
                    severity: 'low',
                    message: `Consider increasing trade size for ${bucket} USD bucket. Current ROI: ${(stats.averageROI * 100).toFixed(2)}%`,
                    action: 'INCREASE_TRADE_SIZE',
                    parameters: { bucket, suggestedIncrease: this.calculateSuggestedSizeIncrease(stats) }
                });
            }
            
            if (stats.successRate < 0.7) { // Less than 70% success rate
                suggestions.push({
                    type: 'TRADE_SIZE',
                    severity: 'medium',
                    message: `Consider reducing trade size for ${bucket} USD bucket to improve success rate.`,
                    action: 'DECREASE_TRADE_SIZE',
                    parameters: { bucket, suggestedDecrease: this.calculateSuggestedSizeDecrease(stats) }
                });
            }
        });
    }

    analyzeGasAndTiming(suggestions) {
        const timeStats = new Map();
        
        // Analyze performance by time of day
        this.botLearning.successfulTrades.forEach(trade => {
            const hour = new Date(trade.timestamp).getHours();
            const stats = timeStats.get(hour) || {
                trades: 0,
                totalProfit: 0,
                averageGas: 0,
                successRate: 0
            };
            
            stats.trades++;
            stats.totalProfit += trade.profit;
            stats.averageGas = ((stats.averageGas * (stats.trades - 1)) + trade.gasUsed) / stats.trades;
            stats.successRate = this.calculateHourlySuccessRate(hour);
            
            timeStats.set(hour, stats);
        });
        
        // Generate timing and gas optimization suggestions
        timeStats.forEach((stats, hour) => {
            if (stats.averageGas > this.botLearning.learningThresholds.maxGasUsage) {
                suggestions.push({
                    type: 'GAS_OPTIMIZATION',
                    severity: 'medium',
                    message: `High gas usage during ${hour}:00. Consider adjusting gas price strategy or avoiding this time period.`,
                    action: 'OPTIMIZE_GAS_STRATEGY',
                    parameters: { hour, currentGas: stats.averageGas }
                });
            }
            
            if (stats.successRate < 0.5) { // Less than 50% success rate
                suggestions.push({
                    type: 'TIMING_OPTIMIZATION',
                    severity: 'high',
                    message: `Poor performance during ${hour}:00. Consider suspending operations during this hour.`,
                    action: 'ADJUST_OPERATING_HOURS',
                    parameters: { hour, successRate: stats.successRate }
                });
            }
        });
    }

    updateOptimizationDisplay() {
        const container = document.getElementById('optimization-suggestions');
        if (!container) return;
        
        container.innerHTML = '';
        
        this.botLearning.optimizationSuggestions
            .sort((a, b) => this.getSeverityWeight(b.severity) - this.getSeverityWeight(a.severity))
            .forEach(suggestion => {
                const element = this.createSuggestionElement(suggestion);
                container.appendChild(element);
            });
    }

    createSuggestionElement(suggestion) {
        const element = document.createElement('div');
        element.className = `suggestion-item suggestion-${suggestion.severity}`;
        
        element.innerHTML = `
            <div class="suggestion-header">
                <span class="suggestion-type">${suggestion.type}</span>
                <span class="suggestion-severity">${suggestion.severity}</span>
            </div>
            <div class="suggestion-message">${suggestion.message}</div>
            <div class="suggestion-actions">
                <button onclick="window.dashboardManager.applySuggestion('${suggestion.action}', ${JSON.stringify(suggestion.parameters)})">
                    Apply
                </button>
                <button onclick="window.dashboardManager.dismissSuggestion('${suggestion.action}')">
                    Dismiss
                </button>
            </div>
        `;
        
        return element;
    }

    getSeverityWeight(severity) {
        const weights = {
            high: 3,
            medium: 2,
            low: 1
        };
        return weights[severity] || 0;
    }

    applySuggestion(action, parameters) {
        switch (action) {
            case 'ADJUST_LIQUIDITY_THRESHOLD':
                this.updateLiquidityThreshold(parameters.pair, parameters.suggestedThreshold);
                break;
            case 'ADJUST_PROFIT_THRESHOLD':
                this.updateProfitThreshold(parameters.pair, parameters.suggestedThreshold);
                break;
            case 'OPTIMIZE_ROUTE':
                this.optimizeRoute(parameters.exchangePair);
                break;
            case 'IMPLEMENT_SAFETY_CHECKS':
                this.implementSafetyChecks(parameters.exchangePair);
                break;
            case 'INCREASE_TRADE_SIZE':
                this.adjustTradeSize(parameters.bucket, parameters.suggestedIncrease);
                break;
            case 'DECREASE_TRADE_SIZE':
                this.adjustTradeSize(parameters.bucket, -parameters.suggestedDecrease);
                break;
            case 'OPTIMIZE_GAS_STRATEGY':
                this.optimizeGasStrategy(parameters.hour);
                break;
            case 'ADJUST_OPERATING_HOURS':
                this.adjustOperatingHours(parameters.hour);
                break;
        }
    }

    // Helper functions for parameter adjustments
    updateLiquidityThreshold(pair, threshold) {
        this.botLearning.learningThresholds.minLiquidity = Math.max(
            threshold,
            this.botLearning.learningThresholds.minLiquidity
        );
        this.notifyParameterUpdate('Liquidity threshold updated', { pair, threshold });
    }

    updateProfitThreshold(pair, threshold) {
        this.botLearning.learningThresholds.minProfitUSD = Math.max(
            threshold,
            this.botLearning.learningThresholds.minProfitUSD
        );
        this.notifyParameterUpdate('Profit threshold updated', { pair, threshold });
    }

    optimizeRoute(exchangePair) {
        // Implement route optimization logic
        this.notifyParameterUpdate('Route optimization initiated', { exchangePair });
    }

    implementSafetyChecks(exchangePair) {
        // Implement additional safety checks
        this.notifyParameterUpdate('Safety checks implemented', { exchangePair });
    }

    adjustTradeSize(bucket, adjustment) {
        // Implement trade size adjustment logic
        this.notifyParameterUpdate('Trade size adjusted', { bucket, adjustment });
    }

    optimizeGasStrategy(hour) {
        // Implement gas strategy optimization
        this.notifyParameterUpdate('Gas strategy optimized', { hour });
    }

    adjustOperatingHours(hour) {
        // Implement operating hours adjustment
        this.notifyParameterUpdate('Operating hours adjusted', { hour });
    }

    notifyParameterUpdate(message, parameters) {
        console.log('Parameter Update:', message, parameters);
        // Implement notification system for parameter updates
        this.showNotification({
            type: 'PARAMETER_UPDATE',
            message: message,
            parameters: parameters,
            timestamp: new Date()
        });
    }

    async initializePoolMonitoring() {
        this.setupPoolCharts();
        this.setupPoolMetricsGrid();
        this.setupPoolAlerts();
        await this.refreshPoolData();
        this.startPoolMonitoring();
    }

    setupPoolCharts() {
        // TVL Chart
        this.tvlChart = new Chart(document.getElementById('tvl-chart'), {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Total Value Locked',
                    borderColor: '#4CAF50',
                    fill: false
                }]
            },
            options: this.getChartOptions('TVL Over Time')
        });

        // Volume Chart
        this.volumeChart = new Chart(document.getElementById('volume-chart'), {
            type: 'bar',
            data: {
                datasets: [{
                    label: '24h Volume',
                    backgroundColor: '#2196F3'
                }]
            },
            options: this.getChartOptions('Daily Volume')
        });

        // Fee Chart
        this.feeChart = new Chart(document.getElementById('fee-chart'), {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Fee Revenue',
                    borderColor: '#FFC107',
                    fill: false
                }]
            },
            options: this.getChartOptions('Fee Revenue')
        });
    }

    setupPoolMetricsGrid() {
        const grid = document.getElementById('pool-metrics-grid');
        this.poolMetricsGrid = new GridJS({
            columns: [
                { name: 'Pool', sort: true },
                { name: 'TVL', sort: true },
                { name: 'Volume (24h)', sort: true },
                { name: 'Fees (24h)', sort: true },
                { name: 'APY', sort: true },
                { name: 'Utilization', sort: true },
                { name: 'Risk Score', sort: true },
                { name: 'Actions' }
            ],
            pagination: true,
            search: true,
            sort: true
        }).render(grid);
    }

    async refreshPoolData() {
        try {
            const [poolStats, poolMetrics] = await Promise.all([
                this.fetchPoolStats(),
                this.fetchPoolMetrics()
            ]);

            this.updatePoolCharts(poolStats);
            this.updatePoolMetricsGrid(poolMetrics);
            this.checkPoolAlerts(poolMetrics);
        } catch (error) {
            console.error('Failed to refresh pool data:', error);
            this.showNotification('Failed to update pool data', 'error');
        }
    }

    updatePoolCharts(stats) {
        // Update TVL Chart
        this.tvlChart.data.labels = stats.tvl.map(d => d.timestamp);
        this.tvlChart.data.datasets[0].data = stats.tvl.map(d => d.value);
        this.tvlChart.update();

        // Update Volume Chart
        this.volumeChart.data.labels = stats.volume.map(d => d.timestamp);
        this.volumeChart.data.datasets[0].data = stats.volume.map(d => d.value);
        this.volumeChart.update();

        // Update Fee Chart
        this.feeChart.data.labels = stats.fees.map(d => d.timestamp);
        this.feeChart.data.datasets[0].data = stats.fees.map(d => d.value);
        this.feeChart.update();
    }

    updatePoolMetricsGrid(metrics) {
        const gridData = metrics.map(pool => [
            pool.name,
            this.formatCurrency(pool.tvl),
            this.formatCurrency(pool.volume24h),
            this.formatCurrency(pool.fees24h),
            this.formatPercentage(pool.apy),
            this.formatPercentage(pool.utilization),
            this.formatRiskScore(pool.riskScore),
            this.createPoolActions(pool)
        ]);

        this.poolMetricsGrid.updateConfig({ data: gridData }).forceRender();
    }

    createPoolActions(pool) {
        return `
            <div class="pool-actions">
                <button onclick="dashboard.showPoolDetails('${pool.address}')" class="btn-details">
                    Details
                </button>
                <button onclick="dashboard.analyzePool('${pool.address}')" class="btn-analyze">
                    Analyze
                </button>
            </div>
        `;
    }

    async showPoolDetails(poolAddress) {
        const details = await this.fetchPoolDetails(poolAddress);
        const modal = document.getElementById('pool-details-modal');
        
        modal.innerHTML = this.formatPoolDetails(details);
        modal.style.display = 'block';
    }

    formatPoolDetails(details) {
        return `
            <div class="pool-details">
                <h2>Pool Details</h2>
                <div class="metrics-grid">
                    ${this.createMetricCards(details)}
                </div>
                <div class="charts-container">
                    ${this.createDetailCharts(details)}
                </div>
                <div class="transactions-table">
                    ${this.createTransactionsTable(details.recentTransactions)}
                </div>
            </div>
        `;
    }

    setupPoolAlerts() {
        this.poolAlerts = {
            liquidityThreshold: 1000000, // $1M
            volumeChangeThreshold: 20, // 20%
            utilizationThreshold: 80, // 80%
            riskScoreThreshold: 0.7 // 70/100
        };
    }

    checkPoolAlerts(metrics) {
        const alerts = [];
        
        for (const pool of metrics) {
            if (pool.tvl < this.poolAlerts.liquidityThreshold) {
                alerts.push({
                    type: 'warning',
                    message: `Low liquidity in pool ${pool.name}`,
                    details: `Current TVL: ${this.formatCurrency(pool.tvl)}`
                });
            }

            if (Math.abs(pool.volumeChange24h) > this.poolAlerts.volumeChangeThreshold) {
                alerts.push({
                    type: 'info',
                    message: `Significant volume change in pool ${pool.name}`,
                    details: `${pool.volumeChange24h > 0 ? 'Increase' : 'Decrease'} of ${Math.abs(pool.volumeChange24h)}%`
                });
            }

            // Add more alert checks...
        }

        this.showPoolAlerts(alerts);
    }

    startPoolMonitoring() {
        setInterval(() => this.refreshPoolData(), 60000); // Refresh every minute
    }
}

// Initialize dashboard when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboardManager = new DashboardManager();
}); 