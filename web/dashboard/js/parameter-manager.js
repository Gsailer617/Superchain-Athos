class ParameterManager {
    constructor() {
        this.parameters = {
            profitOptimizer: {
                title: 'Profit Optimizer',
                params: {
                    MIN_CONFIDENCE: { value: 0.8, min: 0, max: 1, step: 0.05, description: 'Minimum confidence score required for trade execution' },
                    MIN_LIQUIDITY_SCORE: { value: 0.7, min: 0, max: 1, step: 0.05, description: 'Minimum liquidity score required for trading pairs' },
                    MAX_RISK_SCORE: { value: 0.3, min: 0, max: 1, step: 0.05, description: 'Maximum acceptable risk score for trades' },
                    MIN_GAS_EFFICIENCY: { value: 0.85, min: 0, max: 1, step: 0.05, description: 'Minimum gas efficiency score required' }
                }
            },
            environment: {
                title: 'Environment Settings',
                params: {
                    POLLING_INTERVAL: { value: 1000, min: 500, max: 5000, step: 100, description: 'Interval between market checks (ms)' },
                    MIN_PROFIT_USD: { value: 10, min: 1, max: 100, step: 1, description: 'Minimum profit required in USD' },
                    GAS_PRICE_LIMIT: { value: 50, min: 10, max: 200, step: 5, description: 'Maximum gas price in Gwei' }
                }
            },
            qualityThresholds: {
                title: 'Quality Thresholds',
                params: {
                    MIN_24H_VOLUME: { value: 1000000, min: 100000, max: 10000000, step: 100000, description: 'Minimum 24h trading volume in USD' },
                    MAX_PRICE_IMPACT: { value: 0.01, min: 0.001, max: 0.05, step: 0.001, description: 'Maximum acceptable price impact' },
                    MIN_TVL: { value: 500000, min: 100000, max: 5000000, step: 100000, description: 'Minimum Total Value Locked in USD' }
                }
            }
        };

        this.initializeUI();
        this.loadSavedParameters();
        this.setupEventListeners();
    }

    initializeUI() {
        const container = document.getElementById('parameter-settings');
        
        Object.entries(this.parameters).forEach(([category, categoryData]) => {
            const section = document.createElement('div');
            section.className = 'parameter-category';
            
            section.innerHTML = `
                <h3>${categoryData.title}</h3>
                <div class="parameter-grid">
                    ${Object.entries(categoryData.params).map(([key, param]) => `
                        <div class="parameter-container">
                            <label class="parameter-label" for="${key}">${this.formatLabel(key)}</label>
                            <div class="parameter-input-group">
                                <input type="number" 
                                    id="${key}"
                                    value="${param.value}"
                                    min="${param.min}"
                                    max="${param.max}"
                                    step="${param.step}"
                                    class="parameter-input"
                                >
                                <button class="reset-btn" data-param="${key}">Reset</button>
                            </div>
                            <div class="parameter-description">${param.description}</div>
                        </div>
                    `).join('')}
                </div>
            `;
            
            container.appendChild(section);
        });
    }

    formatLabel(key) {
        return key.split('_').map(word => 
            word.charAt(0) + word.slice(1).toLowerCase()
        ).join(' ');
    }

    setupEventListeners() {
        document.querySelectorAll('.parameter-input').forEach(input => {
            input.addEventListener('change', (e) => this.handleParameterChange(e));
        });

        document.querySelectorAll('.reset-btn').forEach(button => {
            button.addEventListener('click', (e) => this.handleParameterReset(e));
        });
    }

    async handleParameterChange(event) {
        const input = event.target;
        const paramKey = input.id;
        const newValue = parseFloat(input.value);

        if (this.validateParameter(paramKey, newValue)) {
            await this.updateParameter(paramKey, newValue);
            this.showNotification(`Updated ${this.formatLabel(paramKey)} to ${newValue}`, 'success');
        } else {
            this.showNotification(`Invalid value for ${this.formatLabel(paramKey)}`, 'error');
            this.resetParameter(paramKey);
        }
    }

    validateParameter(paramKey, value) {
        for (const category of Object.values(this.parameters)) {
            if (paramKey in category.params) {
                const param = category.params[paramKey];
                return value >= param.min && value <= param.max;
            }
        }
        return false;
    }

    async updateParameter(paramKey, value) {
        // Find and update the parameter in our local state
        for (const category of Object.values(this.parameters)) {
            if (paramKey in category.params) {
                category.params[paramKey].value = value;
                break;
            }
        }

        // Save to localStorage
        this.saveParameters();

        // Update bot configuration via API
        try {
            const response = await fetch('/api/update-parameters', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    parameter: paramKey,
                    value: value
                })
            });

            if (!response.ok) {
                throw new Error('Failed to update parameter on server');
            }
        } catch (error) {
            console.error('Error updating parameter:', error);
            this.showNotification('Failed to update parameter on server', 'error');
        }
    }

    handleParameterReset(event) {
        const paramKey = event.target.dataset.param;
        this.resetParameter(paramKey);
    }

    resetParameter(paramKey) {
        for (const category of Object.values(this.parameters)) {
            if (paramKey in category.params) {
                const input = document.getElementById(paramKey);
                input.value = category.params[paramKey].value;
                break;
            }
        }
    }

    saveParameters() {
        localStorage.setItem('botParameters', JSON.stringify(this.parameters));
    }

    loadSavedParameters() {
        const saved = localStorage.getItem('botParameters');
        if (saved) {
            const savedParams = JSON.parse(saved);
            
            // Update our parameters with saved values
            Object.entries(savedParams).forEach(([category, categoryData]) => {
                if (this.parameters[category]) {
                    Object.entries(categoryData.params).forEach(([key, param]) => {
                        if (this.parameters[category].params[key]) {
                            this.parameters[category].params[key].value = param.value;
                            const input = document.getElementById(key);
                            if (input) {
                                input.value = param.value;
                            }
                        }
                    });
                }
            });
        }
    }

    showNotification(message, type = 'success') {
        const notifications = document.getElementById('parameter-notifications');
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        notifications.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => {
                notifications.removeChild(notification);
            }, 300);
        }, 3000);
    }
}

// Initialize the parameter manager when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.parameterManager = new ParameterManager();
}); 