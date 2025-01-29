class BotLearningManager {
    constructor() {
        this.metrics = {
            successRate: 0,
            averageROI: 0,
            gasEfficiency: 0,
            optimalPairs: 0
        };
        this.parameters = {
            minProfitThreshold: 0.15,
            gasPriceLimit: 50,
            slippageTolerance: 0.5
        };
        this.suggestions = [];
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Refresh button
        document.getElementById('refreshBtn').addEventListener('click', () => this.refreshAnalysis());

        // Parameter inputs
        document.querySelectorAll('.parameter-input').forEach(input => {
            input.addEventListener('change', (e) => this.handleParameterChange(e));
        });

        // Suggestion action buttons
        document.querySelectorAll('.suggestion-item').forEach(item => {
            const applyBtn = item.querySelector('button:first-child');
            const dismissBtn = item.querySelector('button:last-child');
            
            applyBtn.addEventListener('click', () => this.applySuggestion(item));
            dismissBtn.addEventListener('click', () => this.dismissSuggestion(item));
        });

        // Parameter update notification close button
        document.querySelectorAll('.parameter-update-close').forEach(btn => {
            btn.addEventListener('click', (e) => this.closeParameterUpdate(e));
        });
    }

    async refreshAnalysis() {
        try {
            // Show loading state
            const refreshBtn = document.getElementById('refreshBtn');
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '<span class="icon">🔄</span> Analyzing...';

            // Fetch updated metrics and suggestions
            await this.fetchMetrics();
            await this.fetchSuggestions();
            
            // Update UI
            this.updateMetricsDisplay();
            this.updateSuggestionsDisplay();
            
            // Reset button state
            refreshBtn.disabled = false;
            refreshBtn.innerHTML = '<span class="icon">🔄</span> Refresh Analysis';
            
            // Show success notification
            this.showNotification('Analysis updated successfully', 'success');
        } catch (error) {
            console.error('Error refreshing analysis:', error);
            this.showNotification('Failed to update analysis', 'error');
        }
    }

    async fetchMetrics() {
        try {
            const response = await fetch('/api/bot/metrics');
            const data = await response.json();
            this.metrics = data;
        } catch (error) {
            console.error('Error fetching metrics:', error);
            throw error;
        }
    }

    async fetchSuggestions() {
        try {
            const response = await fetch('/api/bot/suggestions');
            const data = await response.json();
            this.suggestions = data;
        } catch (error) {
            console.error('Error fetching suggestions:', error);
            throw error;
        }
    }

    updateMetricsDisplay() {
        const { successRate, averageROI, gasEfficiency, optimalPairs } = this.metrics;
        
        // Update metric values
        document.querySelector('.metric-value:nth-child(1)').textContent = `${successRate}%`;
        document.querySelector('.metric-value:nth-child(2)').textContent = `${averageROI}%`;
        document.querySelector('.metric-value:nth-child(3)').textContent = `${gasEfficiency}%`;
        document.querySelector('.metric-value:nth-child(4)').textContent = optimalPairs;

        // Update trends (assuming we have previous values to compare)
        this.updateMetricTrend('success-rate', successRate);
        this.updateMetricTrend('average-roi', averageROI);
        this.updateMetricTrend('gas-efficiency', gasEfficiency);
        this.updateMetricTrend('optimal-pairs', optimalPairs);
    }

    updateMetricTrend(metricId, currentValue) {
        const trendElement = document.querySelector(`#${metricId} .metric-trend`);
        const previousValue = parseFloat(localStorage.getItem(`previous_${metricId}`)) || currentValue;
        const difference = currentValue - previousValue;
        
        trendElement.className = `metric-trend ${difference >= 0 ? 'positive' : 'negative'}`;
        trendElement.innerHTML = `
            <span class="icon">${difference >= 0 ? '↑' : '↓'}</span>
            <span>${Math.abs(difference).toFixed(2)}% this week</span>
        `;
        
        localStorage.setItem(`previous_${metricId}`, currentValue);
    }

    updateSuggestionsDisplay() {
        const container = document.querySelector('.optimization-container');
        container.innerHTML = '<h2>Optimization Suggestions</h2>';
        
        this.suggestions.forEach(suggestion => {
            container.appendChild(this.createSuggestionElement(suggestion));
        });
    }

    createSuggestionElement(suggestion) {
        const element = document.createElement('div');
        element.className = `suggestion-item suggestion-${suggestion.priority.toLowerCase()}`;
        element.innerHTML = `
            <div class="suggestion-header">
                <span class="suggestion-type">${suggestion.type}</span>
                <span class="suggestion-severity">${suggestion.priority} Priority</span>
            </div>
            <div class="suggestion-message">${suggestion.message}</div>
            <div class="suggestion-actions">
                <button>Apply Change</button>
                <button>Dismiss</button>
            </div>
        `;
        
        // Add event listeners
        const applyBtn = element.querySelector('button:first-child');
        const dismissBtn = element.querySelector('button:last-child');
        
        applyBtn.addEventListener('click', () => this.applySuggestion(element));
        dismissBtn.addEventListener('click', () => this.dismissSuggestion(element));
        
        return element;
    }

    async handleParameterChange(event) {
        const input = event.target;
        const parameterName = input.parentElement.querySelector('.parameter-label').textContent;
        const newValue = parseFloat(input.value);
        
        try {
            await this.updateParameter(parameterName, newValue);
            this.showParameterUpdate(parameterName, newValue);
        } catch (error) {
            console.error('Error updating parameter:', error);
            this.showNotification('Failed to update parameter', 'error');
        }
    }

    async updateParameter(name, value) {
        try {
            const response = await fetch('/api/bot/parameters', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ name, value })
            });
            
            if (!response.ok) {
                throw new Error('Failed to update parameter');
            }
            
            // Update local state
            this.parameters[this.getParameterKey(name)] = value;
        } catch (error) {
            console.error('Error updating parameter:', error);
            throw error;
        }
    }

    getParameterKey(displayName) {
        // Convert display name to camelCase key
        const key = displayName.toLowerCase()
            .replace(/[^a-zA-Z0-9]+(.)/g, (m, chr) => chr.toUpperCase());
        return key;
    }

    async applySuggestion(suggestionElement) {
        const type = suggestionElement.querySelector('.suggestion-type').textContent;
        const message = suggestionElement.querySelector('.suggestion-message').textContent;
        
        try {
            await this.applyOptimization(type, message);
            this.dismissSuggestion(suggestionElement);
            this.showNotification('Optimization applied successfully', 'success');
        } catch (error) {
            console.error('Error applying suggestion:', error);
            this.showNotification('Failed to apply optimization', 'error');
        }
    }

    async applyOptimization(type, message) {
        try {
            const response = await fetch('/api/bot/optimize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ type, message })
            });
            
            if (!response.ok) {
                throw new Error('Failed to apply optimization');
            }
            
            // Refresh analysis after applying optimization
            await this.refreshAnalysis();
        } catch (error) {
            console.error('Error applying optimization:', error);
            throw error;
        }
    }

    dismissSuggestion(suggestionElement) {
        suggestionElement.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => suggestionElement.remove(), 300);
    }

    showParameterUpdate(parameterName, newValue) {
        const updateElement = document.createElement('div');
        updateElement.className = 'parameter-update';
        updateElement.innerHTML = `
            <div class="parameter-update-message">
                ${parameterName} updated to ${newValue}
            </div>
            <button class="parameter-update-close">×</button>
        `;
        
        document.querySelector('.container').insertBefore(
            updateElement,
            document.querySelector('.parameter-update')
        );
        
        // Add close button listener
        updateElement.querySelector('.parameter-update-close')
            .addEventListener('click', () => this.closeParameterUpdate(updateElement));
        
        // Auto-remove after 5 seconds
        setTimeout(() => this.closeParameterUpdate(updateElement), 5000);
    }

    closeParameterUpdate(element) {
        if (element.target) {
            element = element.target.parentElement;
        }
        element.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => element.remove(), 300);
    }

    showNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'fadeOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// Initialize the manager when the page loads
window.addEventListener('DOMContentLoaded', () => {
    window.botLearningManager = new BotLearningManager();
}); 