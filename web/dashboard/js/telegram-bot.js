class TelegramBot {
    constructor(aiAgent) {
        this.aiAgent = aiAgent;
        this.token = process.env.TELEGRAM_BOT_TOKEN;
        this.chatIds = new Set();
        this.messageQueue = [];
        this.isProcessing = false;
        
        this.initializeBot();
    }

    async initializeBot() {
        try {
            // Initialize Telegram bot
            this.bot = new TelegramBot(this.token, { polling: true });
            
            // Set up command handlers
            this.setupCommandHandlers();
            
            // Start message processing
            this.startMessageProcessing();
            
            console.log('Telegram bot initialized successfully');
        } catch (error) {
            console.error('Error initializing Telegram bot:', error);
            throw error;
        }
    }

    setupCommandHandlers() {
        // Start command
        this.bot.onText(/\/start/, async (msg) => {
            const chatId = msg.chat.id;
            this.chatIds.add(chatId);
            
            await this.sendMessage(chatId, 
                'Welcome to the Flash Loan Arbitrage Bot! 🤖\n\n' +
                'I can help you monitor and manage your flash loan arbitrage trades.\n\n' +
                'Available commands:\n' +
                '/status - Get current bot status\n' +
                '/performance - View performance metrics\n' +
                '/settings - Adjust bot parameters\n' +
                '/help - Show this help message'
            );
        });

        // Status command
        this.bot.onText(/\/status/, async (msg) => {
            const chatId = msg.chat.id;
            const status = await this.aiAgent.getStatus();
            await this.sendMessage(chatId, this.formatStatus(status));
        });

        // Performance command
        this.bot.onText(/\/performance/, async (msg) => {
            const chatId = msg.chat.id;
            const performance = await this.aiAgent.getPerformanceMetrics();
            await this.sendMessage(chatId, this.formatPerformance(performance));
        });

        // Settings command
        this.bot.onText(/\/settings/, async (msg) => {
            const chatId = msg.chat.id;
            const settings = await this.aiAgent.getSettings();
            await this.sendMessage(chatId, this.formatSettings(settings));
        });

        // Help command
        this.bot.onText(/\/help/, async (msg) => {
            const chatId = msg.chat.id;
            await this.sendHelpMessage(chatId);
        });

        // Handle natural language messages
        this.bot.on('message', async (msg) => {
            if (!msg.text.startsWith('/')) {
                const chatId = msg.chat.id;
                await this.handleNaturalLanguage(chatId, msg.text);
            }
        });
    }

    async handleNaturalLanguage(chatId, text) {
        try {
            // Process message through AI agent
            const response = await this.aiAgent.processUserInput(text);
            
            // Format and send response
            await this.sendMessage(chatId, this.formatResponse(response));
        } catch (error) {
            console.error('Error handling natural language input:', error);
            await this.sendMessage(chatId, 'Sorry, I encountered an error processing your message.');
        }
    }

    formatStatus(status) {
        return `📊 Bot Status\n\n` +
               `Status: ${status.isActive ? '🟢 Active' : '🔴 Inactive'}\n` +
               `Uptime: ${this.formatDuration(status.uptime)}\n` +
               `Active Trades: ${status.activeTrades}\n` +
               `Last Trade: ${status.lastTrade ? this.formatTimeAgo(status.lastTrade) : 'N/A'}\n` +
               `Current Balance: ${this.formatCurrency(status.balance)}`;
    }

    formatPerformance(performance) {
        return `📈 Performance Metrics\n\n` +
               `Total Trades: ${performance.totalTrades}\n` +
               `Success Rate: ${performance.successRate}%\n` +
               `Total Profit: ${this.formatCurrency(performance.totalProfit)}\n` +
               `Average ROI: ${performance.averageROI}%\n` +
               `Gas Efficiency: ${performance.gasEfficiency}%`;
    }

    formatSettings(settings) {
        return `⚙️ Current Settings\n\n` +
               `Min Profit: ${settings.minProfit}%\n` +
               `Gas Limit: ${settings.gasLimit} Gwei\n` +
               `Slippage: ${settings.slippage}%\n` +
               `Active Pairs: ${settings.activePairs.join(', ')}\n\n` +
               `To change settings, use:\n` +
               `/set [parameter] [value]`;
    }

    async sendHelpMessage(chatId) {
        const helpText = 
            '🤖 Flash Loan Arbitrage Bot Help\n\n' +
            'Commands:\n' +
            '/start - Start the bot\n' +
            '/status - Get current bot status\n' +
            '/performance - View performance metrics\n' +
            '/settings - View and adjust settings\n' +
            '/help - Show this help message\n\n' +
            'You can also send natural language messages to:\n' +
            '- Check specific trading pairs\n' +
            '- Request market analysis\n' +
            '- Adjust bot parameters\n' +
            '- Get detailed performance reports';
        
        await this.sendMessage(chatId, helpText);
    }

    startMessageProcessing() {
        setInterval(() => {
            this.processMessageQueue();
        }, 100);
    }

    async processMessageQueue() {
        if (this.isProcessing || this.messageQueue.length === 0) {
            return;
        }
        
        this.isProcessing = true;
        
        try {
            const message = this.messageQueue.shift();
            await this.bot.sendMessage(message.chatId, message.text, message.options);
        } catch (error) {
            console.error('Error sending Telegram message:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    async sendMessage(chatId, text, options = {}) {
        this.messageQueue.push({ chatId, text, options });
    }

    async broadcastMessage(message) {
        for (const chatId of this.chatIds) {
            await this.sendMessage(chatId, message);
        }
    }

    async notifyTrade(trade) {
        const message = this.formatTradeNotification(trade);
        await this.broadcastMessage(message);
    }

    formatTradeNotification(trade) {
        return `🔄 Trade Executed\n\n` +
               `Pair: ${trade.pair}\n` +
               `Type: ${trade.type}\n` +
               `Profit: ${this.formatCurrency(trade.profit)}\n` +
               `ROI: ${trade.roi}%\n` +
               `Gas Used: ${trade.gasUsed} Gwei\n` +
               `Time: ${this.formatDate(trade.timestamp)}`;
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }

    formatDate(timestamp) {
        return new Date(timestamp).toLocaleString();
    }

    formatTimeAgo(timestamp) {
        const seconds = Math.floor((Date.now() - timestamp) / 1000);
        
        if (seconds < 60) return `${seconds}s ago`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        return `${Math.floor(seconds / 86400)}d ago`;
    }

    formatDuration(ms) {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);
        
        if (days > 0) return `${days}d ${hours % 24}h`;
        if (hours > 0) return `${hours}h ${minutes % 60}m`;
        if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
        return `${seconds}s`;
    }
}

// Export the TelegramBot class
module.exports = TelegramBot; 