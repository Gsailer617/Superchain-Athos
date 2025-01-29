import { BigNumberish } from 'ethers';

interface RewardSnapshot {
    timestamp: number;
    amount: bigint;
    rewardToken: string;
    protocol: string;
    type: RewardType;
    apy: number;
}

interface ProtocolMetrics {
    totalRewards: { [token: string]: bigint };
    historicalAPY: number[];
    volumeContribution: number;
    lastUpdate: number;
    lockPeriod?: number;
    governanceWeight?: number;
}

enum RewardType {
    TRADING_FEE = 'TRADING_FEE',
    LIQUIDITY_MINING = 'LIQUIDITY_MINING',
    GOVERNANCE = 'GOVERNANCE',
    REFERRAL = 'REFERRAL',
    STAKING = 'STAKING',
    LENDING = 'LENDING'
}

interface GovernanceInfo {
    token: string;
    votingPower: bigint;
    proposalThreshold: bigint;
    delegatedPower: bigint;
    activeProposals: number;
    votingHistory: {
        proposalId: string;
        support: boolean;
        weight: bigint;
        timestamp: number;
    }[];
}

// Add new interfaces for portfolio and notifications
interface PortfolioAction {
    type: 'STAKE' | 'UNSTAKE' | 'CLAIM' | 'VOTE' | 'DELEGATE';
    protocol: string;
    amount?: bigint;
    token?: string;
    proposalId?: string;
    reason?: string;
    estimatedReward?: bigint;
    deadline?: number;
}

interface NotificationConfig {
    telegramBotToken: string;
    chatId: string;
    minTradeValueUSD: number;    // Minimum trade value to notify
    minStakingAPY: number;       // Minimum APY to notify staking opportunity
    minGovernanceValue: number;  // Minimum voting power value to notify governance
}

interface AutomationConfig {
    enabledActions: Set<PortfolioAction['type']>;
    minRewardThresholdUSD: number;
    maxStakePerProtocol: number;
    autoCompound: boolean;
    governanceAutoVote: boolean;
}

export class RewardTracker {
    private protocolRewards: Map<string, ProtocolMetrics> = new Map();
    private rewardHistory: RewardSnapshot[] = [];
    private governanceData: Map<string, GovernanceInfo> = new Map();
    
    private readonly HISTORY_LIMIT = 1000;
    private readonly APY_WINDOW = 30 * 24 * 60 * 60 * 1000; // 30 days
    private readonly MIN_GOVERNANCE_PARTICIPATION = 5; // Minimum proposals to calculate engagement
    
    private notificationConfig: NotificationConfig | null = null;
    private automationConfig: AutomationConfig | null = null;
    private pendingActions: PortfolioAction[] = [];
    private lastNotificationTime: { [key: string]: number } = {};
    
    private readonly NOTIFICATION_COOLDOWN = 3600000; // 1 hour
    private readonly IMPORTANT_APY_CHANGE = 20; // 20% change
    private readonly MIN_REWARD_NOTIFICATION = 100; // $100 USD
    
    private readonly NOTIFICATION_PRIORITIES = {
        HIGH_APY_THRESHOLD: 15,           // 15% APY
        SIGNIFICANT_TRADE_CHANGE: 1000,   // $1000 USD
        GOVERNANCE_POWER_THRESHOLD: 100   // $100 worth of voting power
    };
    
    constructor() {}

    addReward(
        protocol: string,
        amount: bigint,
        rewardToken: string,
        type: RewardType,
        apy: number
    ): void {
        // Add to history
        this.rewardHistory.push({
            timestamp: Date.now(),
            amount,
            rewardToken,
            protocol,
            type,
            apy
        });

        // Trim history if needed
        if (this.rewardHistory.length > this.HISTORY_LIMIT) {
            this.rewardHistory = this.rewardHistory.slice(-this.HISTORY_LIMIT);
        }

        // Update protocol metrics
        const metrics = this.protocolRewards.get(protocol) || this.getDefaultMetrics();
        if (!metrics.totalRewards[rewardToken]) {
            metrics.totalRewards[rewardToken] = 0n;
        }
        metrics.totalRewards[rewardToken] += amount;
        metrics.historicalAPY.push(apy);
        metrics.lastUpdate = Date.now();
        
        this.protocolRewards.set(protocol, metrics);
    }

    updateGovernanceData(
        protocol: string,
        votingPower: bigint,
        proposalThreshold: bigint,
        delegatedPower: bigint,
        activeProposals: number
    ): void {
        const govInfo = this.governanceData.get(protocol) || {
            token: '',
            votingPower: 0n,
            proposalThreshold: 0n,
            delegatedPower: 0n,
            activeProposals: 0,
            votingHistory: []
        };

        govInfo.votingPower = votingPower;
        govInfo.proposalThreshold = proposalThreshold;
        govInfo.delegatedPower = delegatedPower;
        govInfo.activeProposals = activeProposals;

        this.governanceData.set(protocol, govInfo);
    }

    addVotingRecord(
        protocol: string,
        proposalId: string,
        support: boolean,
        weight: bigint
    ): void {
        const govInfo = this.governanceData.get(protocol);
        if (!govInfo) return;

        govInfo.votingHistory.push({
            proposalId,
            support,
            weight,
            timestamp: Date.now()
        });

        this.governanceData.set(protocol, govInfo);
    }

    getProtocolMetrics(protocol: string): ProtocolMetrics | null {
        return this.protocolRewards.get(protocol) || null;
    }

    getRewardHistory(
        protocol?: string,
        type?: RewardType,
        startTime?: number
    ): RewardSnapshot[] {
        return this.rewardHistory.filter(reward => 
            (!protocol || reward.protocol === protocol) &&
            (!type || reward.type === type) &&
            (!startTime || reward.timestamp >= startTime)
        );
    }

    calculateProtocolAPY(protocol: string): number {
        const metrics = this.protocolRewards.get(protocol);
        if (!metrics || metrics.historicalAPY.length === 0) return 0;

        const recentAPY = metrics.historicalAPY.slice(-30);
        return recentAPY.reduce((sum, apy) => sum + apy, 0) / recentAPY.length;
    }

    getGovernanceEngagement(protocol: string): {
        participationRate: number;
        proposalCreationEligible: boolean;
        votingPower: bigint;
        delegatedPower: bigint;
    } {
        const govInfo = this.governanceData.get(protocol);
        if (!govInfo) return {
            participationRate: 0,
            proposalCreationEligible: false,
            votingPower: 0n,
            delegatedPower: 0n
        };

        const recentVotes = govInfo.votingHistory.filter(
            vote => Date.now() - vote.timestamp < this.APY_WINDOW
        );

        return {
            participationRate: recentVotes.length >= this.MIN_GOVERNANCE_PARTICIPATION
                ? recentVotes.length / Math.max(1, govInfo.activeProposals)
                : 0,
            proposalCreationEligible: govInfo.votingPower >= govInfo.proposalThreshold,
            votingPower: govInfo.votingPower,
            delegatedPower: govInfo.delegatedPower
        };
    }

    analyzeRewardTrends(protocol: string): {
        rewardGrowth: number;
        averageAPY: number;
        bestRewardType: RewardType;
        governanceQualified: boolean;
        pendingActions: PortfolioAction[];
    } {
        const metrics = this.protocolRewards.get(protocol);
        const govInfo = this.governanceData.get(protocol);
        if (!metrics) return {
            rewardGrowth: 0,
            averageAPY: 0,
            bestRewardType: RewardType.TRADING_FEE,
            governanceQualified: false,
            pendingActions: []
        };

        // Calculate reward growth
        const recentRewards = this.rewardHistory.filter(
            r => r.protocol === protocol && 
            Date.now() - r.timestamp < this.APY_WINDOW
        );

        const rewardsByType = new Map<RewardType, {total: bigint, count: number}>();
        recentRewards.forEach(reward => {
            const current = rewardsByType.get(reward.type) || {total: 0n, count: 0};
            rewardsByType.set(reward.type, {
                total: current.total + reward.amount,
                count: current.count + 1
            });
        });

        let bestType = RewardType.TRADING_FEE;
        let maxAverage = 0n;
        rewardsByType.forEach((value, type) => {
            const average = value.total / BigInt(Math.max(1, value.count));
            if (average > maxAverage) {
                maxAverage = average;
                bestType = type;
            }
        });

        const baseAnalysis = {
            rewardGrowth: this.calculateRewardGrowth(recentRewards),
            averageAPY: this.calculateProtocolAPY(protocol),
            bestRewardType: bestType,
            governanceQualified: govInfo ? govInfo.votingPower >= govInfo.proposalThreshold : false,
            pendingActions: this.pendingActions.filter(a => a.protocol === protocol)
        };

        return baseAnalysis;
    }

    private calculateRewardGrowth(rewards: RewardSnapshot[]): number {
        if (rewards.length < 2) return 0;

        const oldestAmount = rewards[0].amount;
        const newestAmount = rewards[rewards.length - 1].amount;
        
        return Number((newestAmount - oldestAmount) * 100n / oldestAmount);
    }

    private getDefaultMetrics(): ProtocolMetrics {
        return {
            totalRewards: {},
            historicalAPY: [],
            volumeContribution: 0,
            lastUpdate: Date.now()
        };
    }

    configureNotifications(config: NotificationConfig): void {
        this.notificationConfig = config;
    }

    configureAutomation(config: AutomationConfig): void {
        this.automationConfig = config;
    }

    async checkAndExecuteActions(): Promise<void> {
        if (!this.automationConfig) return;

        // Check for reward claims
        this.checkRewardClaims();
        
        // Check for staking opportunities
        this.checkStakingOpportunities();
        
        // Check governance participation
        this.checkGovernanceActions();
        
        // Execute pending actions
        await this.executePendingActions();
    }

    private checkRewardClaims(): void {
        if (!this.automationConfig?.enabledActions.has('CLAIM')) return;

        this.protocolRewards.forEach((metrics, protocol) => {
            Object.entries(metrics.totalRewards).forEach(([token, amount]) => {
                const usdValue = this.estimateUSDValue(amount, token);
                if (usdValue >= this.automationConfig!.minRewardThresholdUSD) {
                    this.addPendingAction({
                        type: 'CLAIM',
                        protocol,
                        token,
                        amount,
                        estimatedReward: amount
                    });
                }
            });
        });
    }

    private checkStakingOpportunities(): void {
        this.protocolRewards.forEach((metrics, protocol) => {
            const apy = this.calculateProtocolAPY(protocol);
            if (apy > this.NOTIFICATION_PRIORITIES.HIGH_APY_THRESHOLD) {
                const estimatedAnnualRewardUSD = this.estimateAnnualRewardUSD(protocol, apy);
                this.notifyStakingOpportunity(protocol, apy, estimatedAnnualRewardUSD);
            }
        });
    }

    private checkGovernanceActions(): void {
        this.governanceData.forEach((govInfo, protocol) => {
            if (govInfo.activeProposals > 0 && govInfo.votingPower > 0n) {
                const votingPowerUSD = this.estimateUSDValue(govInfo.votingPower, govInfo.token);
                if (votingPowerUSD >= this.NOTIFICATION_PRIORITIES.GOVERNANCE_POWER_THRESHOLD) {
                    this.notifyGovernanceParticipation(protocol, 'latest', votingPowerUSD);
                }
            }
        });
    }

    private async executePendingActions(): Promise<void> {
        for (const action of this.pendingActions) {
            try {
                await this.executeAction(action);
                await this.sendNotification(
                    `✅ Executed ${action.type} for ${action.protocol}\n` +
                    `${action.reason ? `Reason: ${action.reason}\n` : ''}` +
                    `${action.amount ? `Amount: ${action.amount}\n` : ''}`
                );
            } catch (error: unknown) {
                const errorMessage = error instanceof Error ? error.message : 'Unknown error';
                await this.sendNotification(
                    `❌ Failed to execute ${action.type} for ${action.protocol}\n` +
                    `Error: ${errorMessage}`
                );
            }
        }
        this.pendingActions = [];
    }

    private async executeAction(action: PortfolioAction): Promise<void> {
        // This would integrate with your actual transaction execution logic
        switch (action.type) {
            case 'CLAIM':
                // Execute claim transaction
                break;
            case 'STAKE':
                // Execute stake transaction
                break;
            case 'VOTE':
                // Execute vote transaction
                break;
            // ... handle other action types
        }
    }

    private addPendingAction(action: PortfolioAction): void {
        const existingAction = this.pendingActions.find(
            a => a.type === action.type && a.protocol === action.protocol
        );
        if (!existingAction) {
            this.pendingActions.push(action);
        }
    }

    private async sendNotification(message: string): Promise<void> {
        if (!this.notificationConfig) return;

        const now = Date.now();
        const messageHash = message.slice(0, 50); // Use first 50 chars as identifier

        // Check cooldown
        if (this.lastNotificationTime[messageHash] && 
            now - this.lastNotificationTime[messageHash] < this.NOTIFICATION_COOLDOWN) {
            return;
        }

        try {
            const url = `https://api.telegram.org/bot${this.notificationConfig.telegramBotToken}/sendMessage`;
            const body = {
                chat_id: this.notificationConfig.chatId,
                text: message,
                parse_mode: 'HTML',
                disable_notification: false  // Important notifications should make sound
            };

            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });

            if (!response.ok) {
                throw new Error(`Telegram API error: ${response.status} ${response.statusText}`);
            }

            this.lastNotificationTime[messageHash] = now;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
            console.error('Failed to send notification:', errorMessage);
            throw new Error(`Notification failed: ${errorMessage}`);
        }
    }

    private calculateOptimalStake(protocol: string): bigint {
        const metrics = this.protocolRewards.get(protocol);
        if (!metrics) return 0n;

        const apy = this.calculateProtocolAPY(protocol);
        const maxStake = BigInt(Math.floor(this.automationConfig!.maxStakePerProtocol * 100)) * 
            (10n ** 18n) / 100n;

        // Simple calculation - could be made more sophisticated
        return apy > 50 ? maxStake : maxStake / 2n;
    }

    private estimateUSDValue(amount: bigint, token: string): number {
        // This would integrate with your price feed system
        return 0; // Placeholder
    }

    async notifyStakingOpportunity(protocol: string, apy: number, estimatedRewardUSD: number): Promise<void> {
        if (!this.notificationConfig || apy < this.notificationConfig.minStakingAPY) return;

        const message = `🔥 High APY Opportunity!\n\n` +
            `Protocol: ${protocol}\n` +
            `APY: ${apy.toFixed(2)}%\n` +
            `Estimated Annual Reward: $${estimatedRewardUSD.toFixed(2)}\n\n` +
            `Consider staking in this protocol for good returns.`;

        await this.sendNotification(message);
    }

    async notifyGovernanceParticipation(
        protocol: string, 
        proposalId: string, 
        votingPowerUSD: number,
        deadline?: number
    ): Promise<void> {
        if (!this.notificationConfig || 
            votingPowerUSD < this.notificationConfig.minGovernanceValue) return;

        const deadlineStr = deadline 
            ? `\nDeadline: ${new Date(deadline).toLocaleString()}`
            : '';

        const message = `🏛 Governance Participation Opportunity!\n\n` +
            `Protocol: ${protocol}\n` +
            `Proposal: ${proposalId}\n` +
            `Your Voting Power: $${votingPowerUSD.toFixed(2)}${deadlineStr}\n\n` +
            `Your vote matters in shaping the protocol's future!`;

        await this.sendNotification(message);
    }

    async notifySuccessfulTrade(
        protocol: string,
        tradeValueUSD: number,
        profitUSD: number,
        tokens: { from: string, to: string }
    ): Promise<void> {
        if (!this.notificationConfig || 
            tradeValueUSD < this.notificationConfig.minTradeValueUSD) return;

        const profitPercentage = (profitUSD / tradeValueUSD) * 100;
        const message = `💰 Successful Trade!\n\n` +
            `Protocol: ${protocol}\n` +
            `Trade Value: $${tradeValueUSD.toFixed(2)}\n` +
            `Profit: $${profitUSD.toFixed(2)} (${profitPercentage.toFixed(2)}%)\n` +
            `Tokens: ${tokens.from} ➜ ${tokens.to}`;

        await this.sendNotification(message);
    }

    private estimateAnnualRewardUSD(protocol: string, apy: number): number {
        const metrics = this.protocolRewards.get(protocol);
        if (!metrics) return 0;

        // Calculate based on current rewards and APY
        const totalRewardValue = Object.entries(metrics.totalRewards)
            .reduce((sum, [token, amount]) => 
                sum + this.estimateUSDValue(amount, token), 0);

        return totalRewardValue * (apy / 100);
    }

    // Example configuration setup
    private setupDefaultConfig(): void {
        this.configureNotifications({
            telegramBotToken: '7787762959:AAHW2pZI2WbUuSx6qksPgWshZVFGD-nRHHA',  // Your bot token
            chatId: '7683094993',  // Your chat ID
            minTradeValueUSD: 1000,    // Notify for trades over $1000
            minStakingAPY: 15,        // Notify for APY over 15%
            minGovernanceValue: 100   // Notify for voting power over $100
        });
    }

    // Test method to verify notifications
    public async testNotification(): Promise<void> {
        if (!this.notificationConfig) {
            this.setupDefaultConfig();
        }
        await this.sendNotification('🔔 Test notification from your arbitrage bot!\n\nIf you see this message, notifications are working correctly.\n\nYou will receive notifications for:\n- Trades over $1000\n- APY opportunities over 15%\n- Governance events over $100');
    }

    async notifyBoostAdjustment(
        pair: string,
        newBoost: number,
        reason: string,
        metrics: {
            successRate: number;
            averageProfit: number;
            volatilityScore: number;
            liquidityScore: number;
            gasEfficiency: number;
        }
    ): Promise<void> {
        if (!this.notificationConfig) return;

        const message = `🔄 Boost Adjustment for ${pair}\n\n` +
            `New Boost Multiplier: ${newBoost.toFixed(2)}x\n` +
            `Reason: ${reason}\n\n` +
            `Metrics:\n` +
            `- Success Rate: ${(metrics.successRate * 100).toFixed(1)}%\n` +
            `- Avg Profit: $${metrics.averageProfit.toFixed(2)}\n` +
            `- Volatility: ${(metrics.volatilityScore * 100).toFixed(1)}%\n` +
            `- Liquidity Score: ${(metrics.liquidityScore * 100).toFixed(1)}%\n` +
            `- Gas Efficiency: ${metrics.gasEfficiency.toFixed(1)}x`;

        await this.sendNotification(message);
    }
} 