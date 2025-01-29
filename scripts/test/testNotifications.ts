import { RewardTracker } from '../utils/RewardTracker';

async function testNotifications() {
    try {
        console.log('Starting notification test...');
        const tracker = new RewardTracker();
        
        // Test basic notification
        console.log('Sending test notification...');
        await tracker.testNotification();
        
        // Wait a bit and test a trade notification
        setTimeout(async () => {
            console.log('Sending trade notification...');
            await tracker.notifySuccessfulTrade(
                'DackieSwap',
                1500,  // $1500 trade (above new $1000 threshold)
                75,    // $75 profit (5% profit)
                { from: 'ETH', to: 'USDC' }
            );
        }, 2000);
        
        console.log('Test completed! Check your Telegram for messages.');
    } catch (error) {
        console.error('Test failed:', error);
    }
}

// Run the test
testNotifications(); 