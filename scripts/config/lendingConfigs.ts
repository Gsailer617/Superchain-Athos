import { FlashLoanProvider, LendingProtocolConfig } from '../types/dex';

export const LENDING_CONFIGS: Record<string, LendingProtocolConfig> = {
    MOONWELL: {
        name: 'Moonwell',
        address: '0x6DB96BBEB081d2a85E0954C252f2c1dC108b3f81' as `0x${string}`, // Base Mainnet
        provider: FlashLoanProvider.MOONWELL,
        supportedTokens: [
            '0x4200000000000000000000000000000000000006' as `0x${string}`, // WETH
            '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913' as `0x${string}`, // USDC
            '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb' as `0x${string}`, // DAI
        ],
        flashLoanFee: 0.0009, // 0.09%
        maxLTV: 0.8 // 80%
    },
    BASESWAP: {
        name: 'BaseSwap',
        address: '0x327Df1E6de05895d2ab08513aaDD9313Fe505d86' as `0x${string}`, // Base Mainnet
        provider: FlashLoanProvider.BASESWAP,
        supportedTokens: [
            '0x4200000000000000000000000000000000000006' as `0x${string}`, // WETH
            '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913' as `0x${string}`, // USDC
            '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb' as `0x${string}`, // DAI
        ],
        flashLoanFee: 0.001, // 0.1%
        maxLTV: 0.75 // 75%
    },
    MORPHO: {
        name: 'Morpho',
        address: '0x64c7044050Ba0431252df24fEd4d9635a275CB41' as `0x${string}`, // Base Mainnet
        provider: FlashLoanProvider.MORPHO,
        supportedTokens: [
            '0x4200000000000000000000000000000000000006' as `0x${string}`, // WETH
            '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913' as `0x${string}`, // USDC
            '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb' as `0x${string}`, // DAI
            '0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22' as `0x${string}`, // cbETH
        ],
        flashLoanFee: 0.0008, // 0.08%
        maxLTV: 0.82 // 82%
    }
}; 