export interface DEXConfig {
    name: string;
    factory: `0x${string}`;
    router: `0x${string}`;
    fees: number[];
    initCodeHash: `0x${string}`;
}

export const DEX_CONFIGS: Record<string, DEXConfig> = {
    UNISWAP_V2: {
        name: 'Uniswap V2',
        factory: '0x8909Dc15e40173Ff4699343b6eB8132c65e18eC6' as `0x${string}`, // Base Mainnet
        router: '0x2626664c2603336E57B271c5C0b26F421741e481' as `0x${string}`,
        fees: [0.003], // 0.3%
        initCodeHash: '0x96e8ac4277198ff8b6f785478aa9a39f403cb768dd02cbee326c3e7da348845f' as `0x${string}`
    },
    SUSHISWAP: {
        name: 'SushiSwap',
        factory: '0x71524B4f93c58fcbF659783284E38825f0622859' as `0x${string}`, // Base Mainnet
        router: '0x6BDED42c6DA8FD5b7Df1D8bB777077644592AA90' as `0x${string}`,
        fees: [0.003], // 0.3%
        initCodeHash: '0xe18a34eb0e04b04f7a0ac29a6e80748dca96319b42c54d679cb821dca90c6303' as `0x${string}`
    },
    AERODROME: {
        name: 'Aerodrome',
        factory: '0x420DD381b31aEf6683db6B902084cB0FFEe076318' as `0x${string}`, // Base Mainnet
        router: '0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43' as `0x${string}`,
        fees: [0.0017, 0.003], // 0.17% stable, 0.3% volatile
        initCodeHash: '0x8b52a29943e60d0da73d4ef4825b35e0260ce7617d8c58642dfc6749d83b0b47' as `0x${string}`
    },
    BASESWAP: {
        name: 'BaseSwap',
        factory: '0xFDa619b6d20975be80A10332cD39b9a4b0FAa8BB' as `0x${string}`, // Base Mainnet
        router: '0x327Df1E6de05895d2ab08513aaDD9313Fe505d86' as `0x${string}`,
        fees: [0.002, 0.003], // 0.2% stable, 0.3% volatile
        initCodeHash: '0x84845e7ccb283dec564acfcd3d9287a491dec6d675705545a2ab8be22ad78f31' as `0x${string}`
    }
}; 