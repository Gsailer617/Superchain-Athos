// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

import "@openzeppelin/contracts/access/Ownable.sol";

contract ArbitrageStorage is Ownable {
    // Enums
    enum FlashLoanProvider { MOONWELL, MORPHO, AAVE, BALANCER }
    enum DEX { BASESWAP, AERODROME, SUSHISWAP, PANCAKESWAP, UNISWAP_V3 }

    // Core storage
    mapping(address => bool) public isTokenSupported;
    address[] public supportedTokens;
    mapping(FlashLoanProvider => uint256) public providerFees;

    // Trade history
    struct TradeHistory {
        address tokenIn;
        address tokenOut;
        uint256 amountIn;
        uint256 amountOut;
        uint256 profit;
        uint256 timestamp;
        string sourceDex;
        string targetDex;
    }
    TradeHistory[] public tradeHistory;

    // Events
    event TokenAdded(address token);
    event TokenRemoved(address token);
    event TradeRecorded(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 amountOut,
        uint256 profit,
        uint256 timestamp
    );

    // Storage management functions
    function addSupportedToken(address token) external onlyOwner {
        require(token != address(0), "Invalid token");
        require(!isTokenSupported[token], "Token already supported");
        
        isTokenSupported[token] = true;
        supportedTokens.push(token);
        emit TokenAdded(token);
    }

    function removeSupportedToken(address token) external onlyOwner {
        require(isTokenSupported[token], "Token not supported");
        
        isTokenSupported[token] = false;
        for (uint i = 0; i < supportedTokens.length; i++) {
            if (supportedTokens[i] == token) {
                supportedTokens[i] = supportedTokens[supportedTokens.length - 1];
                supportedTokens.pop();
                break;
            }
        }
        emit TokenRemoved(token);
    }

    function recordTrade(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 amountOut,
        uint256 profit,
        string memory sourceDex,
        string memory targetDex
    ) internal {
        tradeHistory.push(TradeHistory({
            tokenIn: tokenIn,
            tokenOut: tokenOut,
            amountIn: amountIn,
            amountOut: amountOut,
            profit: profit,
            timestamp: block.timestamp,
            sourceDex: sourceDex,
            targetDex: targetDex
        }));

        emit TradeRecorded(
            tokenIn,
            tokenOut,
            amountIn,
            amountOut,
            profit,
            block.timestamp
        );
    }
} 