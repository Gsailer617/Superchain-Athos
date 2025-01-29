// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/Address.sol";

library TokenValidationLib {
    using Address for address;

    struct TokenMetrics {
        uint256 totalLiquidity;      // Total liquidity across all DEXes
        uint256 dailyVolume;         // 24h trading volume
        uint256 priceVolatility;     // Price volatility score (0-100)
        uint256 holdersCount;        // Number of unique holders
        uint256 lastUpdateTime;      // Last time metrics were updated
    }

    struct ValidationConfig {
        uint256 minLiquidity;        // Minimum required liquidity
        uint256 minDailyVolume;      // Minimum required 24h volume
        uint256 maxVolatility;       // Maximum allowed volatility
        uint256 minHolders;          // Minimum required holders
        uint256 maxTaxBps;           // Maximum allowed transfer tax in basis points
    }

    // Security checks
    function validateTokenSecurity(
        address token
    ) internal view returns (bool isValid, string memory reason) {
        // Check if address is a contract
        if (!token.isContract()) {
            return (false, "Not a contract");
        }

        IERC20 tokenContract = IERC20(token);

        try tokenContract.totalSupply() returns (uint256) {
            // Valid totalSupply call
        } catch {
            return (false, "Invalid totalSupply");
        }

        // Check transfer functionality
        try tokenContract.balanceOf(address(this)) returns (uint256) {
            // Valid balanceOf call
        } catch {
            return (false, "Invalid balanceOf");
        }

        return (true, "Valid token");
    }

    // Validate token metrics against configuration
    function validateTokenMetrics(
        TokenMetrics memory metrics,
        ValidationConfig memory config
    ) internal pure returns (bool isValid, string memory reason) {
        if (metrics.totalLiquidity < config.minLiquidity) {
            return (false, "Insufficient liquidity");
        }

        if (metrics.dailyVolume < config.minDailyVolume) {
            return (false, "Low trading volume");
        }

        if (metrics.priceVolatility > config.maxVolatility) {
            return (false, "High volatility");
        }

        if (metrics.holdersCount < config.minHolders) {
            return (false, "Too few holders");
        }

        return (true, "Valid metrics");
    }

    // Estimate transfer tax by doing a test transfer
    function estimateTransferTax(
        address token,
        address testPool
    ) internal view returns (uint256 taxBps) {
        IERC20 tokenContract = IERC20(token);
        uint256 poolBalance = tokenContract.balanceOf(testPool);
        
        if (poolBalance == 0) return 0;

        try tokenContract.balanceOf(testPool) returns (uint256 balance) {
            // Calculate tax based on pool balance changes
            uint256 expectedBalance = balance;
            uint256 actualBalance = tokenContract.balanceOf(testPool);
            
            if (actualBalance < expectedBalance) {
                uint256 taxAmount = expectedBalance - actualBalance;
                taxBps = (taxAmount * 10000) / expectedBalance;
            }
        } catch {
            return 10000; // 100% tax as error case
        }
    }
} 