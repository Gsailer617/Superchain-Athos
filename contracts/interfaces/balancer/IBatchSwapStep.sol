// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

interface IBatchSwapStep {
    struct BatchSwapStep {
        bytes32 poolId;
        uint256 assetInIndex;
        uint256 assetOutIndex;
        uint256 amount;
        bytes userData;
    }
} 