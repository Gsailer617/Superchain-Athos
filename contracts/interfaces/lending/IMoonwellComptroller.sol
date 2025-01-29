// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

interface IMoonwellComptroller {
    function enterMarkets(address[] calldata mTokens) external returns (uint[] memory);
    function exitMarket(address mToken) external returns (uint);
} 