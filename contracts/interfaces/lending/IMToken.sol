// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

interface IMToken {
    function mint(uint mintAmount) external returns (uint);
    function flashLoan(
        address receiver,
        uint256 amount,
        bytes calldata params
    ) external returns (bool);
    function underlying() external view returns (address);
} 