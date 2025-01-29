// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

interface IMorphoFlashLoanCallback {
    function onMorphoFlashLoan(
        address caller,
        address token,
        uint256 amount,
        bytes calldata data
    ) external returns (bytes32);
}

interface IMorpho {
    function flashLoan(
        address token,
        uint256 amount,
        bytes calldata data
    ) external returns (bytes32);
} 