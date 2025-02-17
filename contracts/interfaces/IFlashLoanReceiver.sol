// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IFlashLoanReceiver {
    /**
     * @notice Execute operation after receiving flash loaned amounts
     * @param assets The addresses of the assets being flash-borrowed
     * @param amounts The amounts of the assets being flash-borrowed
     * @param premiums The premiums (fees) for each borrowed asset
     * @param initiator The address initiating the flash loan
     * @param params Arbitrary bytes encoded params passed through flash loan
     * @return success Whether the operation was successful
     */
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external returns (bool success);

    /**
     * @notice Balancer flash loan callback
     * @param tokens The addresses of the tokens being flash-borrowed
     * @param amounts The amounts of the tokens being flash-borrowed
     * @param feeAmounts The fee amounts to be paid for each token
     * @param userData Arbitrary user-encoded data
     */
    function receiveFlashLoan(
        address[] memory tokens,
        uint256[] memory amounts,
        uint256[] memory feeAmounts,
        bytes memory userData
    ) external;
} 