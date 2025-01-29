// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

library SafeTransferLib {
    function safeTransferETH(address to, uint256 amount) internal {
        bool success;

        assembly {
            // Transfer the ETH and store if it succeeded or not.
            success := call(gas(), to, amount, 0, 0, 0, 0)
        }

        require(success, "ETH transfer failed");
    }
} 