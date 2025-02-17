// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface ILayerZeroReceiver {
    // LayerZero endpoint will invoke this function to deliver the message on the destination
    function lzReceive(
        uint32 _srcChainId,
        bytes calldata _srcAddress,
        uint64 _nonce,
        bytes calldata _payload
    ) external;

    // Called to validate an address for the UA
    function validateReceiver(
        uint32 _srcChainId,
        bytes calldata _srcAddress
    ) external view returns (bool);
} 