// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface ILayerZeroEndpoint {
    // Messaging functions
    function send(
        uint32 _dstChainId,
        bytes calldata _destination,
        bytes calldata _payload,
        address payable _refundAddress,
        address _zroPaymentAddress,
        bytes calldata _adapterParams
    ) external payable;

    function estimateFees(
        uint32 _dstChainId,
        address _userApplication,
        bytes calldata _payload,
        bool _payInZRO,
        bytes calldata _adapterParams
    ) external view returns (uint256 nativeFee, uint256 zroFee);

    // View functions
    function getChainId() external view returns (uint32);
    function getInboundNonce(uint32 _srcChainId, bytes calldata _srcAddress) external view returns (uint64);
    function getOutboundNonce(uint32 _dstChainId, address _srcAddress) external view returns (uint64);
    function getConfig(
        uint32 _version,
        uint32 _chainId,
        address _userApplication,
        uint256 _configType
    ) external view returns (bytes memory);

    // Events
    event PayloadReceived(
        uint32 indexed srcChainId,
        bytes indexed srcAddress,
        address indexed dstAddress,
        uint64 nonce,
        bytes payload
    );
    event PayloadStored(
        uint32 indexed srcChainId,
        bytes indexed srcAddress,
        address indexed dstAddress,
        uint64 nonce,
        bytes payload,
        bytes reason
    );
} 