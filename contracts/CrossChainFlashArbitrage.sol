// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./interfaces/IFlashLoanReceiver.sol";
import "./interfaces/IFlashSwapCallback.sol";
import "./interfaces/ILayerZeroEndpoint.sol";
import "./interfaces/ILayerZeroReceiver.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title CrossChainFlashArbitrage
 * @notice Contract for executing flash loan arbitrage and cross-chain operations using LayerZero
 */
contract CrossChainFlashArbitrage is 
    IFlashLoanReceiver,
    IFlashSwapCallback,
    ILayerZeroReceiver,
    Ownable,
    ReentrancyGuard 
{
    using SafeERC20 for IERC20;

    // Events
    event FlashLoanExecuted(
        address[] tokens,
        uint256[] amounts,
        uint256[] premiums,
        address initiator
    );

    event FlashSwapExecuted(
        address token0,
        address token1,
        uint256 amount0,
        uint256 amount1,
        uint256 fee
    );

    event ArbitrageProfitRealized(
        address token,
        uint256 amount,
        address beneficiary
    );

    event CrossChainMessageSent(
        uint32 dstChainId,
        bytes destination,
        bytes payload,
        uint256 fee
    );

    event CrossChainMessageReceived(
        uint32 srcChainId,
        bytes srcAddress,
        bytes payload
    );

    // State variables
    mapping(address => bool) public authorizedCallers;
    mapping(string => address) public chainBridges;
    mapping(bytes32 => bool) public pendingBridgeTransfers;
    mapping(uint32 => bytes) public trustedRemotes;
    
    // LayerZero endpoint
    ILayerZeroEndpoint public immutable lzEndpoint;
    
    // Profit tracking
    mapping(address => uint256) public realizedProfits;

    constructor(address _endpoint) {
        require(_endpoint != address(0), "Invalid endpoint address");
        lzEndpoint = ILayerZeroEndpoint(_endpoint);
        authorizedCallers[msg.sender] = true;
    }

    // LayerZero receiver function
    function lzReceive(
        uint32 _srcChainId,
        bytes calldata _srcAddress,
        uint64 _nonce,
        bytes calldata _payload
    ) external override {
        require(msg.sender == address(lzEndpoint), "Invalid endpoint caller");
        require(trustedRemotes[_srcChainId].length > 0, "Source chain not trusted");
        require(
            keccak256(_srcAddress) == keccak256(trustedRemotes[_srcChainId]),
            "Invalid source address"
        );

        // Decode and handle the payload
        (
            address token,
            uint256 amount,
            address beneficiary,
            bytes memory executionData
        ) = abi.decode(_payload, (address, uint256, address, bytes));

        // Execute the cross-chain operation
        _handleCrossChainOperation(
            _srcChainId,
            token,
            amount,
            beneficiary,
            executionData
        );

        emit CrossChainMessageReceived(_srcChainId, _srcAddress, _payload);
    }

    // Validate receiver
    function validateReceiver(
        uint32 _srcChainId,
        bytes calldata _srcAddress
    ) external view override returns (bool) {
        return keccak256(_srcAddress) == keccak256(trustedRemotes[_srcChainId]);
    }

    // Send cross-chain message
    function sendCrossChainMessage(
        uint32 _dstChainId,
        bytes calldata _destination,
        address _token,
        uint256 _amount,
        address _beneficiary,
        bytes calldata _executionData
    ) external payable onlyAuthorized nonReentrant {
        require(trustedRemotes[_dstChainId].length > 0, "Destination chain not trusted");

        // Encode the payload
        bytes memory payload = abi.encode(
            _token,
            _amount,
            _beneficiary,
            _executionData
        );

        // Get the fee for sending message
        (uint256 nativeFee, ) = lzEndpoint.estimateFees(
            _dstChainId,
            address(this),
            payload,
            false,
            bytes("")
        );

        require(msg.value >= nativeFee, "Insufficient native token for fees");

        // Send the message
        lzEndpoint.send{value: nativeFee}(
            _dstChainId,
            _destination,
            payload,
            payable(msg.sender),
            address(0),
            bytes("")
        );

        emit CrossChainMessageSent(_dstChainId, _destination, payload, nativeFee);
    }

    // Internal function to handle received cross-chain messages
    function _handleCrossChainOperation(
        uint32 _srcChainId,
        address _token,
        uint256 _amount,
        address _beneficiary,
        bytes memory _executionData
    ) internal {
        // Execute the cross-chain operation based on execution data
        // This could be a swap, liquidity provision, etc.
        (bool success, ) = address(this).call(_executionData);
        require(success, "Cross-chain execution failed");

        // Transfer tokens to beneficiary if needed
        if (_amount > 0 && _beneficiary != address(0)) {
            IERC20(_token).safeTransfer(_beneficiary, _amount);
        }
    }

    // Admin function to set trusted remote
    function setTrustedRemote(
        uint32 _chainId,
        bytes calldata _path
    ) external onlyOwner {
        trustedRemotes[_chainId] = _path;
    }

    // Modifiers
    modifier onlyAuthorized() {
        require(authorizedCallers[msg.sender], "Caller not authorized");
        _;
    }

    // Flash loan callback
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        // Execute arbitrage logic
        for (uint256 i = 0; i < assets.length; i++) {
            // Execute trades using the borrowed assets
            // This should complete within the same transaction
            _executeArbitrageTrades(assets[i], amounts[i]);

            // Ensure we have enough to repay
            uint256 amountOwed = amounts[i] + premiums[i];
            require(
                IERC20(assets[i]).balanceOf(address(this)) >= amountOwed,
                "Insufficient funds to repay flash loan"
            );

            // Approve repayment
            IERC20(assets[i]).safeApprove(msg.sender, amountOwed);
        }

        emit FlashLoanExecuted(assets, amounts, premiums, initiator);
        return true;
    }

    // Balancer flash loan callback
    function receiveFlashLoan(
        address[] memory tokens,
        uint256[] memory amounts,
        uint256[] memory feeAmounts,
        bytes memory userData
    ) external override {
        // Execute arbitrage logic
        for (uint256 i = 0; i < tokens.length; i++) {
            // Execute trades using the borrowed tokens
            _executeArbitrageTrades(tokens[i], amounts[i]);

            // Ensure we have enough to repay
            uint256 amountOwed = amounts[i] + feeAmounts[i];
            require(
                IERC20(tokens[i]).balanceOf(address(this)) >= amountOwed,
                "Insufficient funds to repay flash loan"
            );

            // Approve repayment
            IERC20(tokens[i]).safeApprove(msg.sender, amountOwed);
        }

        emit FlashLoanExecuted(tokens, amounts, feeAmounts, msg.sender);
    }

    // Uniswap V3 flash swap callback
    function uniswapV3SwapCallback(
        int256 amount0Delta,
        int256 amount1Delta,
        bytes calldata data
    ) external override {
        require(amount0Delta > 0 || amount1Delta > 0, "Invalid flash swap");

        (address token0, address token1) = abi.decode(data, (address, address));

        // Execute arbitrage trades if needed
        if (amount0Delta > 0) {
            _executeArbitrageTrades(token0, uint256(amount0Delta));
            IERC20(token0).safeTransfer(msg.sender, uint256(amount0Delta));
        }
        if (amount1Delta > 0) {
            _executeArbitrageTrades(token1, uint256(amount1Delta));
            IERC20(token1).safeTransfer(msg.sender, uint256(amount1Delta));
        }

        emit FlashSwapExecuted(
            token0,
            token1,
            uint256(amount0Delta),
            uint256(amount1Delta),
            0
        );
    }

    // Curve flash swap callback
    function curveFlashSwapCallback(
        address sender,
        address[] calldata tokensBorrowed,
        uint256[] calldata amountsBorrowed,
        uint256[] calldata feeAmounts,
        bytes calldata data
    ) external override {
        for (uint256 i = 0; i < tokensBorrowed.length; i++) {
            // Execute arbitrage trades
            _executeArbitrageTrades(tokensBorrowed[i], amountsBorrowed[i]);

            uint256 totalAmount = amountsBorrowed[i] + feeAmounts[i];
            IERC20(tokensBorrowed[i]).safeTransfer(msg.sender, totalAmount);
        }

        emit FlashLoanExecuted(
            tokensBorrowed,
            amountsBorrowed,
            feeAmounts,
            sender
        );
    }

    // Separate function to initiate cross-chain transfer of profits
    function bridgeProfits(
        address token,
        uint256 amount,
        string calldata targetChain
    ) external onlyAuthorized nonReentrant {
        require(amount > 0, "Amount must be greater than 0");
        require(bytes(targetChain).length > 0, "Invalid target chain");

        address bridge = chainBridges[targetChain];
        require(bridge != address(0), "Bridge not configured");

        // Check if we have enough realized profits
        require(realizedProfits[token] >= amount, "Insufficient realized profits");

        // Generate transfer ID
        bytes32 transferId = keccak256(
            abi.encodePacked(
                block.timestamp,
                token,
                amount,
                targetChain
            )
        );

        // Update profit tracking
        realizedProfits[token] -= amount;

        // Approve bridge contract
        IERC20(token).safeApprove(bridge, amount);

        // Mark transfer as pending
        pendingBridgeTransfers[transferId] = true;

        emit CrossChainMessageSent(
            uint32(block.chainid),
            abi.encodePacked(bridge),
            abi.encode(token, amount, targetChain),
            0
        );
    }

    // Internal function to execute arbitrage trades
    function _executeArbitrageTrades(
        address token,
        uint256 amount
    ) internal {
        // Implement your arbitrage logic here
        // This should execute trades using DEXs on the same chain
        // Any profits should be tracked in realizedProfits
    }

    // Admin functions
    function setAuthorizedCaller(
        address caller,
        bool authorized
    ) external onlyOwner {
        authorizedCallers[caller] = authorized;
    }

    function setChainBridge(
        string calldata chain,
        address bridge
    ) external onlyOwner {
        require(bridge != address(0), "Invalid bridge address");
        chainBridges[chain] = bridge;
    }

    function updateBridgeTransferStatus(
        bytes32 transferId,
        bool completed
    ) external onlyAuthorized {
        require(pendingBridgeTransfers[transferId], "Transfer not pending");
        pendingBridgeTransfers[transferId] = !completed;
    }

    // Emergency functions
    function rescueTokens(
        address token,
        address to,
        uint256 amount
    ) external onlyOwner {
        IERC20(token).safeTransfer(to, amount);
    }

    // Add interface for generic router calls
    interface IGenericRouter {
        function swap(bytes memory data) external returns (uint256);
    }

    receive() external payable {}
} 