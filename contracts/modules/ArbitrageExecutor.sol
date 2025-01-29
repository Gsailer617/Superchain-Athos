// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "../interfaces/dex/IRouter.sol";
import "../interfaces/uniswap/IUniswapV3FactoryWrapper.sol";
import "../interfaces/uniswap/ISwapRouterWrapper.sol";

contract ArbitrageExecutor is Ownable {
    using SafeERC20 for IERC20;

    // Move storage here
    mapping(address => bool) public isTokenSupported;
    address[] public supportedTokens;

    address public immutable uniswapFactory;
    address public immutable swapRouter;
    uint24 public constant UNISWAP_FEE_TIER = 3000; // 0.3%

    // Events
    event TradeRecorded(
        address indexed tokenIn,
        address indexed tokenOut,
        uint256 amountIn,
        uint256 amountOut,
        uint256 profit,
        uint256 timestamp
    );

    constructor(address _uniswapFactory, address _swapRouter) {
        require(_uniswapFactory != address(0), "Invalid factory");
        require(_swapRouter != address(0), "Invalid router");
        uniswapFactory = _uniswapFactory;
        swapRouter = _swapRouter;
    }

    function _executeArbitrage(
        address tokenIn,
        uint256 amountIn,
        address tokenOut,
        bytes memory path
    ) internal returns (uint256) {
        // Approve the router to spend our tokens
        IERC20(tokenIn).approve(swapRouter, amountIn);
        
        // Execute the swap
        ISwapRouterWrapper router = ISwapRouterWrapper(swapRouter);
        uint256 amountOut = router.exactInput(
            ISwapRouterWrapper.ExactInputParams({
                path: path,
                recipient: address(this),
                deadline: block.timestamp + 300,
                amountIn: amountIn,
                amountOutMinimum: 0 // We check profitability in the calling function
            })
        );

        // Record the trade
        recordTrade(
            tokenIn,
            tokenOut,
            amountIn,
            amountOut,
            amountOut > amountIn ? amountOut - amountIn : 0,
            "UniswapV3",
            "UniswapV3"
        );

        return amountOut;
    }

    function getOptimalPath(
        address tokenIn,
        address tokenOut
    ) public view returns (bytes memory) {
        require(isTokenSupported[tokenIn], "Token in not supported");
        require(isTokenSupported[tokenOut], "Token out not supported");
        
        return abi.encodePacked(
            tokenIn,
            uint24(UNISWAP_FEE_TIER),
            tokenOut
        );
    }

    function recordTrade(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 amountOut,
        uint256 profit,
        string memory /* sourceDex */,  // Unused but kept for interface compatibility
        string memory /* targetDex */   // Unused but kept for interface compatibility
    ) internal virtual {
        _recordTrade(tokenIn, tokenOut, amountIn, amountOut, profit, "", "");
    }

    function _recordTrade(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 amountOut,
        uint256 profit,
        string memory /* sourceDex */,  // Commented out unused parameter
        string memory /* targetDex */   // Commented out unused parameter
    ) internal {
        emit TradeRecorded(
            tokenIn,
            tokenOut,
            amountIn,
            amountOut,
            profit,
            block.timestamp
        );
    }
} 