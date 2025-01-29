// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "./interfaces/dex/IRouter.sol";
import "./interfaces/uniswap/IUniswapV3PoolWrapper.sol";
import "./interfaces/uniswap/IUniswapV3FactoryWrapper.sol";
import "./interfaces/IArbitrageTypes.sol";

contract FlashArbitrage is Ownable, ReentrancyGuard, Pausable, IArbitrageTypes {
    using SafeERC20 for IERC20;

    // DEX addresses
    address public immutable uniswapFactory;
    address public immutable uniswapRouter;
    address public immutable sushiswapRouter;
    address public immutable pancakeswapRouter;
    address public immutable WETH;
    
    // Minimum profit threshold (in basis points)
    uint256 public minProfitBps;
    uint256 public maxSlippageBps;
    
    // Slippage protection settings
    uint256 public constant MAX_SLIPPAGE_BPS = 1000; // 10% maximum allowed slippage setting

    // Add deadline settings
    uint256 public constant MAX_DEADLINE_EXTENSION = 3600; // 1 hour
    uint256 public deadlineExtension = 300; // 5 minutes default

    // Constants
    uint256 private immutable MINIMUM_PROFIT_BPS;
    uint256 private immutable MAXIMUM_SLIPPAGE_BPS;
    uint256 public constant DEFAULT_FEE_TIER = 3000; // 0.3%

    // Events
    event MinProfitUpdated(uint256 oldValue, uint256 newValue);
    event MaxSlippageUpdated(uint256 oldValue, uint256 newValue);
    event DeadlineExtensionUpdated(uint256 oldValue, uint256 newValue);
    event EmergencyWithdrawal(address token, address recipient, uint256 amount);
    event ArbitrageExecuted(
        address indexed tokenIn,
        address indexed tokenOut,
        uint256 amountIn,
        uint256 amountOut,
        uint256 profit,
        uint256 fees,
        DexRoute dexRoute
    );
    event TokenWhitelisted(address indexed token, bool status);
    event FlashSwapInitiated(
        address indexed tokenIn,
        address indexed tokenOut,
        uint256 amount,
        DexRoute dexRoute
    );

    // Token whitelist
    mapping(address => bool) public whitelistedTokens;

    constructor(
        address _uniswapFactory,
        address _uniswapRouter,
        address _sushiRouter,
        address _pancakeRouter,
        uint256 _minProfitBps,
        uint256 _maxSlippageBps,
        address _weth
    ) {
        require(_minProfitBps > 0, "Min profit must be > 0");
        require(_maxSlippageBps <= 1000, "Max slippage too high");
        
        minProfitBps = _minProfitBps;
        maxSlippageBps = _maxSlippageBps;
        MINIMUM_PROFIT_BPS = 50; // 0.5% minimum allowed
        MAXIMUM_SLIPPAGE_BPS = 1000; // 10% maximum allowed

        uniswapFactory = _uniswapFactory;
        uniswapRouter = _uniswapRouter;
        sushiswapRouter = _sushiRouter;
        pancakeswapRouter = _pancakeRouter;
        WETH = _weth;
    }

    function _validateAndCalculateProfit(
        uint256 amountIn,
        uint256 amountOut,
        uint256 fees
    ) internal view returns (uint256) {
        require(amountOut > amountIn + fees, "No profit");
        
        uint256 profit;
        unchecked {
            profit = amountOut - (amountIn + fees);
            uint256 minProfit = (amountIn * minProfitBps) / 10000;
            require(profit >= minProfit, "Insufficient profit margin");
        }
        
        return profit;
    }

    function _validateToken(address token) internal view {
        require(token != address(0), "Zero address token");
        require(token != WETH, "Direct WETH operations not allowed");
        require(whitelistedTokens[token], "Token not whitelisted");
    }

    function _calculateMinimumAmountOut(
        uint256 expectedAmount
    ) internal view returns (uint256) {
        unchecked {
            return expectedAmount - ((expectedAmount * maxSlippageBps) / 10000);
        }
    }

    function _executeSushiSwap(
        address tokenIn,
        address tokenOut,
        uint256 amountIn
    ) internal returns (uint256) {
        // Approve SushiSwap router
        IERC20(tokenIn).approve(sushiswapRouter, amountIn);
        
        // Create path
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;
        
        // Get minimum amount out
        uint256[] memory amounts = IRouter(sushiswapRouter).getAmountsOut(amountIn, path);
        uint256 minAmountOut = _calculateMinimumAmountOut(amounts[1]);
        
        // Execute swap
        uint256[] memory received = IRouter(sushiswapRouter).swapExactTokensForTokens(
            amountIn,
            minAmountOut,
            path,
            address(this),
            block.timestamp + deadlineExtension
        );
        
        // Reset approval
        IERC20(tokenIn).approve(sushiswapRouter, 0);
        
        return received[1];
    }

    function _executePancakeSwap(
        address tokenIn,
        address tokenOut,
        uint256 amountIn
    ) internal returns (uint256) {
        // Approve PancakeSwap router
        IERC20(tokenIn).approve(pancakeswapRouter, amountIn);
        
        // Create path
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;
        
        // Get minimum amount out
        uint256[] memory amounts = IRouter(pancakeswapRouter).getAmountsOut(amountIn, path);
        uint256 minAmountOut = _calculateMinimumAmountOut(amounts[1]);
        
        // Execute swap
        uint256[] memory received = IRouter(pancakeswapRouter).swapExactTokensForTokens(
            amountIn,
            minAmountOut,
            path,
            address(this),
            block.timestamp + deadlineExtension
        );
        
        // Reset approval
        IERC20(tokenIn).approve(pancakeswapRouter, 0);
        
        return received[1];
    }

    // Admin functions
    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    function setDeadlineExtension(uint256 _extension) external onlyOwner {
        require(_extension <= MAX_DEADLINE_EXTENSION, "Extension too long");
        uint256 oldValue = deadlineExtension;
        deadlineExtension = _extension;
        emit DeadlineExtensionUpdated(oldValue, _extension);
    }

    function setMinProfitBps(uint256 _minProfitBps) external onlyOwner {
        require(_minProfitBps >= MINIMUM_PROFIT_BPS, "Min profit too low");
        uint256 oldValue = minProfitBps;
        minProfitBps = _minProfitBps;
        emit MinProfitUpdated(oldValue, _minProfitBps);
    }

    function setMaxSlippageBps(uint256 _maxSlippageBps) external onlyOwner {
        require(_maxSlippageBps <= MAXIMUM_SLIPPAGE_BPS, "Slippage too high");
        uint256 oldValue = maxSlippageBps;
        maxSlippageBps = _maxSlippageBps;
        emit MaxSlippageUpdated(oldValue, _maxSlippageBps);
    }

    function whitelistToken(address token, bool status) external onlyOwner {
        require(token != address(0), "Invalid token");
        whitelistedTokens[token] = status;
        emit TokenWhitelisted(token, status);
    }

    function whitelistTokens(
        address[] calldata tokens,
        bool[] calldata statuses
    ) external onlyOwner {
        require(tokens.length == statuses.length, "Length mismatch");
        require(tokens.length > 0, "Empty arrays");
        
        for (uint256 i = 0; i < tokens.length; i++) {
            require(tokens[i] != address(0), "Invalid token");
            whitelistedTokens[tokens[i]] = statuses[i];
            emit TokenWhitelisted(tokens[i], statuses[i]);
        }
    }

    // Emergency functions
    function recoverStuckTokens(
        address[] calldata tokens,
        address recipient
    ) external onlyOwner nonReentrant {
        require(recipient != address(0), "Invalid recipient");
        
        for (uint256 i = 0; i < tokens.length; i++) {
            address token = tokens[i];
            require(token != address(0), "Invalid token");
            
            uint256 balance = IERC20(token).balanceOf(address(this));
            if (balance > 0) {
                IERC20(token).safeTransfer(recipient, balance);
                emit EmergencyWithdrawal(token, recipient, balance);
            }
        }
    }

    // View functions
    function getWhitelistStatuses(
        address[] calldata tokens
    ) external view returns (bool[] memory) {
        bool[] memory statuses = new bool[](tokens.length);
        for (uint256 i = 0; i < tokens.length; i++) {
            statuses[i] = whitelistedTokens[tokens[i]];
        }
        return statuses;
    }

    function getTokenBalances(
        address[] calldata tokens
    ) external view returns (uint256[] memory) {
        uint256[] memory balances = new uint256[](tokens.length);
        for (uint256 i = 0; i < tokens.length; i++) {
            balances[i] = IERC20(tokens[i]).balanceOf(address(this));
        }
        return balances;
    }

    // Flash swap functions
    function uniswapV3FlashCallback(
        uint256 fee0,
        uint256 fee1,
        bytes calldata data
    ) external nonReentrant whenNotPaused {
        require(msg.sender == address(IUniswapV3PoolWrapper(msg.sender).pool()), "Invalid caller");
        
        // Decode the flash loan parameters
        (address tokenIn, address tokenOut, uint256 amount, DexRoute dexRoute) = abi.decode(
            data,
            (address, address, uint256, DexRoute)
        );
        
        _validateToken(tokenIn);
        _validateToken(tokenOut);
        
        // Calculate total amount to repay including fees
        uint256 totalFee = fee0 + fee1;
        uint256 amountToRepay = amount + totalFee;
        
        // Execute the arbitrage based on the dexRoute
        uint256 amountReceived;
        if (dexRoute == DexRoute.UNISWAP_TO_SUSHI) {
            // Uniswap V3 -> SushiSwap
            amountReceived = _executeSushiSwap(tokenIn, tokenOut, amount);
        } else if (dexRoute == DexRoute.UNISWAP_TO_PANCAKE) {
            // Uniswap V3 -> PancakeSwap
            amountReceived = _executePancakeSwap(tokenIn, tokenOut, amount);
        } else {
            revert("Invalid DEX route");
        }
        
        // Validate profit
        uint256 profit = _validateAndCalculateProfit(amount, amountReceived, totalFee);
        
        // Approve and repay the flash loan
        IERC20(tokenIn).approve(msg.sender, amountToRepay);

        emit ArbitrageExecuted(
            tokenIn,
            tokenOut,
            amount,
            amountReceived,
            profit,
            totalFee,
            dexRoute
        );
    }

    function initiateFlashSwap(
        address tokenIn,
        address tokenOut,
        uint256 amount,
        DexRoute dexRoute
    ) external onlyOwner whenNotPaused {
        require(dexRoute != DexRoute.NONE, "Invalid DEX route");
        _validateToken(tokenIn);
        _validateToken(tokenOut);

        // Get the pool address
        address pool = IUniswapV3FactoryWrapper(uniswapFactory).getPool(
            tokenIn,
            tokenOut,
            uint24(DEFAULT_FEE_TIER)
        );
        require(pool != address(0), "Pool does not exist");

        // Encode the flash swap data
        bytes memory data = abi.encode(tokenIn, tokenOut, amount, dexRoute);

        emit FlashSwapInitiated(tokenIn, tokenOut, amount, dexRoute);

        // Call flash swap on the pool
        IUniswapV3PoolWrapper(pool).flash(
            address(this),
            0, // amount0
            amount, // amount1
            data
        );
    }
} 