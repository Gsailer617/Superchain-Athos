// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "./interfaces/uniswap/IUniswapV3PoolWrapper.sol";
import "./interfaces/uniswap/IUniswapV3FlashCallbackWrapper.sol";
import "./interfaces/uniswap/IUniswapV3FactoryWrapper.sol";
import "./interfaces/uniswap/ISwapRouterWrapper.sol";
import "./interfaces/uniswap/TickMathWrapper.sol";
import "./interfaces/IArbitrageTypes.sol";
import "./TradeRecorder.sol";
import "./modules/ArbitrageExecutor.sol";

contract FlashLoanArbitrage is 
    Ownable,
    ArbitrageExecutor,
    ReentrancyGuard,
    Pausable,
    IArbitrageTypes
{
    using SafeERC20 for IERC20;

    // Protocol addresses
    address public immutable WETH;
    address public immutable UNISWAP_QUOTER_V2;
    address public immutable uniswapRouter;
    address public immutable sushiswapRouter;
    address public immutable pancakeswapRouter;
    
    // Minimum profit threshold (in basis points)
    uint256 public minProfitBps;
    uint256 public maxSlippageBps;
    
    // Slippage protection settings
    uint256 public constant MAX_SLIPPAGE_BPS = 1000; // 10% maximum allowed slippage setting

    // Storage for detailed records
    ITradeRecorder public tradeRecorder;

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
        uint256 fees
    );
    event TokenWhitelisted(address indexed token, bool status);

    // Token whitelist
    mapping(address => bool) public whitelistedTokens;

    // Initialization flag
    bool private initialized;
    event ContractInitialized(address indexed owner, uint256 timestamp);

    modifier whenInitialized() {
        require(initialized, "Contract not initialized");
        _;
    }

    constructor(
        address _uniswapFactory,
        address _swapRouter,
        address _sushiRouter,
        address _pancakeRouter,
        uint256 _minProfitBps,
        uint256 _maxSlippageBps,
        address _weth,
        address _uniswapQuoterV2
    ) 
        ArbitrageExecutor(_uniswapFactory, _swapRouter)
    {
        require(_minProfitBps > 0, "Min profit must be > 0");
        require(_maxSlippageBps <= 1000, "Max slippage too high");
        
        tradeRecorder = new TradeRecorder();
        minProfitBps = _minProfitBps;
        maxSlippageBps = _maxSlippageBps;
        MINIMUM_PROFIT_BPS = 50; // 0.5% minimum allowed
        MAXIMUM_SLIPPAGE_BPS = 1000; // 10% maximum allowed

        WETH = _weth;
        UNISWAP_QUOTER_V2 = _uniswapQuoterV2;
        uniswapRouter = _swapRouter;
        sushiswapRouter = _sushiRouter;
        pancakeswapRouter = _pancakeRouter;

        initialized = true;
        emit ContractInitialized(msg.sender, block.timestamp);
    }

    // Core flash loan functions
    function _validateAndCalculateProfit(
        uint256 amountIn,
        uint256 amountOut,
        uint256 fees
    ) internal returns (uint256) {
        require(amountOut > amountIn + fees, "No profit");
        
        // Use unchecked for gas optimization where overflow is impossible
        uint256 profit;
        unchecked {
            profit = amountOut - (amountIn + fees);
            // Calculate minimum required profit
            uint256 minProfit = (amountIn * minProfitBps) / 10000;
            require(profit >= minProfit, "Insufficient profit margin");
        }

        emit ArbitrageExecuted(
            msg.sender,
            address(this),
            amountIn,
            amountOut,
            profit,
            fees
        );
        
        return profit;
    }

    function _validateToken(address token) internal view {
        require(token != address(0), "Zero address token");
        require(token != WETH, "Direct WETH operations not allowed");
        require(whitelistedTokens[token], "Token not whitelisted");
    }

    function uniswapV3FlashCallback(
        uint256 fee0,
        uint256 fee1,
        bytes calldata data
    ) external nonReentrant whenNotPaused whenInitialized {
        require(msg.sender == uniswapFactory || IUniswapV3PoolWrapper(msg.sender).pool() == msg.sender, "Invalid caller");
        
        // Decode the flash loan parameters
        (address tokenIn, address tokenOut, uint256 amount, uint8 dexRoute) = abi.decode(data, (address, address, uint256, uint8));
        
        _validateToken(tokenIn);
        _validateToken(tokenOut);
        
        // Calculate total amount to repay including fees
        uint256 totalFee = fee0 + fee1;
        uint256 amountToRepay = amount + totalFee;
        
        // Execute the arbitrage based on the dexRoute
        uint256 amountReceived;
        if (dexRoute == 1) {
            // Uniswap V3 -> SushiSwap
            amountReceived = _executeSushiSwap(tokenIn, tokenOut, amount);
        } else if (dexRoute == 2) {
            // Uniswap V3 -> PancakeSwap
            amountReceived = _executePancakeSwap(tokenIn, tokenOut, amount);
        } else {
            revert("Invalid DEX route");
        }
        
        // Validate profit
        _validateAndCalculateProfit(amount, amountReceived, totalFee);
        
        // Approve and repay the flash loan
        IERC20(tokenIn).approve(msg.sender, amountToRepay);
    }

    // Admin functions
    function pause() external onlyOwner whenInitialized {
        _pause();
    }

    function unpause() external onlyOwner whenInitialized {
        _unpause();
    }

    // Emergency functions
    function recoverStuckTokens(
        address[] calldata tokens,
        address recipient
    ) external onlyOwner nonReentrant whenInitialized {
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

    function setDeadlineExtension(uint256 _extension) external onlyOwner whenInitialized {
        require(_extension <= MAX_DEADLINE_EXTENSION, "Extension too long");
        uint256 oldValue = deadlineExtension;
        deadlineExtension = _extension;
        emit DeadlineExtensionUpdated(oldValue, _extension);
    }

    function _executeUniswapV3Swap(
        address tokenIn,
        uint256 amountIn,
        address tokenOut,
        bytes memory path
    ) internal returns (uint256) {
        require(tokenIn != address(0) && tokenOut != address(0), "Invalid tokens");
        require(amountIn > 0, "Invalid amount");

        // Approve Uniswap router to spend tokens
        IERC20(tokenIn).approve(address(uniswapRouter), amountIn);

        // Prepare swap parameters
        ISwapRouterWrapper.ExactInputParams memory params = ISwapRouterWrapper.ExactInputParams({
            path: path,
            recipient: address(this),
            deadline: block.timestamp + deadlineExtension,
            amountIn: amountIn,
            amountOutMinimum: 0 // We check the output amount after the swap
        });

        // Execute the swap
        uint256 amountOut = ISwapRouterWrapper(uniswapRouter).exactInput(params);
        require(amountOut > 0, "Zero amount received from Uniswap");

        // Reset approval
        IERC20(tokenIn).approve(address(uniswapRouter), 0);

        return amountOut;
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

        // Execute swap
        uint256[] memory amounts = IRouter(sushiswapRouter).swapExactTokensForTokens(
            amountIn,
            0, // We check the output amount after the swap
            path,
            address(this),
            block.timestamp + deadlineExtension
        );

        // Reset approval
        IERC20(tokenIn).approve(sushiswapRouter, 0);

        return amounts[1];
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

        // Execute swap
        uint256[] memory amounts = IRouter(pancakeswapRouter).swapExactTokensForTokens(
            amountIn,
            0, // We check the output amount after the swap
            path,
            address(this),
            block.timestamp + deadlineExtension
        );

        // Reset approval
        IERC20(tokenIn).approve(pancakeswapRouter, 0);

        return amounts[1];
    }

    function whitelistToken(address token, bool status) external onlyOwner {
        require(token != address(0), "Invalid token address");
        whitelistedTokens[token] = status;
        emit TokenWhitelisted(token, status);
    }

    function setMinProfitBps(uint256 _minProfitBps) external onlyOwner whenInitialized {
        require(_minProfitBps >= MINIMUM_PROFIT_BPS, "Min profit too low");
        uint256 oldValue = minProfitBps;
        minProfitBps = _minProfitBps;
        emit MinProfitUpdated(oldValue, _minProfitBps);
    }

    function setMaxSlippageBps(uint256 _maxSlippageBps) external onlyOwner whenInitialized {
        require(_maxSlippageBps <= MAXIMUM_SLIPPAGE_BPS, "Max slippage too high");
        uint256 oldValue = maxSlippageBps;
        maxSlippageBps = _maxSlippageBps;
        emit MaxSlippageUpdated(oldValue, _maxSlippageBps);
    }
} 