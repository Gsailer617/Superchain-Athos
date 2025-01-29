// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IMToken {
    function flashLoan(
        address receiver,
        address token,
        uint256 amount,
        bytes calldata params
    ) external;
    function underlying() external view returns (address);
}

interface IMoonwellComptroller {
    function cTokensByUnderlying(address) external view returns (address);
}

interface IMorpho {
    function flashLoan(
        address token,
        uint256 amount,
        bytes calldata params
    ) external;
    function availableLiquidity(address token) external view returns (uint256);
}

interface IPool {
    struct ReserveData {
        uint256 configuration;
        uint128 liquidityIndex;
        uint128 currentLiquidityRate;
        uint128 variableBorrowIndex;
        uint128 currentVariableBorrowRate;
        uint128 currentStableBorrowRate;
        uint40 lastUpdateTimestamp;
        uint16 id;
        address aTokenAddress;
        address stableDebtTokenAddress;
        address variableDebtTokenAddress;
        address interestRateStrategyAddress;
        uint128 accruedToTreasury;
        uint128 unbacked;
        uint128 isolationModeTotalDebt;
        uint256 availableLiquidity;
    }

    function flashLoanSimple(
        address receiverAddress,
        address asset,
        uint256 amount,
        bytes calldata params,
        uint16 referralCode
    ) external;

    function getReserveData(address asset) external view returns (ReserveData memory);
}

interface IBalancerVault {
    function flashLoan(
        address recipient,
        address[] memory tokens,
        uint256[] memory amounts,
        bytes memory userData
    ) external;

    function getPoolId(address token) external view returns (bytes32);
    
    function getPoolTokens(bytes32 poolId) external view returns (
        address[] memory tokens,
        uint256[] memory balances,
        uint256 lastChangeBlock
    );
} 