#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    color=$1
    message=$2
    printf "${color}${message}${NC}\n"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to start local test chains
start_test_chains() {
    print_color $YELLOW "Starting local test chains..."
    
    # Start each chain in background
    ganache-cli --port 8545 --chainId 1 --deterministic > /dev/null 2>&1 &      # Ethereum
    ganache-cli --port 8546 --chainId 8453 --deterministic > /dev/null 2>&1 &   # Base
    ganache-cli --port 8547 --chainId 137 --deterministic > /dev/null 2>&1 &    # Polygon
    ganache-cli --port 8548 --chainId 42161 --deterministic > /dev/null 2>&1 &  # Arbitrum
    ganache-cli --port 8549 --chainId 10 --deterministic > /dev/null 2>&1 &     # Optimism
    ganache-cli --port 8550 --chainId 56 --deterministic > /dev/null 2>&1 &     # BNB Chain
    ganache-cli --port 8551 --chainId 59144 --deterministic > /dev/null 2>&1 &  # Linea
    ganache-cli --port 8552 --chainId 5000 --deterministic > /dev/null 2>&1 &   # Mantle
    ganache-cli --port 8553 --chainId 43114 --deterministic > /dev/null 2>&1 &  # Avalanche
    ganache-cli --port 8554 --chainId 100 --deterministic > /dev/null 2>&1 &    # Gnosis
    ganache-cli --port 8555 --chainId 34443 --deterministic > /dev/null 2>&1 &  # Mode
    ganache-cli --port 8556 --chainId 8899 --deterministic > /dev/null 2>&1 &   # Sonic
    
    # Wait for chains to start
    sleep 5
    
    # Set environment variables
    export ETHEREUM_RPC_URL="http://localhost:8545"
    export BASE_RPC_URL="http://localhost:8546"
    export POLYGON_RPC_URL="http://localhost:8547"
    export ARBITRUM_RPC_URL="http://localhost:8548"
    export OPTIMISM_RPC_URL="http://localhost:8549"
    export BNB_RPC_URL="http://localhost:8550"
    export LINEA_RPC_URL="http://localhost:8551"
    export MANTLE_RPC_URL="http://localhost:8552"
    export AVALANCHE_RPC_URL="http://localhost:8553"
    export GNOSIS_RPC_URL="http://localhost:8554"
    export MODE_RPC_URL="http://localhost:8555"
    export SONIC_RPC_URL="http://localhost:8556"
    
    # Set test contract addresses
    # Mode bridge contracts
    export MODE_L1_BRIDGE="0x0000000000000000000000000000000000001010"
    export MODE_L2_BRIDGE="0x0000000000000000000000000000000000001010"
    export MODE_MESSAGE_SERVICE="0x0000000000000000000000000000000000001011"
    
    # Sonic bridge contracts
    export SONIC_BRIDGE_ROUTER="0x0000000000000000000000000000000000001010"
    export SONIC_TOKEN_BRIDGE="0x0000000000000000000000000000000000001011"
    export SONIC_LIQUIDITY_POOL="0x0000000000000000000000000000000000001012"
    
    # Other bridge contracts
    export ARBITRUM_BRIDGE="0x72Ce9c846789fdB6fC1f34aC4AD25Dd9ef7031ef"
    export OPTIMISM_BRIDGE="0x99C9fc46f92E8a1c0deC1b1747d010903E884bE1"
    export BASE_BRIDGE="0x3154Cf16ccdb4C6d922629664174b904d80F2C35"
    export POLYGON_POS_BRIDGE="0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0"
    export POLYGON_PLASMA_BRIDGE="0x401F6c983eA34274ec46f84D70b31C151321188b"
    export LINEA_BRIDGE="0xE87d317eB8dcc9afE24d9f63D6C760e52Bc18A40"
    export MANTLE_BRIDGE="0x0000000000000000000000000000000000001010"
    export AVALANCHE_BRIDGE="0x1a44076050125825900e736c501f859c50fe728c"
    export GNOSIS_BRIDGE="0x88ad09518695c6c3712AC10a214bE5109a655671"
    
    # Set test gas settings
    export MODE_MAX_GAS_PRICE=500
    export MODE_PRIORITY_FEE=2
    export SONIC_MAX_GAS_PRICE=1000
    export SONIC_PRIORITY_FEE=1
    
    print_color $GREEN "Local test chains started"
}

# Function to stop local test chains
stop_test_chains() {
    print_color $YELLOW "Stopping local test chains..."
    pkill -f ganache-cli
}

# Function to run tests
run_tests() {
    test_type=$1
    
    case $test_type in
        "unit")
            print_color $GREEN "Running unit tests..."
            pytest tests/core -v -m "unit"
            ;;
        "integration")
            print_color $GREEN "Running integration tests..."
            pytest tests/integration -v -m "integration"
            ;;
        "all")
            print_color $GREEN "Running all tests..."
            pytest tests -v
            ;;
        *)
            print_color $RED "Invalid test type. Use: unit, integration, or all"
            exit 1
            ;;
    esac
}

# Function to setup test environment
setup_env() {
    # Check Python environment
    if [ ! -d "venv" ]; then
        print_color $YELLOW "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    print_color $YELLOW "Installing dependencies..."
    pip install -r requirements.txt
    
    # Start local test chains
    start_test_chains
}

# Function to cleanup
cleanup() {
    print_color $YELLOW "Cleaning up..."
    stop_test_chains
    deactivate
}

# Main execution
main() {
    # Parse command line arguments
    test_type=${1:-"all"}
    
    # Setup trap for cleanup
    trap cleanup EXIT
    
    # Setup environment
    setup_env
    
    # Run tests
    run_tests $test_type
    
    # Return test exit code
    exit $?
}

# Run main function
main "$@" 