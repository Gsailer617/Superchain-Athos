# Configuration Guide

## Environment Variables

This document details all configuration options available through environment variables.

### Required Variables

1. `TELEGRAM_BOT_TOKEN`
   - Purpose: Authentication token for the Telegram bot
   - How to get: Contact @BotFather on Telegram
   - Required: Yes

2. `TELEGRAM_CHAT_ID`
   - Purpose: Target chat for bot notifications
   - How to get: Send `/start` to @userinfobot
   - Required: Yes

3. `TELEGRAM_ADMIN_IDS`
   - Purpose: Authorized users for bot commands
   - Format: Comma-separated list of Telegram user IDs
   - Required: Yes

4. `MAINNET_PRIVATE_KEY`
   - Purpose: Wallet mainnet private key for transactions
   - Security: Keep this secret and never share
   - Required: Yes

5. `WEB3_PROVIDER_URI`
   - Purpose: Connection to Base blockchain
   - Example: https://mainnet.base.org
   - Required: Yes

### Optional Variables

1. `LOG_LEVEL`
   - Purpose: Controls logging verbosity
   - Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
   - Default: INFO
   - Usage:
     - DEBUG: Detailed debugging information
     - INFO: General operational information
     - WARNING: Potential issues that need attention
     - ERROR: Error events that might still allow operation
     - CRITICAL: Serious errors that prevent operation

2. `HF_API_KEY`
   - Purpose: HuggingFace API access
   - Required: No
   - Default: None

3. `DEFILLAMA_API_KEY`
   - Purpose: DeFiLlama market data access
   - Required: No
   - Default: None

4. `BASESCAN_API_KEY`
   - Purpose: BaseScan contract verification
   - Required: No
   - Default: None

## Logging Configuration

The application uses a structured logging system that includes:
- Timestamp
- Module name
- Log level
- Message

Example log format:
