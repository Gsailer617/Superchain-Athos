import os
import json
import logging
from web3 import Web3
import asyncio
from telegram import Bot
from telegram.error import InvalidToken, Unauthorized

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_telegram_bot(bot_token: str) -> tuple:
    """Verify Telegram bot token and get bot info"""
    try:
        bot = Bot(bot_token)
        bot_info = await bot.get_me()
        return True, bot_info.id, f"Bot verified: @{bot_info.username}"
    except InvalidToken:
        return False, None, "Invalid bot token"
    except Unauthorized:
        return False, None, "Unauthorized bot token"
    except Exception as e:
        return False, None, f"Error verifying bot: {str(e)}"

async def verify_admin_id(bot_token: str, admin_id: int) -> tuple:
    """Verify if admin ID is valid"""
    try:
        bot = Bot(bot_token)
        chat = await bot.get_chat(admin_id)
        return True, f"Admin verified: {chat.username}"
    except Exception as e:
        return False, f"Invalid admin ID: {str(e)}"

def verify_private_key(private_key: str) -> tuple:
    """Verify if private key is valid and get address"""
    try:
        # Remove '0x' prefix if present
        private_key = private_key.replace('0x', '')
        if len(private_key) != 64:
            return False, None, "Invalid private key length"
            
        web3 = Web3()
        account = web3.eth.account.from_key(private_key)
        return True, account.address, "Private key verified"
    except Exception as e:
        return False, None, f"Invalid private key: {str(e)}"

async def setup_test_config():
    """Interactive setup for test configuration"""
    config_file = 'test_config.py'
    
    try:
        # 1. Telegram Bot Setup
        print("\n=== Telegram Bot Setup ===")
        print("Please create a new bot with @BotFather on Telegram and enter the token below.")
        bot_token = input("Enter your Telegram bot token: ").strip()
        success, bot_id, message = await verify_telegram_bot(bot_token)
        if not success:
            print(f"Error: {message}")
            return
        print(f"Success: {message}")
        
        # 2. Admin Setup
        print("\n=== Admin Setup ===")
        print("Get your Telegram user ID from @userinfobot")
        admin_id = input("Enter your Telegram user ID: ").strip()
        try:
            admin_id = int(admin_id)
            success, message = await verify_admin_id(bot_token, admin_id)
            if not success:
                print(f"Error: {message}")
                return
            print(f"Success: {message}")
        except ValueError:
            print("Error: Admin ID must be a number")
            return
            
        # 3. Wallet Setup
        print("\n=== Wallet Setup ===")
        print("Enter a test wallet private key (never use a wallet with real funds)")
        private_key = input("Enter your test wallet private key: ").strip()
        success, address, message = verify_private_key(private_key)
        if not success:
            print(f"Error: {message}")
            return
        print(f"Success: Wallet address: {address}")
        
        # 4. Update Configuration
        print("\n=== Creating Configuration ===")
        # First check if template exists
        if not os.path.exists('test_config.template.py'):
            print("Error: test_config.template.py not found!")
            return
            
        # Copy template to new config file
        if not os.path.exists(config_file):
            with open('test_config.template.py', 'r') as src, open(config_file, 'w') as dst:
                dst.write(src.read())
        
        # Update configuration
        with open(config_file, 'r') as f:
            config_content = f.read()
            
        # Replace placeholders
        config_content = config_content.replace(
            "'<YOUR_PRIVATE_KEY>'", f"'{private_key}'"
        )
        config_content = config_content.replace(
            "'<YOUR_WALLET_ADDRESS>'", f"'{address}'"
        )
        config_content = config_content.replace(
            "'<YOUR_BOT_TOKEN>'", f"'{bot_token}'"
        )
        config_content = config_content.replace(
            "[123456789]", f"[{admin_id}]"
        )
        config_content = config_content.replace(
            "-100123456789", f"{bot_id}"
        )
        
        # Save updated configuration
        with open(config_file, 'w') as f:
            f.write(config_content)
            
        print("\n=== Setup Complete ===")
        print("Configuration has been updated successfully!")
        print(f"Bot Token: {bot_token[:10]}...{bot_token[-5:]}")
        print(f"Admin ID: {admin_id}")
        print(f"Wallet Address: {address}")
        
        # Create a backup of the original values
        backup = {
            'bot_token': bot_token,
            'admin_id': admin_id,
            'private_key': private_key,
            'address': address
        }
        with open('test_credentials_backup.json', 'w') as f:
            json.dump(backup, f, indent=2)
        print("\nCredentials backup saved to 'test_credentials_backup.json'")
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        print(f"\nError during setup: {str(e)}")

if __name__ == "__main__":
    asyncio.run(setup_test_config()) 