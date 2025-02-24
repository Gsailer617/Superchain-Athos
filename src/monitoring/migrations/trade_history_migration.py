"""
Migration script to help transition from TradeHistoryManager to TradeMonitor
"""

import asyncio
from pathlib import Path
import pandas as pd
import structlog
from typing import Optional
import shutil
from datetime import datetime

logger = structlog.get_logger(__name__)

async def migrate_trade_history(
    old_storage_path: str,
    new_storage_path: str,
    backup: bool = True
) -> bool:
    """Migrate trade history data from old format to new format
    
    Args:
        old_storage_path: Path to old trade history storage
        new_storage_path: Path to new trade monitor storage
        backup: Whether to create backup of old data
        
    Returns:
        bool: True if migration successful
    """
    try:
        old_path = Path(old_storage_path)
        new_path = Path(new_storage_path)
        
        if not old_path.exists():
            logger.error("Old storage path does not exist", path=str(old_path))
            return False
            
        # Create new directories
        new_path.mkdir(parents=True, exist_ok=True)
        historical_path = new_path / "historical"
        historical_path.mkdir(exist_ok=True)
        
        # Backup old data if requested
        if backup:
            backup_path = old_path.parent / f"{old_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(old_path, backup_path)
            logger.info("Created backup of old data", backup_path=str(backup_path))
        
        # Migrate parquet files
        migrated_count = 0
        for parquet_file in old_path.glob("*.parquet"):
            try:
                # Read old format
                df = pd.read_parquet(parquet_file)
                
                # Ensure all required columns exist
                required_columns = [
                    'timestamp', 'strategy', 'token_pair', 'dex',
                    'profit', 'gas_price', 'execution_time', 'success',
                    'additional_data'
                ]
                
                for col in required_columns:
                    if col not in df.columns:
                        logger.warning(
                            f"Missing column in old data",
                            file=parquet_file.name,
                            column=col
                        )
                        df[col] = None
                
                # Convert numeric columns to float
                numeric_cols = ['profit', 'gas_price', 'execution_time']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                
                # Save to new location
                new_file = historical_path / parquet_file.name
                df.to_parquet(new_file)
                migrated_count += 1
                
            except Exception as e:
                logger.error(
                    "Error migrating file",
                    file=parquet_file.name,
                    error=str(e)
                )
        
        logger.info(
            "Migration completed",
            files_migrated=migrated_count
        )
        
        return True
        
    except Exception as e:
        logger.error("Migration failed", error=str(e))
        return False

def get_migration_status(new_storage_path: str) -> dict:
    """Get status of migrated data
    
    Args:
        new_storage_path: Path to new storage
        
    Returns:
        dict: Migration status information
    """
    try:
        new_path = Path(new_storage_path)
        historical_path = new_path / "historical"
        
        if not historical_path.exists():
            return {
                'status': 'not_started',
                'files_migrated': 0,
                'total_trades': 0
            }
            
        # Count migrated files and trades
        files = list(historical_path.glob("*.parquet"))
        total_trades = 0
        
        for file in files:
            try:
                df = pd.read_parquet(file)
                total_trades += len(df)
            except Exception:
                continue
                
        return {
            'status': 'completed',
            'files_migrated': len(files),
            'total_trades': total_trades
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

async def verify_migration(
    old_storage_path: str,
    new_storage_path: str
) -> dict:
    """Verify migration was successful
    
    Args:
        old_storage_path: Path to old storage
        new_storage_path: Path to new storage
        
    Returns:
        dict: Verification results
    """
    try:
        old_path = Path(old_storage_path)
        new_path = Path(new_storage_path) / "historical"
        
        if not old_path.exists() or not new_path.exists():
            return {
                'success': False,
                'error': 'Storage paths do not exist'
            }
            
        # Compare file counts
        old_files = set(f.name for f in old_path.glob("*.parquet"))
        new_files = set(f.name for f in new_path.glob("*.parquet"))
        
        missing_files = old_files - new_files
        
        # Compare trade counts
        old_trades = 0
        new_trades = 0
        
        for file in old_path.glob("*.parquet"):
            try:
                df = pd.read_parquet(file)
                old_trades += len(df)
            except Exception:
                continue
                
        for file in new_path.glob("*.parquet"):
            try:
                df = pd.read_parquet(file)
                new_trades += len(df)
            except Exception:
                continue
                
        return {
            'success': len(missing_files) == 0 and old_trades == new_trades,
            'old_files': len(old_files),
            'new_files': len(new_files),
            'missing_files': list(missing_files),
            'old_trades': old_trades,
            'new_trades': new_trades
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        } 