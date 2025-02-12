import gzip
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import structlog
from prometheus_client import Counter, Gauge
import shutil
import aiofiles
import asyncio

logger = structlog.get_logger(__name__)

class DataManager:
    def __init__(
        self,
        data_dir: str = "monitoring_data",
        retention_days: int = 90,
        compression_threshold_mb: int = 100
    ):
        self.data_dir = data_dir
        self.retention_days = retention_days
        self.compression_threshold_bytes = compression_threshold_mb * 1024 * 1024
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "compressed"), exist_ok=True)
        
        # Metrics
        self.data_size = Gauge(
            'monitoring_data_size_bytes',
            'Size of monitoring data in bytes',
            ['type']
        )
        
        self.compression_ratio = Gauge(
            'data_compression_ratio',
            'Compression ratio of monitoring data',
            ['type']
        )
        
        self.retention_deletions = Counter(
            'retention_deletions_total',
            'Number of files deleted due to retention policy',
            ['type']
        )

    async def store_data(
        self,
        data_type: str,
        data: Union[Dict, List],
        timestamp: Optional[datetime] = None
    ) -> str:
        """Store monitoring data with optional compression"""
        timestamp = timestamp or datetime.now()
        filename = f"{data_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            # Write data
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(data, indent=2))
            
            # Check file size
            file_size = os.path.getsize(filepath)
            self.data_size.labels(type=data_type).set(file_size)
            
            # Compress if needed
            if file_size > self.compression_threshold_bytes:
                compressed_path = await self._compress_file(filepath)
                os.remove(filepath)  # Remove original
                return compressed_path
            
            return filepath
            
        except Exception as e:
            logger.error("Error storing data",
                        data_type=data_type,
                        error=str(e))
            return ""

    async def load_data(
        self,
        filepath: str
    ) -> Optional[Union[Dict, List]]:
        """Load data from file, handling compression"""
        try:
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rt') as f:
                    return json.loads(f.read())
            else:
                async with aiofiles.open(filepath, 'r') as f:
                    content = await f.read()
                    return json.loads(content)
                    
        except Exception as e:
            logger.error("Error loading data",
                        filepath=filepath,
                        error=str(e))
            return None

    async def enforce_retention(self):
        """Enforce data retention policy"""
        retention_date = datetime.now() - timedelta(days=self.retention_days)
        
        try:
            # Check regular files
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.data_dir, filename)
                    file_date = datetime.strptime(
                        filename.split('_')[1].split('.')[0],
                        '%Y%m%d'
                    )
                    
                    if file_date < retention_date:
                        os.remove(filepath)
                        self.retention_deletions.labels(
                            type='uncompressed'
                        ).inc()
            
            # Check compressed files
            compressed_dir = os.path.join(self.data_dir, "compressed")
            for filename in os.listdir(compressed_dir):
                if filename.endswith('.gz'):
                    filepath = os.path.join(compressed_dir, filename)
                    file_date = datetime.strptime(
                        filename.split('_')[1].split('.')[0],
                        '%Y%m%d'
                    )
                    
                    if file_date < retention_date:
                        os.remove(filepath)
                        self.retention_deletions.labels(
                            type='compressed'
                        ).inc()
                        
        except Exception as e:
            logger.error("Error enforcing retention policy",
                        error=str(e))

    async def _compress_file(self, filepath: str) -> str:
        """Compress a file using gzip"""
        try:
            filename = os.path.basename(filepath)
            compressed_path = os.path.join(
                self.data_dir,
                "compressed",
                f"{filename}.gz"
            )
            
            # Compress file
            with open(filepath, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Calculate and record compression ratio
            original_size = os.path.getsize(filepath)
            compressed_size = os.path.getsize(compressed_path)
            ratio = original_size / compressed_size
            
            self.compression_ratio.labels(
                type=filename.split('_')[0]
            ).set(ratio)
            
            return compressed_path
            
        except Exception as e:
            logger.error("Error compressing file",
                        filepath=filepath,
                        error=str(e))
            return filepath

    async def start_retention_task(self):
        """Start periodic retention enforcement"""
        while True:
            await self.enforce_retention()
            await asyncio.sleep(24 * 60 * 60)  # Run daily 