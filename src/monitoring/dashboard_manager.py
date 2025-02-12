import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import structlog
import aiofiles
import aiohttp
from prometheus_client import Counter, Gauge

logger = structlog.get_logger(__name__)

class DashboardManager:
    def __init__(
        self,
        backup_dir: str = "dashboard_backups",
        grafana_url: str = "http://localhost:3000",
        grafana_token: Optional[str] = None
    ):
        self.backup_dir = backup_dir
        self.grafana_url = grafana_url
        self.grafana_token = grafana_token or os.getenv("GRAFANA_API_TOKEN")
        
        # Ensure backup directory exists
        os.makedirs(backup_dir, exist_ok=True)
        
        # Dashboard metrics
        self.dashboard_versions = Counter(
            'dashboard_versions_total',
            'Total number of dashboard versions',
            ['dashboard']
        )
        
        self.last_backup_time = Gauge(
            'dashboard_backup_timestamp',
            'Timestamp of last dashboard backup',
            ['dashboard']
        )
        
        self.backup_size = Gauge(
            'dashboard_backup_size_bytes',
            'Size of dashboard backup in bytes',
            ['dashboard']
        )

    async def backup_dashboard(self, dashboard_uid: str) -> bool:
        """
        Create a backup of a dashboard.
        
        Args:
            dashboard_uid: Grafana dashboard UID
        """
        try:
            # Fetch dashboard from Grafana
            dashboard_json = await self._fetch_dashboard(dashboard_uid)
            if not dashboard_json:
                return False
            
            # Create timestamp for version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create backup filename
            backup_file = os.path.join(
                self.backup_dir,
                f"{dashboard_uid}_{timestamp}.json"
            )
            
            # Save backup
            async with aiofiles.open(backup_file, 'w') as f:
                await f.write(json.dumps(dashboard_json, indent=2))
            
            # Update metrics
            self.dashboard_versions.labels(dashboard=dashboard_uid).inc()
            self.last_backup_time.labels(dashboard=dashboard_uid).set_to_current_time()
            self.backup_size.labels(dashboard=dashboard_uid).set(
                os.path.getsize(backup_file)
            )
            
            # Cleanup old versions (keep last 10)
            await self._cleanup_old_versions(dashboard_uid)
            
            logger.info("Dashboard backup created",
                       dashboard=dashboard_uid,
                       backup_file=backup_file)
            return True
            
        except Exception as e:
            logger.error("Error backing up dashboard",
                        dashboard=dashboard_uid,
                        error=str(e))
            return False

    async def restore_dashboard(
        self,
        dashboard_uid: str,
        version: Optional[str] = None
    ) -> bool:
        """
        Restore a dashboard from backup.
        
        Args:
            dashboard_uid: Dashboard UID
            version: Specific version timestamp to restore (default: latest)
        """
        try:
            # Find backup file
            backup_file = await self._find_backup(dashboard_uid, version)
            if not backup_file:
                return False
            
            # Read backup
            async with aiofiles.open(backup_file, 'r') as f:
                dashboard_json = json.loads(await f.read())
            
            # Restore to Grafana
            success = await self._update_dashboard(dashboard_json)
            
            if success:
                logger.info("Dashboard restored",
                           dashboard=dashboard_uid,
                           version=version or "latest")
            
            return success
            
        except Exception as e:
            logger.error("Error restoring dashboard",
                        dashboard=dashboard_uid,
                        version=version,
                        error=str(e))
            return False

    async def list_versions(self, dashboard_uid: str) -> List[Dict]:
        """List available versions for a dashboard"""
        try:
            versions = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith(dashboard_uid):
                    timestamp = filename.split('_')[1].split('.')[0]
                    backup_file = os.path.join(self.backup_dir, filename)
                    
                    versions.append({
                        'version': timestamp,
                        'size': os.path.getsize(backup_file),
                        'created_at': datetime.strptime(
                            timestamp,
                            "%Y%m%d_%H%M%S"
                        ).isoformat()
                    })
            
            return sorted(versions, key=lambda x: x['version'], reverse=True)
            
        except Exception as e:
            logger.error("Error listing dashboard versions",
                        dashboard=dashboard_uid,
                        error=str(e))
            return []

    async def _fetch_dashboard(self, dashboard_uid: str) -> Optional[Dict]:
        """Fetch dashboard from Grafana API"""
        headers = {
            'Authorization': f'Bearer {self.grafana_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.grafana_url}/api/dashboards/uid/{dashboard_uid}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error("Failed to fetch dashboard",
                                   status=response.status,
                                   dashboard=dashboard_uid)
                        return None
                        
        except Exception as e:
            logger.error("Error fetching dashboard",
                        dashboard=dashboard_uid,
                        error=str(e))
            return None

    async def _update_dashboard(self, dashboard_json: Dict) -> bool:
        """Update dashboard in Grafana"""
        headers = {
            'Authorization': f'Bearer {self.grafana_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.grafana_url}/api/dashboards/db",
                    headers=headers,
                    json=dashboard_json
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error("Error updating dashboard", error=str(e))
            return False

    async def _cleanup_old_versions(self, dashboard_uid: str):
        """Keep only the last 10 versions of a dashboard"""
        versions = await self.list_versions(dashboard_uid)
        if len(versions) > 10:
            for version in versions[10:]:
                backup_file = os.path.join(
                    self.backup_dir,
                    f"{dashboard_uid}_{version['version']}.json"
                )
                try:
                    os.remove(backup_file)
                except Exception as e:
                    logger.error("Error removing old backup",
                               file=backup_file,
                               error=str(e))

    async def _find_backup(
        self,
        dashboard_uid: str,
        version: Optional[str] = None
    ) -> Optional[str]:
        """Find backup file for dashboard"""
        versions = await self.list_versions(dashboard_uid)
        if not versions:
            return None
            
        if version:
            for v in versions:
                if v['version'] == version:
                    return os.path.join(
                        self.backup_dir,
                        f"{dashboard_uid}_{version}.json"
                    )
            return None
        
        # Return latest version
        latest = versions[0]
        return os.path.join(
            self.backup_dir,
            f"{dashboard_uid}_{latest['version']}.json"
        ) 