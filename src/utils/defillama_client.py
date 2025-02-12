import aiohttp
from typing import Dict, Optional

class DefiLlama:
    def __init__(self):
        self.base_url = "https://api.llama.fi"
        
    async def get_protocol(self, protocol_slug: str) -> Optional[Dict]:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/protocol/{protocol_slug}") as response:
                if response.status == 200:
                    return await response.json()
                return None
                
    async def get_protocol_tvl(self, protocol_slug: str) -> Optional[Dict]:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/tvl/{protocol_slug}") as response:
                if response.status == 200:
                    return await response.json()
                return None 