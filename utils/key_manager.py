import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
from config.environment import env_config

logger = logging.getLogger(__name__)

class KeyManager:
    """Secure key management system"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self._keys = {}
        self._setup_encryption()
        
    def _setup_encryption(self):
        """Setup encryption for sensitive data"""
        # Use environment-specific salt
        salt = base64.b64encode(os.urandom(16))
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(env_config.MAINNET_PRIVATE_KEY.encode()))
        self._fernet = Fernet(key)
        
    def encrypt_key(self, key_type: str, key_value: str) -> str:
        """Encrypt sensitive key data"""
        try:
            return self._fernet.encrypt(key_value.encode()).decode()
        except Exception as e:
            logger.error(f"Error encrypting {key_type}: {str(e)}")
            raise
            
    def decrypt_key(self, key_type: str, encrypted_key: str) -> str:
        """Decrypt sensitive key data"""
        try:
            return self._fernet.decrypt(encrypted_key.encode()).decode()
        except Exception as e:
            logger.error(f"Error decrypting {key_type}: {str(e)}")
            raise
            
    def store_key(self, key_type: str, key_value: str):
        """Securely store a key"""
        try:
            encrypted_key = self.encrypt_key(key_type, key_value)
            self._keys[key_type] = encrypted_key
        except Exception as e:
            logger.error(f"Error storing {key_type}: {str(e)}")
            raise
            
    def get_key(self, key_type: str) -> str:
        """Retrieve and decrypt a stored key"""
        try:
            encrypted_key = self._keys.get(key_type)
            if not encrypted_key:
                raise ValueError(f"Key not found: {key_type}")
            return self.decrypt_key(key_type, encrypted_key)
        except Exception as e:
            logger.error(f"Error retrieving {key_type}: {str(e)}")
            raise
            
    def clear_keys(self):
        """Clear all stored keys"""
        self._keys.clear()
        
# Global instance
key_manager = KeyManager() 