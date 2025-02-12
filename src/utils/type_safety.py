"""
Type safety utilities for data conversions and validation
"""

from typing import TypeVar, Type, Union, Optional, Any, Dict, List, cast
from decimal import Decimal, InvalidOperation
from web3.types import Wei
from eth_typing import ChecksumAddress
from web3 import Web3
import json
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    converted_value: Optional[Any] = None

class ConversionError(Exception):
    """Error during type conversion"""
    pass

class ValidationError(Exception):
    """Error during data validation"""
    pass

class TypeConverter:
    """Safe type conversion utilities"""
    
    @staticmethod
    def to_int(
        value: Any,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> int:
        """Safely convert value to integer"""
        try:
            if isinstance(value, bool):
                raise ValueError("Boolean values cannot be converted to int")
                
            result = int(float(str(value)))
            
            if min_value is not None and result < min_value:
                raise ValueError(f"Value {result} is below minimum {min_value}")
                
            if max_value is not None and result > max_value:
                raise ValueError(f"Value {result} exceeds maximum {max_value}")
                
            return result
            
        except (ValueError, TypeError) as e:
            raise ConversionError(f"Could not convert {value} to integer: {str(e)}")

    @staticmethod
    def to_float(
        value: Any,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        precision: Optional[int] = None
    ) -> float:
        """Safely convert value to float"""
        try:
            if isinstance(value, bool):
                raise ValueError("Boolean values cannot be converted to float")
                
            result = float(str(value))
            
            if min_value is not None and result < min_value:
                raise ValueError(f"Value {result} is below minimum {min_value}")
                
            if max_value is not None and result > max_value:
                raise ValueError(f"Value {result} exceeds maximum {max_value}")
                
            if precision is not None:
                result = round(result, precision)
                
            return result
            
        except (ValueError, TypeError) as e:
            raise ConversionError(f"Could not convert {value} to float: {str(e)}")

    @staticmethod
    def to_decimal(
        value: Any,
        precision: int = 18
    ) -> Decimal:
        """Safely convert value to Decimal"""
        try:
            if isinstance(value, bool):
                raise ValueError("Boolean values cannot be converted to Decimal")
                
            return Decimal(str(value)).quantize(Decimal(f'0.{"0" * precision}'))
            
        except (InvalidOperation, TypeError) as e:
            raise ConversionError(f"Could not convert {value} to Decimal: {str(e)}")

    @staticmethod
    def to_wei(
        value: Union[int, float, str, Decimal],
        decimals: int = 18
    ) -> Wei:
        """Safely convert value to Wei"""
        try:
            if isinstance(value, bool):
                raise ValueError("Boolean values cannot be converted to Wei")
                
            # Convert to Decimal first for precision
            amount = Decimal(str(value))
            wei_value = int(amount * Decimal(10 ** decimals))
            return Wei(wei_value)
            
        except (InvalidOperation, TypeError, ValueError) as e:
            raise ConversionError(f"Could not convert {value} to Wei: {str(e)}")

    @staticmethod
    def from_wei(
        value: Union[Wei, int, str],
        decimals: int = 18
    ) -> Decimal:
        """Safely convert Wei to Decimal"""
        try:
            wei_value = Wei(int(value))
            return Decimal(wei_value) / Decimal(10 ** decimals)
            
        except (ValueError, TypeError) as e:
            raise ConversionError(f"Could not convert {value} from Wei: {str(e)}")

    @staticmethod
    def to_checksum_address(value: str) -> ChecksumAddress:
        """Safely convert address to checksum format"""
        try:
            if not Web3.is_address(value):
                raise ValueError(f"Invalid Ethereum address: {value}")
                
            return Web3.to_checksum_address(value)
            
        except ValueError as e:
            raise ConversionError(f"Could not convert to checksum address: {str(e)}")

    @staticmethod
    def to_bool(value: Any) -> bool:
        """Safely convert value to boolean"""
        if isinstance(value, bool):
            return value
            
        if isinstance(value, (int, float)):
            return bool(value)
            
        if isinstance(value, str):
            value = value.lower().strip()
            if value in ('true', '1', 'yes', 'on'):
                return True
            if value in ('false', '0', 'no', 'off'):
                return False
                
        raise ConversionError(f"Could not convert {value} to boolean")

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_numeric_range(
        value: Union[int, float, Decimal],
        min_value: Optional[Union[int, float, Decimal]] = None,
        max_value: Optional[Union[int, float, Decimal]] = None,
        field_name: str = "Value"
    ) -> ValidationResult:
        """Validate numeric value within range"""
        errors = []
        warnings = []
        
        try:
            if min_value is not None and value < min_value:
                errors.append(f"{field_name} {value} is below minimum {min_value}")
                
            if max_value is not None and value > max_value:
                errors.append(f"{field_name} {value} exceeds maximum {max_value}")
                
            # Add warnings for values close to limits
            if min_value is not None and value < min_value * 1.1:
                warnings.append(f"{field_name} {value} is close to minimum {min_value}")
                
            if max_value is not None and value > max_value * 0.9:
                warnings.append(f"{field_name} {value} is close to maximum {max_value}")
                
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                converted_value=value
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                converted_value=None
            )

    @staticmethod
    def validate_token_decimals(decimals: int) -> ValidationResult:
        """Validate token decimals"""
        errors = []
        warnings = []
        
        if not isinstance(decimals, int):
            errors.append("Decimals must be an integer")
            
        elif decimals < 0:
            errors.append("Decimals cannot be negative")
            
        elif decimals > 18:
            errors.append("Decimals cannot exceed 18")
            
        elif decimals == 0:
            warnings.append("Token has 0 decimals")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            converted_value=decimals if len(errors) == 0 else None
        )

    @staticmethod
    def validate_gas_price(
        gas_price: Union[int, Wei],
        network: str
    ) -> ValidationResult:
        """Validate gas price"""
        errors = []
        warnings = []
        
        try:
            gas_price_gwei = Wei(gas_price) / 10**9
            
            # Network-specific validation
            if network.lower() == 'mainnet':
                if gas_price_gwei < 1:
                    errors.append("Gas price too low for mainnet")
                elif gas_price_gwei > 500:
                    errors.append("Gas price extremely high")
                elif gas_price_gwei > 200:
                    warnings.append("Gas price is high")
                    
            elif network.lower() == 'base':
                if gas_price_gwei < 0.1:
                    errors.append("Gas price too low for Base")
                elif gas_price_gwei > 50:
                    errors.append("Gas price extremely high")
                elif gas_price_gwei > 20:
                    warnings.append("Gas price is high")
                    
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                converted_value=Wei(gas_price) if len(errors) == 0 else None
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Gas price validation error: {str(e)}"],
                warnings=[],
                converted_value=None
            )

    @staticmethod
    def validate_slippage(
        slippage: Union[float, Decimal],
        trade_type: str
    ) -> ValidationResult:
        """Validate slippage tolerance"""
        errors = []
        warnings = []
        
        try:
            slippage_float = float(slippage)
            
            if slippage_float < 0:
                errors.append("Slippage cannot be negative")
                
            elif slippage_float > 50:
                errors.append("Slippage tolerance too high (>50%)")
                
            elif trade_type.lower() == 'market' and slippage_float > 1:
                warnings.append("High slippage for market order")
                
            elif trade_type.lower() == 'limit' and slippage_float > 0.1:
                warnings.append("High slippage for limit order")
                
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                converted_value=Decimal(str(slippage_float)) if len(errors) == 0 else None
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Slippage validation error: {str(e)}"],
                warnings=[],
                converted_value=None
            )

def safe_convert(value: Any, target_type: Type[T], default: Optional[T] = None) -> T:
    """Safely convert value to target type with fallback"""
    try:
        if target_type == bool:
            return cast(T, TypeConverter.to_bool(value))
            
        elif target_type == int:
            return cast(T, TypeConverter.to_int(value))
            
        elif target_type == float:
            return cast(T, TypeConverter.to_float(value))
            
        elif target_type == Decimal:
            return cast(T, TypeConverter.to_decimal(value))
            
        elif target_type == ChecksumAddress:
            return cast(T, TypeConverter.to_checksum_address(value))
            
        elif target_type == datetime:
            if isinstance(value, (int, float)):
                return cast(T, datetime.fromtimestamp(float(value)))
            elif isinstance(value, str):
                return cast(T, datetime.fromisoformat(value))
                
        return target_type(value)
        
    except Exception as e:
        if default is not None:
            logger.warning(
                f"Conversion failed, using default: {str(e)}",
                extra={'value': value, 'target_type': target_type.__name__}
            )
            return default
        raise ConversionError(f"Could not convert {value} to {target_type.__name__}: {str(e)}")

# Example usage:
"""
# Safe type conversions
wei_amount = TypeConverter.to_wei('1.5', decimals=18)
eth_amount = TypeConverter.from_wei(wei_amount)
address = TypeConverter.to_checksum_address('0x123...')

# Data validation
gas_result = DataValidator.validate_gas_price(50000000000, 'mainnet')
if not gas_result.is_valid:
    print(f"Gas price errors: {gas_result.errors}")
    print(f"Gas price warnings: {gas_result.warnings}")

# Safe conversion with fallback
value = safe_convert("123.45", float, default=0.0)
""" 