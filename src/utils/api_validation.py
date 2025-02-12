"""
API response validation and schema checking utilities
"""

from typing import Dict, Any, List, Optional, Union, Type, TypeVar, Callable
from dataclasses import dataclass
import json
import logging
from datetime import datetime
from .type_safety import TypeConverter, ValidationResult, ValidationError
from jsonschema import validate, ValidationError as JsonSchemaError
import aiohttp
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class ApiResponse:
    """Validated API response"""
    status_code: int
    data: Any
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    response_time: float
    timestamp: datetime

class ApiMetrics:
    """API metrics tracking"""
    
    def __init__(self):
        self.request_counter = Counter(
            'api_requests_total',
            'Total number of API requests',
            ['endpoint', 'method', 'status']
        )
        self.response_time = Histogram(
            'api_response_time_seconds',
            'API response time in seconds',
            ['endpoint', 'method']
        )
        self.validation_errors = Counter(
            'api_validation_errors_total',
            'Total number of API validation errors',
            ['endpoint', 'error_type']
        )

class ApiValidator:
    """API response validation utilities"""
    
    def __init__(self):
        self.metrics = ApiMetrics()
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.custom_validators: Dict[str, Callable] = {}
        
    def register_schema(self, endpoint: str, schema: Dict[str, Any]) -> None:
        """Register JSON schema for endpoint"""
        self.schemas[endpoint] = schema
        
    def register_validator(
        self,
        endpoint: str,
        validator: Callable[[Any], ValidationResult]
    ) -> None:
        """Register custom validator for endpoint"""
        self.custom_validators[endpoint] = validator
        
    async def validate_response(
        self,
        endpoint: str,
        response: aiohttp.ClientResponse,
        expected_type: Optional[Type[T]] = None
    ) -> ApiResponse:
        """Validate API response"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            # Record metrics
            self.metrics.request_counter.labels(
                endpoint=endpoint,
                method=response.method,
                status=response.status
            ).inc()
            
            # Check status code
            if response.status != 200:
                errors.append(f"Unexpected status code: {response.status}")
                return ApiResponse(
                    status_code=response.status,
                    data=None,
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    response_time=(datetime.now() - start_time).total_seconds(),
                    timestamp=datetime.now()
                )
                
            # Parse response
            try:
                data = await response.json()
            except Exception as e:
                errors.append(f"Failed to parse JSON response: {str(e)}")
                return ApiResponse(
                    status_code=response.status,
                    data=None,
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    response_time=(datetime.now() - start_time).total_seconds(),
                    timestamp=datetime.now()
                )
                
            # Validate against schema
            if endpoint in self.schemas:
                try:
                    validate(instance=data, schema=self.schemas[endpoint])
                except JsonSchemaError as e:
                    errors.append(f"Schema validation failed: {str(e)}")
                    self.metrics.validation_errors.labels(
                        endpoint=endpoint,
                        error_type='schema'
                    ).inc()
                    
            # Apply custom validation
            if endpoint in self.custom_validators:
                try:
                    result = self.custom_validators[endpoint](data)
                    errors.extend(result.errors)
                    warnings.extend(result.warnings)
                    if not result.is_valid:
                        self.metrics.validation_errors.labels(
                            endpoint=endpoint,
                            error_type='custom'
                        ).inc()
                except Exception as e:
                    errors.append(f"Custom validation failed: {str(e)}")
                    self.metrics.validation_errors.labels(
                        endpoint=endpoint,
                        error_type='custom_error'
                    ).inc()
                    
            # Convert to expected type
            if expected_type and not errors:
                try:
                    if expected_type == dict:
                        if not isinstance(data, dict):
                            errors.append(f"Expected dict, got {type(data)}")
                    elif expected_type == list:
                        if not isinstance(data, list):
                            errors.append(f"Expected list, got {type(data)}")
                    else:
                        data = TypeConverter.safe_convert(data, expected_type)
                except Exception as e:
                    errors.append(f"Type conversion failed: {str(e)}")
                    self.metrics.validation_errors.labels(
                        endpoint=endpoint,
                        error_type='type_conversion'
                    ).inc()
                    
            # Record response time
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.response_time.labels(
                endpoint=endpoint,
                method=response.method
            ).observe(duration)
            
            return ApiResponse(
                status_code=response.status,
                data=data,
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                response_time=duration,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(
                f"API validation error for {endpoint}: {str(e)}",
                exc_info=True
            )
            return ApiResponse(
                status_code=response.status if response else 500,
                data=None,
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=warnings,
                response_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now()
            )

class ApiSchemas:
    """Common API response schemas"""
    
    TOKEN_PRICE = {
        "type": "object",
        "properties": {
            "price": {"type": "number"},
            "timestamp": {"type": "string"},
            "currency": {"type": "string"}
        },
        "required": ["price", "timestamp"]
    }
    
    DEX_PAIR = {
        "type": "object",
        "properties": {
            "token0": {"type": "string"},
            "token1": {"type": "string"},
            "reserve0": {"type": "string"},
            "reserve1": {"type": "string"},
            "fee": {"type": "number"}
        },
        "required": ["token0", "token1", "reserve0", "reserve1"]
    }
    
    TRADE_QUOTE = {
        "type": "object",
        "properties": {
            "input_token": {"type": "string"},
            "output_token": {"type": "string"},
            "input_amount": {"type": "string"},
            "output_amount": {"type": "string"},
            "price_impact": {"type": "number"},
            "gas_estimate": {"type": "number"}
        },
        "required": [
            "input_token",
            "output_token",
            "input_amount",
            "output_amount"
        ]
    }
    
    TOKEN_INFO = {
        "type": "object",
        "properties": {
            "address": {"type": "string"},
            "name": {"type": "string"},
            "symbol": {"type": "string"},
            "decimals": {"type": "integer"},
            "total_supply": {"type": "string"},
            "verified": {"type": "boolean"}
        },
        "required": ["address", "decimals"]
    }

class ApiValidators:
    """Common API response validators"""
    
    @staticmethod
    def validate_price_data(data: Dict[str, Any]) -> ValidationResult:
        """Validate price data"""
        errors = []
        warnings = []
        
        try:
            # Check price
            price = float(data.get('price', 0))
            if price <= 0:
                errors.append("Price must be positive")
            elif price < 0.000001:
                warnings.append("Extremely low price")
                
            # Check timestamp
            timestamp = data.get('timestamp')
            if timestamp:
                try:
                    ts = datetime.fromisoformat(timestamp)
                    age = (datetime.now() - ts).total_seconds()
                    if age > 3600:
                        warnings.append("Price data is over 1 hour old")
                except ValueError:
                    errors.append("Invalid timestamp format")
                    
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                converted_value=data if len(errors) == 0 else None
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Price validation error: {str(e)}"],
                warnings=[],
                converted_value=None
            )
            
    @staticmethod
    def validate_reserves(data: Dict[str, Any]) -> ValidationResult:
        """Validate DEX reserves data"""
        errors = []
        warnings = []
        
        try:
            # Check reserves
            reserve0 = int(data.get('reserve0', 0))
            reserve1 = int(data.get('reserve1', 0))
            
            if reserve0 <= 0 or reserve1 <= 0:
                errors.append("Reserves must be positive")
                
            # Check for imbalanced reserves
            ratio = reserve0 / reserve1 if reserve1 > 0 else float('inf')
            if ratio > 1000 or ratio < 0.001:
                warnings.append("Highly imbalanced reserves")
                
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                converted_value=data if len(errors) == 0 else None
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Reserves validation error: {str(e)}"],
                warnings=[],
                converted_value=None
            )
            
    @staticmethod
    def validate_trade_quote(data: Dict[str, Any]) -> ValidationResult:
        """Validate trade quote data"""
        errors = []
        warnings = []
        
        try:
            # Check amounts
            input_amount = int(data.get('input_amount', 0))
            output_amount = int(data.get('output_amount', 0))
            
            if input_amount <= 0 or output_amount <= 0:
                errors.append("Amounts must be positive")
                
            # Check price impact
            price_impact = float(data.get('price_impact', 0))
            if price_impact > 10:
                errors.append("Price impact too high (>10%)")
            elif price_impact > 3:
                warnings.append("High price impact (>3%)")
                
            # Check gas estimate
            gas_estimate = int(data.get('gas_estimate', 0))
            if gas_estimate > 1000000:
                errors.append("Gas estimate too high")
            elif gas_estimate > 500000:
                warnings.append("High gas estimate")
                
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                converted_value=data if len(errors) == 0 else None
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Quote validation error: {str(e)}"],
                warnings=[],
                converted_value=None
            )

# Example usage:
"""
# Initialize validator
validator = ApiValidator()

# Register schemas and validators
validator.register_schema('price', ApiSchemas.TOKEN_PRICE)
validator.register_validator('price', ApiValidators.validate_price_data)

# Validate response
async with aiohttp.ClientSession() as session:
    async with session.get('https://api.example.com/price') as response:
        result = await validator.validate_response('price', response, dict)
        if not result.is_valid:
            print(f"Validation errors: {result.errors}")
            print(f"Validation warnings: {result.warnings}")
        else:
            price_data = result.data
""" 