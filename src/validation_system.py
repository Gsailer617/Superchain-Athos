"""
Validation System for arbitrage parameters and transactions
"""

from typing import Dict, Tuple, Any, Callable, List

class ValidationSystem:
    def __init__(self, validation_rules: Dict[str, Callable], threshold_config: Dict[str, Any]):
        """
        Initialize validation system with rules and thresholds
        
        Args:
            validation_rules: Dict of rule name to validation function
            threshold_config: Dict of threshold configurations
        """
        self.validation_rules = validation_rules
        self.threshold_config = threshold_config

    async def validate_params(self, params: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate transaction parameters against defined rules
        
        Args:
            params: Parameters to validate
            context: Additional context for validation
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        for rule_name, rule_func in self.validation_rules.items():
            try:
                if not rule_func(params):
                    errors.append(f"Failed validation rule: {rule_name}")
            except Exception as e:
                errors.append(f"Error in validation rule {rule_name}: {str(e)}")
                
        return len(errors) == 0, errors 