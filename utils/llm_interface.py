"""Centralized LLM interface for both agent and Telegram bot"""
import logging
import aiohttp
import json
from typing import Dict, Optional, Any, TypeVar, Union, List
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from transformers import pipeline
import torch
from dataclasses import dataclass

T = TypeVar('T')
PerformanceData = Dict[str, Union[int, float, str]]
MarketData = Dict[str, Union[int, float, str]]
TradingParams = Dict[str, Any]
SentimentResponse = Dict[str, Union[str, float, datetime]]

logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Store conversation context"""
    last_query: str
    last_response: str
    timestamp: float
    topic: str
    user_id: int
    message_history: List[Dict[str, str]]

class LLMInterface:
    """Unified interface for LLM interactions"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.api_url = "https://api-inference.huggingface.co/models/"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.summary_model = "facebook/bart-large-cnn"
        self.analysis_model = "bigscience/bloom"
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = 300  # 5 minutes
        
        # Add new models for specific tasks
        self.opportunity_model = "facebook/bart-large-cnn"
        self.market_model = "bigscience/bloom"
        
        # Initialize NLP capabilities
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis"
        )
        
        # Conversation management
        self.conversations: Dict[int, ConversationContext] = {}
        
        # Intent classification categories
        self.intents = {
            'performance': ['profit', 'gains', 'losses', 'stats', 'performance'],
            'market': ['price', 'volume', 'liquidity', 'market', 'trend'],
            'trading': ['trade', 'buy', 'sell', 'position', 'strategy'],
            'system': ['status', 'health', 'settings', 'configuration'],
            'help': ['help', 'guide', 'tutorial', 'commands', 'instructions']
        }
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_summary(self, performance_data: PerformanceData) -> str:
        """Generate performance summary using BART"""
        try:
            prompt = self._create_summary_prompt(performance_data)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}{self.summary_model}",
                    headers=self.headers,
                    json={"inputs": prompt, "parameters": {"max_length": 150, "min_length": 50}}
                ) as response:
                    result = await response.json()
                    
            if isinstance(result, list) and len(result) > 0:
                return result[0]['summary_text']
            return "Error generating summary"
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating summary"
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_trading_suggestion(
        self, 
        suggestion: str, 
        current_params: TradingParams
    ) -> Dict[str, Any]:
        """Analyze trading suggestion using BLOOM"""
        try:
            prompt = self._create_trading_prompt(suggestion, current_params)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}{self.analysis_model}",
                    headers=self.headers,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_length": 200,
                            "temperature": 0.3,
                            "return_full_text": False
                        }
                    }
                ) as response:
                    result = await response.json()
                    
            return self._parse_trading_response(result[0]['generated_text'])
            
        except Exception as e:
            logger.error(f"Error analyzing trading suggestion: {e}")
            return {}
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_market_sentiment(self, market_data: MarketData) -> SentimentResponse:
        """Analyze market sentiment and conditions"""
        try:
            prompt = self._create_market_prompt(market_data)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}{self.analysis_model}",
                    headers=self.headers,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_length": 150,
                            "temperature": 0.3
                        }
                    }
                ) as response:
                    result = await response.json()
                    
            return self._parse_sentiment_response(result[0]['generated_text'])
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0, 'timestamp': datetime.now()}
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_opportunity(self, opportunity_data: Dict[str, Any]) -> str:
        """Generate detailed analysis of arbitrage opportunity"""
        try:
            prompt = f"""Analyze this arbitrage opportunity:
            
            Type: {opportunity_data.get('type', 'Unknown')}
            Expected Profit: ${opportunity_data.get('expected_profit', 0):.2f}
            Confidence: {opportunity_data.get('confidence', 0)*100:.1f}%
            Risk Score: {opportunity_data.get('risk_score', 0)*100:.1f}%
            Gas Cost: ${opportunity_data.get('gas_cost', 0):.2f}
            Net Profit: ${opportunity_data.get('net_profit', 0):.2f}
            Tokens: {' → '.join(opportunity_data.get('tokens_involved', []))}
            
            Provide a concise analysis of:
            1. Opportunity quality and risk factors
            2. Market conditions impact
            3. Execution recommendations
            4. Risk mitigation strategies"""
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}{self.opportunity_model}",
                    headers=self.headers,
                    json={"inputs": prompt, "parameters": {"max_length": 200, "min_length": 50}}
                ) as response:
                    result = await response.json()
                    
            return result[0]['summary_text'] if isinstance(result, list) and len(result) > 0 else "Error analyzing opportunity"
            
        except Exception as e:
            logger.error(f"Error analyzing opportunity: {e}")
            return "Error analyzing opportunity"
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_market_insight(self, market_data: Dict[str, Any]) -> str:
        """Generate market insight based on current conditions"""
        try:
            prompt = f"""Analyze these market conditions:
            
            Gas Price: {market_data.get('gas_price', 0)} gwei
            Network Load: {market_data.get('network_load', 0)}%
            Recent Volatility: {market_data.get('volatility', 0):.2f}%
            Trading Volume: {market_data.get('volume_24h', 0):.2f} ETH
            DEX Performance: {market_data.get('dex_metrics', {})}
            Token Metrics: {market_data.get('token_metrics', {})}
            
            Provide insights on:
            1. Current market state
            2. Trading opportunities
            3. Risk factors
            4. Recommended strategies"""
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}{self.market_model}",
                    headers=self.headers,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_length": 200,
                            "temperature": 0.3
                        }
                    }
                ) as response:
                    result = await response.json()
                    
            return result[0]['generated_text'] if isinstance(result, list) and len(result) > 0 else "Error generating market insight"
            
        except Exception as e:
            logger.error(f"Error generating market insight: {e}")
            return "Error generating market insight"
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_performance_metrics(self, performance_data: Dict[str, Any]) -> str:
        """Analyze performance metrics with detailed insights"""
        try:
            # Extract key metrics
            total_trades = performance_data.get('total_trades', 0)
            success_rate = performance_data.get('success_rate', 0)
            total_profit = performance_data.get('total_profit', 0)
            
            # Generate natural language analysis
            analysis = (
                f"📊 Performance Analysis:\n\n"
                f"Based on the data, we've executed {total_trades} trades with a "
                f"success rate of {success_rate:.1f}%. The total profit stands at "
                f"${total_profit:.2f}.\n\n"
            )
            
            if 'recent_trades' in performance_data:
                analysis += "Recent Trading Activity:\n"
                for trade in performance_data['recent_trades'][:3]:
                    analysis += f"• {trade['description']}\n"
                    
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing performance metrics: {e}")
            return "I apologize, but I couldn't analyze the performance metrics at this time."

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_market_conditions(self, market_data: Dict) -> str:
        """Analyze current market conditions with insights"""
        try:
            # Extract market metrics
            volatility = market_data.get('volatility', 0)
            volume = market_data.get('volume_24h', 0)
            trend = market_data.get('trend', 'neutral')
            
            # Generate market analysis
            analysis = (
                f"🌐 Market Analysis:\n\n"
                f"The market is currently showing {trend} momentum with "
                f"{volatility:.1f}% volatility. 24-hour volume is "
                f"${volume:,.0f}.\n\n"
            )
            
            if 'opportunities' in market_data:
                analysis += "Notable Opportunities:\n"
                for opp in market_data['opportunities'][:3]:
                    analysis += f"• {opp['description']}\n"
                    
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return "I apologize, but I couldn't analyze the market conditions at this time."

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_trading_opportunities(self, query: str) -> str:
        """Analyze and explain trading opportunities"""
        try:
            # Extract specific tokens or trading pairs from query
            # This is a placeholder - implement actual token extraction logic
            tokens = self._extract_tokens_from_query(query)
            
            if tokens:
                return (
                    f"💡 Trading Analysis for {', '.join(tokens)}:\n\n"
                    f"I'm analyzing potential opportunities... "
                    f"[Implement actual trading analysis here]"
                )
            else:
                return (
                    "I can help analyze trading opportunities. "
                    "Please specify which tokens or trading pairs you're interested in."
                )
                
        except Exception as e:
            logger.error(f"Error analyzing trading opportunities: {e}")
            return "I apologize, but I couldn't analyze trading opportunities at this time."

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_system_status(self) -> str:
        """Get current system status"""
        try:
            # Implement actual system status checks here
            return (
                "🔧 System Status:\n\n"
                "• Bot: Active\n"
                "• Network: Connected\n"
                "• Performance: Optimal\n"
                "• Last Update: Just now"
            )
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return "I apologize, but I couldn't retrieve the system status at this time."

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def explain_error(self, error_data: Dict[str, Any]) -> str:
        """Generate explanation and recovery suggestions for errors"""
        try:
            prompt = f"""Analyze this error scenario:
            
            Error Type: {error_data.get('error_type', 'Unknown')}
            Message: {error_data.get('message', '')}
            Context: {error_data.get('context', '')}
            Stack Trace: {error_data.get('stack_trace', '')}
            
            Provide:
            1. Error explanation in simple terms
            2. Potential causes
            3. Recovery suggestions
            4. Prevention recommendations"""
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}{self.analysis_model}",
                    headers=self.headers,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_length": 200,
                            "temperature": 0.3
                        }
                    }
                ) as response:
                    result = await response.json()
                    
            return result[0]['generated_text'] if isinstance(result, list) and len(result) > 0 else "Error explaining error"
            
        except Exception as e:
            logger.error(f"Error generating error explanation: {e}")
            return "Error explaining error"
            
    def _create_summary_prompt(self, data: Dict) -> str:
        """Enhanced summary prompt creation"""
        return f"""Summarize the following trading performance:
        Total Trades: {data.get('total_trades', 0)}
        Success Rate: {data.get('success_rate', 0):.2f}%
        Total Profit: {data.get('total_profit', 0):.4f} ETH
        Average Gas Cost: {data.get('avg_gas_cost', 0):.4f} ETH
        Best Trade: {data.get('best_trade', 'N/A')}
        Market Conditions: {data.get('market_conditions', 'N/A')}
        Recent Performance: {data.get('recent_performance', 'N/A')}
        Token Analytics: {data.get('token_analytics', {})}
        DEX Analytics: {data.get('dex_analytics', {})}
        
        Focus on:
        1. Key metrics and trends
        2. Performance insights
        3. Market impact analysis
        4. Strategy recommendations
        5. Risk factors and mitigation
        6. Improvement opportunities"""
        
    def _create_trading_prompt(self, suggestion: str, current_params: Dict) -> str:
        """Create trading suggestion analysis prompt"""
        return f"""Given the following trading bot parameters and user suggestion, provide parameter adjustments in JSON format.
        
        Current Parameters:
        {current_params}
        
        User Suggestion: {suggestion}
        
        Respond with a JSON object containing only the parameters that should be adjusted."""
        
    def _create_market_prompt(self, market_data: Dict) -> str:
        """Create market analysis prompt"""
        return f"""Analyze the following market conditions:
        Gas Price: {market_data.get('gas_price', 0)} gwei
        Network Load: {market_data.get('network_load', 0)}%
        Recent Volatility: {market_data.get('volatility', 0):.2f}%
        Trading Volume: {market_data.get('volume_24h', 0):.2f} ETH
        
        Provide sentiment analysis and market insights."""
        
    def _parse_trading_response(self, response: str) -> Dict:
        """Parse trading suggestion response"""
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            return {}
        except Exception as e:
            logger.error(f"Error parsing trading response: {e}")
            return {}
            
    def _parse_sentiment_response(self, response: str) -> Dict:
        """Parse sentiment analysis response"""
        try:
            lines = response.strip().split('\n')
            sentiment = 'neutral'
            confidence = 0.5
            
            for line in lines:
                if 'sentiment:' in line.lower():
                    sentiment = line.split(':')[1].strip().lower()
                elif 'confidence:' in line.lower():
                    try:
                        confidence = float(line.split(':')[1].strip().replace('%', '')) / 100
                    except ValueError:
                        pass
                        
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error parsing sentiment response: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0}
            
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False
            
        cache_time = self.cache[key].get('timestamp')
        if not cache_time:
            return False
            
        cache_age = (datetime.now() - datetime.fromisoformat(cache_time)).total_seconds()
        return cache_age < self.cache_duration
        
    def _cache_data(self, key: str, data: Dict):
        """Cache data with timestamp"""
        self.cache[key] = {
            **data,
            'timestamp': datetime.now().isoformat()
        } 

    async def process_query(self, query: str, user_id: int) -> str:
        """Process natural language query and maintain conversation context"""
        try:
            # Update conversation context
            self._update_conversation_context(query, user_id)
            
            # Classify intent
            intent = self._classify_intent(query)
            
            # Generate response based on intent
            response = await self._generate_response(query, intent, user_id)
            
            # Update conversation with response
            self._store_response(response, user_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I apologize, but I encountered an error processing your request. Please try again."

    def _update_conversation_context(self, query: str, user_id: int) -> None:
        """Update conversation context for user"""
        if user_id not in self.conversations:
            self.conversations[user_id] = ConversationContext(
                last_query="",
                last_response="",
                timestamp=datetime.now().timestamp(),
                topic="general",
                user_id=user_id,
                message_history=[]
            )
        
        context = self.conversations[user_id]
        context.last_query = query
        context.timestamp = datetime.now().timestamp()
        context.message_history.append({"role": "user", "content": query})
        
        # Keep conversation history manageable
        if len(context.message_history) > 10:
            context.message_history = context.message_history[-10:]

    def _classify_intent(self, query: str) -> str:
        """Classify user intent from query"""
        query_lower = query.lower()
        
        # Check each intent category
        for intent, keywords in self.intents.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
                
        return 'general'

    async def _generate_response(self, query: str, intent: str, user_id: int) -> str:
        """Generate contextual response based on intent"""
        context = self.conversations.get(user_id)
        
        if intent == 'performance':
            return await self.analyze_performance_metrics(self._get_performance_data())
            
        elif intent == 'market':
            return await self.analyze_market_conditions(self._get_market_data())
            
        elif intent == 'trading':
            return await self.analyze_trading_opportunities(query)
            
        elif intent == 'system':
            return await self.get_system_status()
            
        elif intent == 'help':
            return self._get_help_response(query)
            
        else:
            # General conversation
            return await self._generate_general_response(query, context)

    def _store_response(self, response: str, user_id: int) -> None:
        """Store bot response in conversation context"""
        if user_id in self.conversations:
            context = self.conversations[user_id]
            context.last_response = response
            context.message_history.append({"role": "assistant", "content": response})

    def _get_help_response(self, query: str) -> str:
        """Get contextual help response"""
        query_lower = query.lower()
        
        if 'trade' in query_lower or 'trading' in query_lower:
            return (
                "🤖 Trading Commands Help:\n\n"
                "• /performance - View trading performance\n"
                "• /status - Check current status\n"
                "• /analyze <token> - Analyze specific token\n"
                "• /market - View market conditions"
            )
        elif 'settings' in query_lower or 'configure' in query_lower:
            return (
                "⚙️ Settings Commands:\n\n"
                "• /settings - View current settings\n"
                "• /thresholds - View trading thresholds\n"
                "• /mode - Switch between auto/manual mode"
            )
        else:
            return (
                "👋 Hello! I'm your arbitrage trading assistant. Here are some things I can help with:\n\n"
                "• Trading analysis and opportunities\n"
                "• Market conditions and trends\n"
                "• Performance metrics and insights\n"
                "• System status and settings\n\n"
                "Just ask me what you'd like to know!"
            )

    async def _generate_general_response(self, query: str, context: Optional[ConversationContext]) -> str:
        """Generate response for general queries using conversation context"""
        try:
            if not context:
                return "I'm here to help! What would you like to know about trading or market conditions?"
                
            # Use conversation history for context
            history = context.message_history
            
            # Generate contextual response
            # This is where you'd integrate with a more sophisticated LLM if available
            return (
                "I understand you're asking about that. "
                "Would you like to know about trading performance, "
                "market conditions, or something else specific?"
            )
            
        except Exception as e:
            logger.error(f"Error generating general response: {e}")
            return "I'm here to help! What would you like to know?"

    def _extract_tokens_from_query(self, query: str) -> List[str]:
        """Extract token symbols or addresses from query"""
        # Implement token extraction logic
        # This is a placeholder
        return []

    def _get_performance_data(self) -> Dict:
        """Get current performance data"""
        # Implement actual performance data retrieval
        return {
            'total_trades': 0,
            'success_rate': 0,
            'total_profit': 0,
            'recent_trades': []
        }

    def _get_market_data(self) -> Dict:
        """Get current market data"""
        # Implement actual market data retrieval
        return {
            'volatility': 0,
            'volume_24h': 0,
            'trend': 'neutral',
            'opportunities': []
        } 