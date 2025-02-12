"""
Sentiment Analysis Module

This module provides NLP-based sentiment analysis for social media content
related to cryptocurrency tokens.
"""

from transformers import pipeline
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    score: float  # Overall sentiment score (-1 to 1)
    confidence: float  # Confidence in the sentiment score (0 to 1)
    sources: Dict[str, float]  # Individual scores by source

class SentimentAnalyzer:
    """
    Sentiment analysis for cryptocurrency-related social media content
    """
    
    def __init__(self):
        """Initialize sentiment analyzer with transformer model"""
        self._nlp = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self.metrics = MetricsManager()  # Initialize metrics
        self._init_nlp()
        
    def _init_nlp(self):
        """Initialize the NLP pipeline"""
        try:
            self._nlp = pipeline(
                "sentiment-analysis",
                model="finiteautomata/bertweet-base-sentiment-analysis",
                device=-1  # CPU
            )
        except Exception as e:
            logger.error(f"Failed to initialize NLP pipeline: {e}")
            raise
            
    async def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment score and confidence
        """
        if not text or not self._nlp:
            return {'score': 0.0, 'confidence': 0.0}
            
        try:
            start_time = time.time()
            # Run sentiment analysis in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._nlp,
                text
            )
            
            # Record inference duration
            duration = time.time() - start_time
            self.metrics.observe('llm_inference_duration_seconds', duration)
            
            # Convert result to normalized score
            label = result[0]['label']
            confidence = result[0]['score']
            
            score_map = {
                'POS': 1.0,
                'NEU': 0.0,
                'NEG': -1.0
            }
            
            # Record sentiment score
            score = score_map[label]
            self.metrics.observe('llm_token_sentiment', score)
            
            return {
                'score': score,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            self.metrics.increment('llm_errors_total', {'type': 'analysis_error'})
            return {'score': 0.0, 'confidence': 0.0}
            
    async def analyze_messages(
        self,
        messages: List[str],
        source: str
    ) -> Dict[str, float]:
        """
        Analyze sentiment of multiple messages
        
        Args:
            messages: List of messages to analyze
            source: Source of the messages (e.g. 'telegram', 'twitter')
            
        Returns:
            Aggregated sentiment score and confidence
        """
        if not messages:
            return {'score': 0.0, 'confidence': 0.0}
            
        results = []
        for msg in messages:
            result = await self.analyze_text(msg)
            results.append(result)
            
        # Aggregate results
        scores = [r['score'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        return {
            'score': float(np.mean(scores)),
            'confidence': float(np.mean(confidences))
        }
        
    async def get_token_sentiment(
        self,
        token_address: str,
        messages: Dict[str, List[str]]
    ) -> SentimentScore:
        """
        Get overall sentiment score for a token
        
        Args:
            token_address: Token address
            messages: Dictionary of messages by source
            
        Returns:
            Overall sentiment score with confidence
        """
        source_scores = {}
        
        for source, msgs in messages.items():
            result = await self.analyze_messages(msgs, source)
            source_scores[source] = result['score']
            
        if not source_scores:
            return SentimentScore(0.0, 0.0, {})
            
        # Calculate weighted average based on number of messages
        weights = {
            source: len(msgs)
            for source, msgs in messages.items()
        }
        total_weight = sum(weights.values())
        
        if total_weight == 0:
            return SentimentScore(0.0, 0.0, source_scores)
            
        weighted_score = sum(
            score * weights[source] / total_weight
            for source, score in source_scores.items()
        )
        
        # Calculate confidence based on message volume and consistency
        score_std = np.std(list(source_scores.values()))
        confidence = 1.0 - min(score_std, 1.0)  # Lower std = higher confidence
        
        return SentimentScore(
            score=float(weighted_score),
            confidence=float(confidence),
            sources=source_scores
        )
        
    def cleanup(self):
        """Clean up resources"""
        if self._executor:
            self._executor.shutdown() 