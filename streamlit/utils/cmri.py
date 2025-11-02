"""
Composite Macro Risk Index (CMRI) Module
Calculates and manages the composite risk index using macro, market, and news indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf

class CompositeMacroRiskIndex:
    """Handles calculation of Composite Macro Risk Index"""
    
    def __init__(self):
        """Initialize CMRI calculator with default parameters"""
        self.weights = {
            'macro': 0.4,      # 40% weight for macro indicators
            'market': 0.35,    # 35% weight for market indicators  
            'news': 0.25       # 25% weight for news sentiment
        }
        
        self.macro_indicators = [
            'fedfundrate',
            'unemployrate', 
            'cpiurban',
            'realgdp'
        ]
        
        self.market_etfs = [
            'SPY',   # S&P 500
            'XLF',   # Financial Sector
            'XLY',   # Consumer Discretionary
            'XLK',   # Technology
            'XLE',   # Energy
            'XLI',   # Industrial
            'XLRE'   # Real Estate
        ]
        
        self.news_topics = [
            'fed_policy',
            'cpi',
            'labor',
            'markets', 
            'energy',
            'real_estate',
            'layoff'
        ]
        
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
    
    def calculate_z_scores(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Calculate z-scores for specified columns
        
        Args:
            data: DataFrame with data
            columns: List of column names to calculate z-scores for
            
        Returns:
            DataFrame with z-scores added
        """
        data_copy = data.copy()
        
        for col in columns:
            if col in data_copy.columns:
                # Calculate rolling z-score with 30-day window
                rolling_mean = data_copy[col].rolling(window=30).mean()
                rolling_std = data_copy[col].rolling(window=30).std()
                data_copy[f'{col}_zscore'] = (data_copy[col] - rolling_mean) / rolling_std
                
                # Fill NaN values with 0
                data_copy[f'{col}_zscore'] = data_copy[f'{col}_zscore'].fillna(0)
        
        return data_copy
    
    def get_macro_risk_score(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate macro risk score from macro indicators
        
        Args:
            macro_data: DataFrame with macro indicators
            
        Returns:
            DataFrame with macro risk score
        """
        # Calculate z-scores for macro indicators
        macro_with_z = self.calculate_z_scores(macro_data, self.macro_indicators)
        
        # Calculate macro risk score (higher unemployment, higher CPI = higher risk)
        macro_with_z['macro_risk'] = (
            macro_with_z.get('unemployrate_zscore', 0) * 0.3 +
            macro_with_z.get('cpiurban_zscore', 0) * 0.3 +
            macro_with_z.get('fedfundrate_zscore', 0) * 0.2 +
            (-macro_with_z.get('realgdp_zscore', 0)) * 0.2  # Negative because higher GDP = lower risk
        )
        
        return macro_with_z
    
    def get_market_risk_score(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market risk score from ETF data
        
        Args:
            market_data: DataFrame with market ETF data
            
        Returns:
            DataFrame with market risk score
        """
        # Calculate z-scores for market indicators
        market_with_z = self.calculate_z_scores(market_data, self.market_etfs)
        
        # Calculate market volatility (risk proxy)
        market_with_z['market_volatility'] = market_with_z[self.market_etfs].std(axis=1)
        
        # Calculate market risk score based on volatility and returns
        market_with_z['market_risk'] = (
            market_with_z['market_volatility'] * 0.5 +
            (-market_with_z.get('SPY_zscore', 0)) * 0.3 +  # Negative because higher SPY = lower risk
            market_with_z.get('XLF_zscore', 0) * 0.2  # Financial sector as risk indicator
        )
        
        return market_with_z
    
    def get_news_risk_score(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate news risk score from sentiment data
        
        Args:
            news_data: DataFrame with news sentiment scores
            
        Returns:
            DataFrame with news risk score
        """
        # Pivot the data to have topics as columns
        news_pivot = news_data.pivot_table(
            index='date',
            columns='topic',
            values='avg_sentiment_score',
            aggfunc='mean'
        ).fillna(0)
        
        # Calculate z-scores for news topics
        news_with_z = self.calculate_z_scores(news_pivot, self.news_topics)
        
        # Calculate news risk score (negative sentiment = higher risk)
        news_with_z['news_risk'] = (
            (-news_with_z.get('fed_policy_zscore', 0)) * 0.25 +
            (-news_with_z.get('labor_zscore', 0)) * 0.2 +
            (-news_with_z.get('markets_zscore', 0)) * 0.2 +
            (-news_with_z.get('energy_zscore', 0)) * 0.15 +
            (-news_with_z.get('real_estate_zscore', 0)) * 0.1 +
            (-news_with_z.get('cpi_zscore', 0)) * 0.05 +
            news_with_z.get('layoff_zscore', 0) * 0.05  # Positive because layoffs = higher risk
        )
        
        return news_with_z
    
    def calculate_cmri(self, 
                      macro_data: pd.DataFrame,
                      market_data: pd.DataFrame, 
                      news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Composite Macro Risk Index
        
        Args:
            macro_data: DataFrame with macro indicators
            market_data: DataFrame with market data
            news_data: DataFrame with news sentiment data
            
        Returns:
            DataFrame with CMRI scores
        """
        # Calculate individual risk scores
        macro_risk = self.get_macro_risk_score(macro_data)
        market_risk = self.get_market_risk_score(market_data)
        news_risk = self.get_news_risk_score(news_data)
        
        # Merge all data on date
        combined_data = macro_risk.merge(
            market_risk[['date', 'market_risk']], 
            on='date', 
            how='outer'
        ).merge(
            news_risk[['date', 'news_risk']], 
            on='date', 
            how='outer'
        )
        
        # Fill missing values with 0
        combined_data['macro_risk'] = combined_data['macro_risk'].fillna(0)
        combined_data['market_risk'] = combined_data['market_risk'].fillna(0)
        combined_data['news_risk'] = combined_data['news_risk'].fillna(0)
        
        # Calculate CMRI
        combined_data['cmri'] = (
            combined_data['macro_risk'] * self.weights['macro'] +
            combined_data['market_risk'] * self.weights['market'] +
            combined_data['news_risk'] * self.weights['news']
        )
        
        # Normalize CMRI to 0-1 scale
        cmri_min = combined_data['cmri'].min()
        cmri_max = combined_data['cmri'].max()
        if cmri_max > cmri_min:
            combined_data['cmri_normalized'] = (combined_data['cmri'] - cmri_min) / (cmri_max - cmri_min)
        else:
            combined_data['cmri_normalized'] = 0.5
        
        # Add risk level classification
        combined_data['risk_level'] = combined_data['cmri_normalized'].apply(
            lambda x: 'Low' if x < self.risk_thresholds['low'] 
                     else 'Medium' if x < self.risk_thresholds['medium'] 
                     else 'High'
        )
        
        return combined_data
    
    def get_risk_alerts(self, cmri_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate risk alerts based on CMRI trends
        
        Args:
            cmri_data: DataFrame with CMRI scores
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        if len(cmri_data) < 2:
            return alerts
        
        # Get latest values
        latest = cmri_data.iloc[-1]
        previous = cmri_data.iloc[-2]
        
        # Alert for high risk level
        if latest['risk_level'] == 'High':
            alerts.append({
                'type': 'warning',
                'title': 'High Risk Alert',
                'message': f"CMRI is at {latest['cmri_normalized']:.2f} - High risk level detected",
                'severity': 'high'
            })
        
        # Alert for rapid increase
        cmri_change = latest['cmri_normalized'] - previous['cmri_normalized']
        if cmri_change > 0.1:
            alerts.append({
                'type': 'trend',
                'title': 'Rapid Risk Increase',
                'message': f"CMRI increased by {cmri_change:.2f} in the last period",
                'severity': 'medium'
            })
        
        # Alert for risk level change
        if latest['risk_level'] != previous['risk_level']:
            alerts.append({
                'type': 'level_change',
                'title': 'Risk Level Changed',
                'message': f"Risk level changed from {previous['risk_level']} to {latest['risk_level']}",
                'severity': 'medium'
            })
        
        return alerts
    
    def get_risk_summary(self, cmri_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for risk analysis
        
        Args:
            cmri_data: DataFrame with CMRI scores
            
        Returns:
            Dictionary with risk summary
        """
        if len(cmri_data) == 0:
            return {}
        
        latest = cmri_data.iloc[-1]
        
        # Calculate trends
        if len(cmri_data) >= 7:
            week_avg = cmri_data['cmri_normalized'].tail(7).mean()
            month_avg = cmri_data['cmri_normalized'].tail(30).mean() if len(cmri_data) >= 30 else week_avg
        else:
            week_avg = latest['cmri_normalized']
            month_avg = latest['cmri_normalized']
        
        return {
            'current_cmri': latest['cmri_normalized'],
            'current_risk_level': latest['risk_level'],
            'week_average': week_avg,
            'month_average': month_avg,
            'trend_direction': 'increasing' if latest['cmri_normalized'] > week_avg else 'decreasing',
            'macro_contribution': latest.get('macro_risk', 0) * self.weights['macro'],
            'market_contribution': latest.get('market_risk', 0) * self.weights['market'],
            'news_contribution': latest.get('news_risk', 0) * self.weights['news']
        }
    
    def fetch_yfinance_data(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Fetch market data from yfinance
        
        Args:
            symbols: List of ticker symbols
            period: Time period for data
            
        Returns:
            DataFrame with market data
        """
        try:
            data = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    
                    if not hist.empty:
                        # Calculate daily returns
                        hist['returns'] = hist['Close'].pct_change()
                        data[symbol] = hist['Close']
                        
                except Exception as e:
                    st.warning(f"Could not fetch data for {symbol}: {str(e)}")
                    continue
            
            if data:
                df = pd.DataFrame(data)
                df.index.name = 'date'
                df = df.reset_index()
                df['date'] = df['date']  # Keep date column
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error fetching yfinance data: {str(e)}")
            return pd.DataFrame()
