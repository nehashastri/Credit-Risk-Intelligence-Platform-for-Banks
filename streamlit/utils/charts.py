"""
Chart Building Module
Provides reusable chart functions using Plotly for data visualization
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st

class ChartBuilder:
    """Handles creation of various chart types using Plotly"""
    
    def __init__(self):
        """Initialize chart builder with default styling"""
        self.color_palette = px.colors.qualitative.Set3
        self.default_colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        }
    
    def line_chart(self, 
                   data: pd.DataFrame, 
                   x: str, 
                   y: str, 
                   title: str = "",
                   color: Optional[str] = None,
                   height: int = 400) -> go.Figure:
        """
        Create a line chart
        
        Args:
            data: DataFrame with data
            x: Column name for x-axis
            y: Column name for y-axis
            title: Chart title
            color: Column name for color grouping
            height: Chart height
            
        Returns:
            Plotly figure object
        """
        fig = px.line(
            data, 
            x=x, 
            y=y, 
            title=title,
            color=color,
            height=height,
            template='plotly_white'
        )
        
        fig.update_layout(
            title_x=0.5,
            hovermode='x unified',
            showlegend=True if color else False
        )
        
        return fig
    
    def bar_chart(self, 
                  data: pd.DataFrame, 
                  x: str, 
                  y: str, 
                  title: str = "",
                  color: Optional[str] = None,
                  height: int = 400) -> go.Figure:
        """
        Create a bar chart
        
        Args:
            data: DataFrame with data
            x: Column name for x-axis
            y: Column name for y-axis
            title: Chart title
            color: Column name for color grouping
            height: Chart height
            
        Returns:
            Plotly figure object
        """
        fig = px.bar(
            data, 
            x=x, 
            y=y, 
            title=title,
            color=color,
            height=height,
            template='plotly_white'
        )
        
        fig.update_layout(
            title_x=0.5,
            hovermode='x unified'
        )
        
        return fig
    
    def scatter_chart(self, 
                     data: pd.DataFrame, 
                     x: str, 
                     y: str, 
                     title: str = "",
                     color: Optional[str] = None,
                     size: Optional[str] = None,
                     height: int = 400) -> go.Figure:
        """
        Create a scatter plot
        
        Args:
            data: DataFrame with data
            x: Column name for x-axis
            y: Column name for y-axis
            title: Chart title
            color: Column name for color grouping
            size: Column name for size mapping
            height: Chart height
            
        Returns:
            Plotly figure object
        """
        fig = px.scatter(
            data, 
            x=x, 
            y=y, 
            title=title,
            color=color,
            size=size,
            height=height,
            template='plotly_white'
        )
        
        fig.update_layout(
            title_x=0.5,
            hovermode='closest'
        )
        
        return fig
    
    def heatmap(self, 
                data: pd.DataFrame, 
                title: str = "",
                height: int = 400) -> go.Figure:
        """
        Create a correlation heatmap
        
        Args:
            data: DataFrame with numeric data
            title: Chart title
            height: Chart height
            
        Returns:
            Plotly figure object
        """
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            height=height,
            template='plotly_white'
        )
        
        return fig
    
    def multi_line_chart(self, 
                        data: pd.DataFrame, 
                        x: str, 
                        y_columns: List[str], 
                        title: str = "",
                        height: int = 500) -> go.Figure:
        """
        Create a multi-line chart with multiple y-axis variables
        
        Args:
            data: DataFrame with data
            x: Column name for x-axis
            y_columns: List of column names for y-axis
            title: Chart title
            height: Chart height
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        colors = self.color_palette[:len(y_columns)]
        
        for i, col in enumerate(y_columns):
            fig.add_trace(go.Scatter(
                x=data[x],
                y=data[col],
                mode='lines',
                name=col,
                line=dict(color=colors[i], width=2)
            ))
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            height=height,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def candlestick_chart(self, 
                         data: pd.DataFrame, 
                         title: str = "",
                         height: int = 500) -> go.Figure:
        """
        Create a candlestick chart for financial data
        
        Args:
            data: DataFrame with OHLC data
            title: Chart title
            height: Chart height
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=go.Candlestick(
            x=data['date'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="Price"
        ))
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            height=height,
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def gauge_chart(self, 
                   value: float, 
                   title: str = "",
                   min_val: float = 0, 
                   max_val: float = 100,
                   height: int = 300) -> go.Figure:
        """
        Create a gauge chart for risk indicators
        
        Args:
            value: Current value
            title: Chart title
            min_val: Minimum value
            max_val: Maximum value
            height: Chart height
            
        Returns:
            Plotly figure object
        """
        # Determine color based on value
        if value < (max_val - min_val) * 0.33:
            color = 'green'
        elif value < (max_val - min_val) * 0.66:
            color = 'orange'
        else:
            color = 'red'
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': (max_val - min_val) / 2},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': color},
                'steps': [
                    {'range': [min_val, (max_val - min_val) * 0.33], 'color': "lightgray"},
                    {'range': [(max_val - min_val) * 0.33, (max_val - min_val) * 0.66], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_val * 0.8
                }
            }
        ))
        
        fig.update_layout(height=height)
        
        return fig
    
    def subplot_charts(self, 
                      charts_config: List[Dict[str, Any]], 
                      rows: int, 
                      cols: int,
                      title: str = "",
                      height: int = 600) -> go.Figure:
        """
        Create subplot with multiple charts
        
        Args:
            charts_config: List of chart configurations
            rows: Number of rows
            cols: Number of columns
            title: Overall title
            height: Chart height
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=[config.get('title', '') for config in charts_config]
        )
        
        for i, config in enumerate(charts_config):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            chart_type = config.get('type', 'scatter')
            data = config.get('data')
            x = config.get('x')
            y = config.get('y')
            
            if chart_type == 'scatter':
                fig.add_trace(
                    go.Scatter(x=data[x], y=data[y], name=config.get('name', '')),
                    row=row, col=col
                )
            elif chart_type == 'bar':
                fig.add_trace(
                    go.Bar(x=data[x], y=data[y], name=config.get('name', '')),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            height=height,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def get_latest_value(self, data: pd.DataFrame, column: str, date_column: str = 'date') -> float:
        """
        Get the latest value for a column, ensuring proper date sorting
        
        Args:
            data: DataFrame with data
            column: Column name to get latest value from
            date_column: Date column name for sorting
            
        Returns:
            Latest value for the column
        """
        if data.empty or column not in data.columns:
            return 0
        
        # Ensure data is sorted by date
        if date_column in data.columns:
            data_sorted = data.sort_values(date_column)
        else:
            data_sorted = data
        
        return data_sorted[column].iloc[-1]
    
    def get_latest_values(self, data: pd.DataFrame, columns: list, date_column: str = 'date') -> dict:
        """
        Get latest values for multiple columns, ensuring proper date sorting
        
        Args:
            data: DataFrame with data
            columns: List of column names to get latest values from
            date_column: Date column name for sorting
            
        Returns:
            Dictionary with latest values for each column
        """
        if data.empty:
            return {col: 0 for col in columns}
        
        # Ensure data is sorted by date
        if date_column in data.columns:
            data_sorted = data.sort_values(date_column)
        else:
            data_sorted = data
        
        return {col: data_sorted[col].iloc[-1] if col in data_sorted.columns else 0 for col in columns}
    
    def rolling_average_chart(self, 
                            data: pd.DataFrame, 
                            x: str, 
                            y: str, 
                            window: int = 30,
                            title: str = "",
                            height: int = 400) -> go.Figure:
        """
        Create a chart with rolling average
        
        Args:
            data: DataFrame with data
            x: Column name for x-axis
            y: Column name for y-axis
            window: Rolling window size
            title: Chart title
            height: Chart height
            
        Returns:
            Plotly figure object
        """
        # Calculate rolling average
        data_sorted = data.sort_values(x)
        data_sorted[f'{y}_rolling'] = data_sorted[y].rolling(window=window).mean()
        
        fig = go.Figure()
        
        # Add original data
        fig.add_trace(go.Scatter(
            x=data_sorted[x],
            y=data_sorted[y],
            mode='lines',
            name=f'{y} (Original)',
            line=dict(color='lightblue', width=1),
            opacity=0.6
        ))
        
        # Add rolling average
        fig.add_trace(go.Scatter(
            x=data_sorted[x],
            y=data_sorted[f'{y}_rolling'],
            mode='lines',
            name=f'{y} ({window}-day avg)',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            height=height,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
