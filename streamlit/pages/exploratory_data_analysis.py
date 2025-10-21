"""
Exploratory Data Analysis Page
Comprehensive analysis combining macro, market, news, and risk data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gcp_connect import BigQueryConnector
from utils.charts import ChartBuilder

def show_exploratory_data_analysis():
    """Main function for the Exploratory Data Analysis page"""
    
    st.markdown("# 🔍 Exploratory Data Analysis")
    st.markdown("Comprehensive analysis of macro indicators, market performance, and news sentiment using all available data")
    
    # Initialize components
    bq_connector = BigQueryConnector()
    chart_builder = ChartBuilder()
    
    st.markdown("---")
    
    # 1. MACRO INDICATORS ANALYSIS
    st.markdown("## 📈 Macro Indicators Analysis")
    
    with st.spinner("Loading macro indicators data..."):
        try:
            # Load daily macro data (past 20 years)
            daily_query = """
            SELECT 
                date,
                marketyield2yr,
                marketyield10yr,
                inflationrate,
                mortgagerate30yr
            FROM `pipeline-882-team-project.landing.fact_macro_indicators_daily`
            WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 20 YEAR)
            ORDER BY date
            """
            
            daily_data = bq_connector.execute_query_to_dataframe(daily_query)
            
            # Load monthly macro data (past 20 years)
            monthly_query = """
            SELECT 
                date,
                fedfundrate,
                cpiurban,
                unemployrate,
                ownedconsumercredit,
                recessionindicator,
                realgdp
            FROM `pipeline-882-team-project.landing.fact_macro_indicators_monthly`
            WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 20 YEAR)
            ORDER BY date
            """
            
            monthly_data = bq_connector.execute_query_to_dataframe(monthly_query)
            
            if not daily_data.empty and not monthly_data.empty:
                st.success(f"✅ Loaded {len(daily_data)} daily and {len(monthly_data)} monthly macro records")
                
                # Convert date columns
                daily_data['date'] = pd.to_datetime(daily_data['date'])
                monthly_data['date'] = pd.to_datetime(monthly_data['date'])
                
                # Store in session state
                st.session_state['daily_macro_data'] = daily_data
                st.session_state['monthly_macro_data'] = monthly_data
                
                # Key Metrics
                st.markdown("### 📊 Key Macro Metrics")
                
                # Sort data to get latest values
                daily_sorted = daily_data.sort_values('date')
                monthly_sorted = monthly_data.sort_values('date')
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    latest_fed_rate = chart_builder.get_latest_value(monthly_sorted, 'fedfundrate')
                    st.metric(
                        "Fed Funds Rate",
                        f"{latest_fed_rate:.2f}%",
                        delta="Current"
                    )
                
                with col2:
                    latest_unemployment = chart_builder.get_latest_value(monthly_sorted, 'unemployrate')
                    st.metric(
                        "Unemployment Rate",
                        f"{latest_unemployment:.2f}%",
                        delta="Current"
                    )
                
                with col3:
                    latest_cpi = chart_builder.get_latest_value(monthly_sorted, 'cpiurban')
                    st.metric(
                        "CPI Urban",
                        f"{latest_cpi:.2f}%",
                        delta="Current"
                    )
                
                with col4:
                    latest_gdp = chart_builder.get_latest_value(monthly_sorted, 'realgdp')
                    st.metric(
                        "Real GDP",
                        f"${latest_gdp:,.0f}B",
                        delta="Current"
                    )
                
                # Macro Trends Visualization
                st.markdown("### 📈 Macro Trends")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Monthly indicators
                    fig = chart_builder.multi_line_chart(
                        monthly_data,
                        x='date',
                        y_columns=['fedfundrate', 'unemployrate', 'cpiurban'],
                        title="Monthly Macro Indicators (Past 20 Years)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Daily yield curves
                    fig = chart_builder.multi_line_chart(
                        daily_data,
                        x='date',
                        y_columns=['marketyield2yr', 'marketyield10yr', 'mortgagerate30yr'],
                        title="Daily Yield Curves (Past 20 Years)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Correlation Analysis
                st.markdown("### 🔗 Macro Correlations")
                
                # Calculate correlations for monthly data
                monthly_numeric = monthly_data.select_dtypes(include=[float, int])
                correlation_matrix = monthly_numeric.corr()
                
                fig = chart_builder.heatmap(
                    correlation_matrix,
                    title="Macro Indicators Correlation Matrix",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Rolling Averages
                st.markdown("### 📊 Rolling Averages")
                
                # Calculate rolling averages
                monthly_data_rolling = monthly_data.copy()
                monthly_data_rolling['fedfundrate_ma12'] = monthly_data_rolling['fedfundrate'].rolling(window=12).mean()
                monthly_data_rolling['unemployrate_ma12'] = monthly_data_rolling['unemployrate'].rolling(window=12).mean()
                monthly_data_rolling['cpiurban_ma12'] = monthly_data_rolling['cpiurban'].rolling(window=12).mean()
                
                fig = chart_builder.multi_line_chart(
                    monthly_data_rolling,
                    x='date',
                    y_columns=['fedfundrate', 'fedfundrate_ma12', 'unemployrate', 'unemployrate_ma12'],
                    title="12-Month Rolling Averages (Past 20 Years)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Decadal Analysis (from notebook EDA)
                st.markdown("### 📅 Decadal Analysis")
                st.markdown("Historical trends by decade showing key monetary policy periods")
                
                # Add decade column
                monthly_data_decade = monthly_data.copy()
                monthly_data_decade['year'] = monthly_data_decade['date'].dt.year
                monthly_data_decade['decade'] = (monthly_data_decade['year'] // 10) * 10
                
                # Calculate decade statistics
                decade_stats = monthly_data_decade.groupby('decade').agg({
                    'fedfundrate': ['mean', 'min', 'max', 'std'],
                    'unemployrate': ['mean', 'min', 'max', 'std'],
                    'cpiurban': ['mean', 'min', 'max', 'std']
                }).round(2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Fed Funds Rate by Decade")
                    fed_decade = monthly_data_decade.groupby('decade')['fedfundrate'].mean()
                    fig = px.bar(
                        x=fed_decade.index,
                        y=fed_decade.values,
                        title="Average Fed Funds Rate by Decade",
                        labels={'x': 'Decade', 'y': 'Average Rate (%)'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Decade Statistics")
                    st.dataframe(decade_stats, use_container_width=True)
                
            else:
                st.warning("⚠️ No macro data found for the selected date range")
                
        except Exception as e:
            st.error(f"❌ Error loading macro data: {str(e)}")
    
    st.markdown("---")
    
    # 2. MARKET ANALYSIS
    st.markdown("## 📊 Market Analysis")
    
    with st.spinner("Loading market data..."):
        try:
            # Load market data using yfinance
            import yfinance as yf
            
            # Define tickers
            tickers = ['SPY', 'XLF', 'XLY', 'XLK', 'XLE', 'XLI', 'XLRE']
            
            # Get market data (past 20 years)
            market_data = {}
            from datetime import datetime, timedelta
            twenty_years_ago = datetime.now() - timedelta(days=20*365)
            start_date = twenty_years_ago.strftime("%Y-%m-%d")
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date)  # Get past 20 years of data
                    if not hist.empty:
                        market_data[ticker] = hist['Close']
                except Exception as e:
                    st.warning(f"Could not load {ticker}: {str(e)}")
            
            if market_data:
                # Create DataFrame
                market_df = pd.DataFrame(market_data)
                market_df.index = pd.to_datetime(market_df.index)
                market_df = market_df.reset_index()
                market_df.rename(columns={'Date': 'date'}, inplace=True)
                
                st.success(f"✅ Loaded market data for {len(market_data)} tickers")
                
                # Market Performance Metrics
                st.markdown("### 📈 Market Performance")
                
                # Calculate returns
                market_returns = market_df.set_index('date').pct_change().dropna()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Total returns
                    total_returns = (market_df.set_index('date').iloc[-1] / market_df.set_index('date').iloc[0] - 1) * 100
                    st.markdown("#### Total Returns (%)")
                    for ticker, ret in total_returns.items():
                        st.metric(ticker, f"{ret:.2f}%")
                
                with col2:
                    # Volatility
                    volatility = market_returns.std() * (252 ** 0.5) * 100
                    st.markdown("#### Annualized Volatility (%)")
                    for ticker, vol in volatility.items():
                        st.metric(ticker, f"{vol:.2f}%")
                
                with col3:
                    # Sharpe ratio (assuming 2% risk-free rate)
                    sharpe_ratios = (market_returns.mean() * 252 - 0.02) / (market_returns.std() * (252 ** 0.5))
                    st.markdown("#### Sharpe Ratio")
                    for ticker, sharpe in sharpe_ratios.items():
                        st.metric(ticker, f"{sharpe:.2f}")
                
                # Market Trends
                st.markdown("### 📊 Market Trends")
                
                fig = chart_builder.multi_line_chart(
                    market_df,
                    x='date',
                    y_columns=tickers,
                    title="Market Performance Over Time (Past 20 Years)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Market Correlations
                st.markdown("### 🔗 Market Correlations")
                
                correlation_matrix = market_returns.corr()
                
                fig = chart_builder.heatmap(
                    correlation_matrix,
                    title="Market Correlation Matrix",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Weekly Correlations with Macro
                st.markdown("### 📈 Market-Macro Correlations")
                
                if not monthly_data.empty:
                    # Resample market data to monthly
                    market_monthly = market_df.set_index('date').resample('M').last()
                    market_monthly = market_monthly.reset_index()
                    
                    # Ensure both date columns are datetime64[ns] without timezone
                    market_monthly['date'] = pd.to_datetime(market_monthly['date']).dt.tz_localize(None)
                    monthly_data['date'] = pd.to_datetime(monthly_data['date']).dt.tz_localize(None)
                    
                    # Merge with macro data
                    merged_data = monthly_data.merge(
                        market_monthly,
                        on='date',
                        how='inner'
                    )
                    
                    if not merged_data.empty:
                        # Calculate correlations
                        macro_cols = ['fedfundrate', 'unemployrate', 'cpiurban', 'realgdp']
                        market_cols = tickers
                        
                        correlation_data = []
                        for macro_col in macro_cols:
                            for market_col in market_cols:
                                if macro_col in merged_data.columns and market_col in merged_data.columns:
                                    corr = merged_data[macro_col].corr(merged_data[market_col])
                                    correlation_data.append({
                                        'Macro Indicator': macro_col,
                                        'Market ETF': market_col,
                                        'Correlation': corr
                                    })
                        
                        if correlation_data:
                            corr_df = pd.DataFrame(correlation_data)
                            
                            # Display correlation matrix
                            pivot_corr = corr_df.pivot(
                                index='Macro Indicator',
                                columns='Market ETF',
                                values='Correlation'
                            )
                            
                            fig = chart_builder.heatmap(
                                pivot_corr,
                                title="Macro-Market Correlations",
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("⚠️ No market data could be loaded")
                
        except Exception as e:
            st.error(f"❌ Error loading market data: {str(e)}")
    
    st.markdown("---")
    
    # Footer note
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🔍 Exploratory Data Analysis Complete</p>
        <p><small>📊 Macro indicators, market performance, and historical trends analyzed</small></p>
    </div>
    """, unsafe_allow_html=True)
