"""
Data Overview Page
Displays row counts, missing data, and timestamps from all landing tables
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.charts import ChartBuilder

def show_data_overview(bq_connector):
    """Display the data overview page"""
    
    st.markdown("# 📊 Data Overview")
    st.markdown("Comprehensive view of all landing tables in the BigQuery dataset")
    
    # Initialize chart builder
    chart_builder = ChartBuilder()
    
    # Get table information
    tables = [
        'fact_credit_outcomes',
        'fact_macro_indicators_daily',
        'fact_macro_indicators_monthly', 
        'news_articles',
        'fact_sector_prices_volumes'  # Added sector prices table
    ]
    
    # Load table information automatically
    with st.spinner("Loading table information..."):
        table_info_data = []
        
        for table in tables:
            try:
                if table == 'fact_sector_prices_volumes':
                    # Handle sector prices table differently - fetch from yfinance
                    sector_info = get_sector_prices_info()
                    table_info_data.append({
                        'Table': table,
                        'Row Count': f"{sector_info['row_count']:,}" if sector_info['row_count'] else 'Error',
                        'Size (MB)': f"{sector_info['size_mb']:.2f}" if sector_info['size_mb'] else 'Error',
                        'Latest Update': sector_info['latest_timestamp'].strftime('%Y-%m-%d') if sector_info['latest_timestamp'] else 'Error'
                    })
                else:
                    # Handle BigQuery tables normally
                    info = bq_connector.get_table_info(table)
                    table_info_data.append({
                        'Table': table,
                        'Row Count': f"{info['row_count']:,}" if info['row_count'] else 'Error',
                        'Size (MB)': f"{info['size_mb']:.2f}" if info['size_mb'] else 'Error',
                        'Latest Update': info['latest_timestamp'].strftime('%Y-%m-%d') if info['latest_timestamp'] else 'Error'
                    })
            except Exception as e:
                table_info_data.append({
                    'Table': table,
                    'Row Count': 'Error',
                    'Size (MB)': 'Error',
                    'Latest Update': 'Error'
                })
        
        # Display table information
        st.markdown("## 📋 Table Information")
        table_df = pd.DataFrame(table_info_data)
        st.dataframe(table_df, use_container_width=True)
    
    # Data quality metrics
    st.markdown("## 📈 Data Quality Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Tables Monitored",
            len(tables),
            delta="5 landing tables"
        )
    
    with col2:
        total_rows = sum([int(info['Row Count'].replace(',', '')) for info in table_info_data if info['Row Count'] != 'Error'])
        st.metric(
            "Total Records",
            f"{total_rows:,}",
            delta="All tables"
        )
    
    with col3:
        active_tables = len([
            info for info in table_info_data 
            if info['Row Count'] != 'Error' and int(info['Row Count'].replace(',', '')) > 0
        ])
        st.metric(
            "Active Tables",
            active_tables,
            delta=f"of {len(tables)} total"
        )
    
    with col4:
        # Calculate data completeness score
        completeness_scores = []
        for info in table_info_data:
            if info['Row Count'] != 'Error':
                try:
                    count = int(info['Row Count'].replace(',', ''))
                    completeness_scores.append(min(count / 1000, 1.0))  # Normalize to 0-1
                except:
                    completeness_scores.append(0)
        
        avg_completeness = sum(completeness_scores) / len(completeness_scores) * 100 if completeness_scores else 0
        
        st.metric(
            "Data Completeness",
            f"{avg_completeness:.1f}%",
            delta="Overall score"
        )
    
    # Macro Indicators Overview
    st.markdown("---")
    st.markdown("## 📈 Macro Indicators Overview")
    
    with st.spinner("Loading macro indicators overview..."):
        try:
            # Get latest macro indicators
            macro_query = """
            SELECT 
                'Daily' as frequency,
                COUNT(*) as record_count,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                AVG(marketyield2yr) as avg_2yr_yield,
                AVG(marketyield10yr) as avg_10yr_yield,
                AVG(inflationrate) as avg_inflation,
                AVG(mortgagerate30yr) as avg_mortgage_rate
            FROM `pipeline-882-team-project.landing.fact_macro_indicators_daily`
            
            UNION ALL
            
            SELECT 
                'Monthly' as frequency,
                COUNT(*) as record_count,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                AVG(fedfundrate) as avg_2yr_yield,
                AVG(unemployrate) as avg_10yr_yield,
                AVG(cpiurban) as avg_inflation,
                AVG(realgdp) as avg_mortgage_rate
            FROM `pipeline-882-team-project.landing.fact_macro_indicators_monthly`
            """
            
            macro_overview = bq_connector.execute_query_to_dataframe(macro_query)
            
            if not macro_overview.empty:
                st.success("✅ Loaded macro indicators overview")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Daily Indicators")
                    daily_data = macro_overview[macro_overview['frequency'] == 'Daily'].iloc[0]
                    
                    st.metric("Records", f"{daily_data['record_count']:,}")
                    st.metric("Date Range", f"{(daily_data['latest_date'] - daily_data['earliest_date']).days} days")
                    st.metric("Avg 2Y Yield", f"{daily_data['avg_2yr_yield']:.2f}%")
                    st.metric("Avg 10Y Yield", f"{daily_data['avg_10yr_yield']:.2f}%")
                
                with col2:
                    st.markdown("### 📊 Monthly Indicators")
                    monthly_data = macro_overview[macro_overview['frequency'] == 'Monthly'].iloc[0]
                    
                    st.metric("Records", f"{monthly_data['record_count']:,}")
                    st.metric("Date Range", f"{(monthly_data['latest_date'] - monthly_data['earliest_date']).days} days")
                    st.metric("Avg Fed Rate", f"{monthly_data['avg_2yr_yield']:.2f}%")
                    st.metric("Avg Unemployment", f"{monthly_data['avg_10yr_yield']:.2f}%")
                
                # Macro trends visualization
                st.markdown("### 📈 Macro Trends Overview")
                
                # Get recent data for visualization (past 20 years)
                recent_macro_query = """
                SELECT 
                    date,
                    'Daily' as type,
                    marketyield2yr as value,
                    '2Y Yield' as indicator
                FROM `pipeline-882-team-project.landing.fact_macro_indicators_daily`
                WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 20 YEAR)
                
                UNION ALL
                
                SELECT 
                    date,
                    'Monthly' as type,
                    fedfundrate as value,
                    'Fed Rate' as indicator
                FROM `pipeline-882-team-project.landing.fact_macro_indicators_monthly`
                WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 20 YEAR)
                
                ORDER BY date
                """
                
                recent_macro_data = bq_connector.execute_query_to_dataframe(recent_macro_query)
                
                if not recent_macro_data.empty:
                    recent_macro_data['date'] = pd.to_datetime(recent_macro_data['date'])
                    
                    fig = px.line(
                        recent_macro_data,
                        x='date',
                        y='value',
                        color='indicator',
                        title="Macro Indicators Trends (Past 20 Years)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("⚠️ No macro indicators data found")
                
        except Exception as e:
            st.error(f"❌ Error loading macro indicators overview: {str(e)}")
    
    # Sector Prices Overview (from yfinance)
    st.markdown("---")
    st.markdown("## 📈 Sector Prices Overview")
    st.markdown("Real-time sector ETF data fetched from yfinance")
    
    with st.spinner("Loading sector prices data..."):
        try:
            sector_data = get_sector_prices_data()
            
            if not sector_data.empty:
                st.success(f"✅ Loaded sector data for {len(sector_data)} trading days")
                
                # Display latest sector prices
                st.markdown("### 📊 Latest Sector Prices")
                latest_prices = sector_data.iloc[-1] if len(sector_data) > 0 else None
                
                if latest_prices is not None:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Financials (XLF)", f"${latest_prices.get('close_Financials', 0):.2f}")
                        st.metric("Technology (XLK)", f"${latest_prices.get('close_Technology', 0):.2f}")
                    
                    with col2:
                        st.metric("Energy (XLE)", f"${latest_prices.get('close_Energy', 0):.2f}")
                        st.metric("Health Care (XLV)", f"${latest_prices.get('close_HealthCare', 0):.2f}")
                    
                    with col3:
                        st.metric("Consumer Disc (XLY)", f"${latest_prices.get('close_ConsDisc', 0):.2f}")
                        st.metric("Industrials (XLI)", f"${latest_prices.get('close_Industrials', 0):.2f}")
                    
                    with col4:
                        st.metric("Materials (XLB)", f"${latest_prices.get('close_Materials', 0):.2f}")
                        st.metric("Utilities (XLU)", f"${latest_prices.get('close_Utilities', 0):.2f}")
                
                # Sector performance chart
                st.markdown("### 📈 Sector Performance Trends")
                
                # Get close price columns for visualization
                close_columns = [col for col in sector_data.columns if col.startswith('close_')]
                
                if close_columns:
                    fig = px.line(
                        sector_data,
                        x='date',
                        y=close_columns,
                        title="Sector ETF Close Prices Over Time (Past 20 Years)",
                        height=500
                    )
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Close Price ($)",
                        legend_title="Sector ETFs"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Volume analysis
                st.markdown("### 📊 Volume Analysis")
                
                # Get volume columns
                volume_columns = [col for col in sector_data.columns if col.startswith('vol_')]
                
                if volume_columns and len(sector_data) > 0:
                    # Calculate average volumes
                    avg_volumes = sector_data[volume_columns].mean()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Average Daily Volume")
                        for col in volume_columns:
                            sector_name = col.replace('vol_', '').replace('_', ' ')
                            st.metric(sector_name, f"{avg_volumes[col]:,.0f}")
                    
                    with col2:
                        # Volume trend chart
                        fig = px.line(
                            sector_data,
                            x='date',
                            y=volume_columns[:5],  # Show first 5 sectors to avoid clutter
                            title="Trading Volume Trends - Top 5 Sectors (Past 20 Years)",
                            height=400
                        )
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Volume",
                            legend_title="Sector ETFs"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("⚠️ No sector prices data could be loaded")
                
        except Exception as e:
            st.error(f"❌ Error loading sector prices data: {str(e)}")
    
    # Footer note
    st.markdown("---")
    st.info("""
    💡 **Data Overview Tips:**
    - All data is automatically loaded and analyzed
    - Monitor the Data Completeness score for overall health
    - Check Latest Timestamps to ensure data freshness
    """)

def get_sector_prices_data():
    """
    Get sector prices data from yfinance formatted according to the schema
    Returns DataFrame with columns matching fact_sector_prices_volumes table
    """
    try:
        import yfinance as yf
        from datetime import datetime
        
        # Define sector tickers according to the schema
        sector_tickers = {
            'XLY': 'ConsDisc',    # Consumer Discretionary
            'XLP': 'ConsStap',    # Consumer Staples  
            'XLF': 'Financials',  # Financials
            'XLK': 'Technology',   # Technology
            'XLE': 'Energy',       # Energy
            'XLI': 'Industrials',  # Industrials
            'XLU': 'Utilities',    # Utilities
            'XLV': 'HealthCare',   # Health Care
            'XLB': 'Materials',    # Materials
            'XLC': 'CommServ'      # Communication Services
        }
        
        # Initialize result DataFrame
        result_data = []
        
        # Get data for all sectors (past 20 years)
        sector_data = {}
        common_dates = None
        
        # Calculate date 20 years ago
        from datetime import datetime, timedelta
        twenty_years_ago = datetime.now() - timedelta(days=20*365)
        start_date = twenty_years_ago.strftime("%Y-%m-%d")
        
        for ticker, sector_name in sector_tickers.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date)
                if not hist.empty:
                    sector_data[sector_name] = hist
                    if common_dates is None:
                        common_dates = hist.index
                    else:
                        # Find common dates across all sectors
                        common_dates = common_dates.intersection(hist.index)
            except Exception as e:
                print(f"Error fetching {ticker}: {str(e)}")
                continue
        
        # Create DataFrame with common dates
        if common_dates is not None and len(common_dates) > 0:
            for date in common_dates:
                row = {'date': date}
                
                # Add close prices and volumes for each sector
                for sector_name, hist_data in sector_data.items():
                    if date in hist_data.index:
                        row[f'close_{sector_name}'] = hist_data.loc[date, 'Close']
                        row[f'vol_{sector_name}'] = int(hist_data.loc[date, 'Volume'])
                    else:
                        row[f'close_{sector_name}'] = None
                        row[f'vol_{sector_name}'] = None
                
                # Add ingest timestamp
                row['ingest_timestamp'] = datetime.now()
                
                result_data.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(result_data)
        
        # Sort by date
        if not df.empty:
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
        
    except ImportError:
        return pd.DataFrame()
    except Exception as e:
        print(f"Error in get_sector_prices_data: {str(e)}")
        return pd.DataFrame()

def get_sector_prices_info():
    """
    Get sector prices information from yfinance
    Returns table info similar to BigQuery table info format
    """
    try:
        import yfinance as yf
        from datetime import datetime
        
        # Define sector tickers according to the schema
        sector_tickers = {
            'XLY': 'ConsDisc',    # Consumer Discretionary
            'XLP': 'ConsStap',    # Consumer Staples  
            'XLF': 'Financials',  # Financials
            'XLK': 'Technology',   # Technology
            'XLE': 'Energy',       # Energy
            'XLI': 'Industrials',  # Industrials
            'XLU': 'Utilities',    # Utilities
            'XLV': 'HealthCare',   # Health Care
            'XLB': 'Materials',    # Materials
            'XLC': 'CommServ'      # Communication Services
        }
        
        # Get data for all sectors (past 20 years)
        total_rows = 0
        latest_date = None
        
        # Calculate date 20 years ago
        from datetime import datetime, timedelta
        twenty_years_ago = datetime.now() - timedelta(days=20*365)
        start_date = twenty_years_ago.strftime("%Y-%m-%d")
        
        for ticker in sector_tickers.keys():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date)
                if not hist.empty:
                    total_rows = len(hist)  # All tickers should have same number of rows
                    if latest_date is None or hist.index[-1] > latest_date:
                        latest_date = hist.index[-1]
            except Exception as e:
                print(f"Error fetching {ticker}: {str(e)}")
                continue
        
        # Estimate size (rough calculation)
        # Each row has 10 sectors * 2 fields (close + volume) = 20 float values
        # Plus date and timestamp = ~200 bytes per row
        estimated_size_mb = (total_rows * 200) / (1024 * 1024) if total_rows > 0 else 0
        
        return {
            'row_count': total_rows,
            'size_mb': estimated_size_mb,
            'latest_timestamp': latest_date
        }
        
    except ImportError:
        return {
            'row_count': 0,
            'size_mb': 0,
            'latest_timestamp': None
        }
    except Exception as e:
        print(f"Error in get_sector_prices_info: {str(e)}")
        return {
            'row_count': 0,
            'size_mb': 0,
            'latest_timestamp': None
        }
