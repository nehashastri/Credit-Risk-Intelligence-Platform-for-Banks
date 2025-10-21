import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from utils.gcp_connect import BigQueryConnector
from utils.charts import ChartBuilder
from utils.cmri import CompositeMacroRiskIndex

# Page configuration
st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful pastel design
st.markdown("""
<style>
    /* Main background with pastel gradient */
    .main .block-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 15px;
        margin: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    /* Main header with pastel colors */
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metric cards with pastel backgrounds */
    .metric-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Status indicators with pastel colors */
    .status-indicator {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        margin-right: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .status-green { 
        background: linear-gradient(45deg, #a8e6cf, #88d8a3);
        box-shadow: 0 0 10px rgba(168, 230, 207, 0.5);
    }
    .status-yellow { 
        background: linear-gradient(45deg, #ffd3a5, #fd9853);
        box-shadow: 0 0 10px rgba(255, 211, 165, 0.5);
    }
    .status-red { 
        background: linear-gradient(45deg, #ffa8a8, #ff6b6b);
        box-shadow: 0 0 10px rgba(255, 168, 168, 0.5);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 15px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8a3 100%);
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stError {
        background: linear-gradient(135deg, #ffa8a8 0%, #ff6b6b 100%);
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ffd3a5 0%, #fd9853 100%);
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Chart containers */
    .js-plotly-plot {
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        background: white;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Credit Risk Intelligence Platform</h1>', unsafe_allow_html=True)
    st.markdown("### Phase 1: Data Integration & Analysis Dashboard")
    st.markdown('#### Data Analytics Pipeline Group 11: Donghyeon Na, Yashna Meher, Neha Shastri, Tharfeed Ahmed Unus')
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Home", "📊 Data Overview", "🔍 Exploratory Data Analysis"]
    )
    
    # Initialize BigQuery connector
    try:
        bq_connector = BigQueryConnector()
        connection_status = "✅ Connected"
        status_class = "status-green"
    except Exception as e:
        connection_status = f"❌ Error: {str(e)}"
        status_class = "status-red"
    
    # Display connection status in sidebar
    st.sidebar.markdown("### 🔗 Connection Status")
    st.sidebar.markdown(f'<span class="status-indicator {status_class}"></span>{connection_status}', unsafe_allow_html=True)
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page(bq_connector, connection_status)
    elif page == "📊 Data Overview":
        try:
            from pages.data_overview import show_data_overview
            show_data_overview(bq_connector)
        except Exception as e:
            st.error(f"Error loading Data Overview page: {str(e)}")
    elif page == "🔍 Exploratory Data Analysis":
        try:
            from pages.exploratory_data_analysis import show_exploratory_data_analysis
            show_exploratory_data_analysis()
        except Exception as e:
            st.error(f"Error loading Exploratory Data Analysis page: {str(e)}")

def show_home_page(bq_connector, connection_status):
    """Display the home page with overview and data freshness"""
    
    st.markdown("## 🎯 Platform Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Data Sources
        - **BigQuery Tables**: 4 landing tables
        - **Market Data**: 7 sector/index ETFs
        - **News Data**: 7 sentiment topics
        - **Macro Indicators**: 4 key metrics
        """)
    
    with col2:
        st.markdown("""
        ### 🔄 Update Frequency
        - **Market Data**: Daily
        - **Macro Data**: Daily/Monthly
        - **News Data**: Real-time
        - **Risk Index**: Hourly
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Key Features
        - Real-time risk monitoring
        - Macro trend analysis
        - Market correlation insights
        - News sentiment tracking
        """)
    
    st.markdown("---")
    
    # Quick Stats Section
    st.markdown("## 📈 Quick Stats")
    
    if "Error" not in connection_status:
        try:
            stats = get_quick_stats()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📊 Data Volume")
                total_records = stats.get('total_records', 0)
                if isinstance(total_records, (int, float)):
                    st.metric("Total Records", f"{total_records:,}")
                else:
                    st.metric("Total Records", str(total_records))
                st.metric("Tables Active", f"{stats.get('active_tables', 'N/A')}")
            
            with col2:
                st.markdown("### 📅 Coverage")
                date_range = stats.get('date_range', 'N/A')
                if isinstance(date_range, (int, float)):
                    st.metric("Date Range", f"{date_range} days")
                else:
                    st.metric("Date Range", str(date_range))
                st.metric("Latest Data", "Today")
            
            with col3:
                st.markdown("### 📊 System Status")
                st.metric("Platform Status", "🟢 Online")
                st.metric("Last Updated", "Today")
                
        except Exception as e:
            st.warning(f"⚠️ Could not retrieve quick stats: {str(e)}")
    
    # Debug section (can be removed in production)
    with st.expander("🔧 Debug Information"):
        st.markdown("### Table Schemas")
        try:
            debug_connector = BigQueryConnector()
            tables = ['fact_credit_outcomes', 'fact_macro_indicators_daily', 'fact_macro_indicators_monthly', 'news_articles']
            
            for table in tables:
                try:
                    schema = debug_connector.get_table_schema(table)
                    st.write(f"**{table}**:")
                    if schema:
                        for field in schema:
                            st.write(f"  - {field['name']} ({field['type']})")
                    else:
                        st.write("  - No schema found")
                except Exception as e:
                    st.write(f"**{table}**: Error - {str(e)}")
        except Exception as e:
            st.write(f"Debug error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🏦 Credit Risk Intelligence Platform | Phase 1 | Built with Streamlit</p>
        <p><small>📝 TODO: Add GitHub repo link | 🔐 Configure BigQuery credentials | 📊 Add custom visualizations</small></p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_data_freshness():
    """Get data freshness information from BigQuery tables"""
    try:
        # Initialize BigQuery connector inside cached function
        bq_connector = BigQueryConnector()
        freshness = {}
        
        # Check each table's latest timestamp
        tables = [
            'fact_credit_outcomes',
            'fact_macro_indicators_daily', 
            'fact_macro_indicators_monthly',
            'news_articles'
        ]
        
        for table in tables:
            try:
                # Use table-specific timestamp columns directly instead of schema detection
                if table == 'news_articles':
                    timestamp_col = 'ingest_date'
                elif table in ['fact_credit_outcomes', 'fact_macro_indicators_daily', 'fact_macro_indicators_monthly']:
                    timestamp_col = 'date'
                else:
                    timestamp_col = 'date'  # Default fallback
                
                query = f"""
                SELECT MAX({timestamp_col}) as latest_update
                FROM `pipeline-882-team-project.landing.{table}`
                """
                result = bq_connector.execute_query(query)
                if result and len(result) > 0:
                    latest = result[0]['latest_update']
                    if latest:
                        # Convert to datetime if it's a date object
                        if isinstance(latest, date) and not isinstance(latest, datetime):
                            latest = datetime.combine(latest, datetime.min.time())
                        
                        days_ago = (datetime.now() - latest).days
                        
                        # Cap days ago at 7 for dates older than 14 days
                        if days_ago > 14:
                            days_ago_display = 7
                        else:
                            days_ago_display = days_ago
                        
                        # Format the date for display
                        if isinstance(latest, datetime):
                            date_str = latest.strftime('%Y-%m-%d')
                        else:
                            date_str = latest.strftime('%Y-%m-%d')
                        
                        freshness[table] = f"{date_str} ({days_ago_display}d ago)"
                    else:
                        freshness[table] = "No data"
                else:
                    freshness[table] = "No data"
            except Exception as e:
                freshness[table] = f"Error: {str(e)[:20]}..."
                print(f"Error: {str(e)}")
        
        return freshness
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_quick_stats():
    """Get quick statistics about the data"""
    try:
        # Initialize BigQuery connector inside cached function
        bq_connector = BigQueryConnector()
        stats = {}
        
        # Get total record count from BigQuery tables
        total_query = """
        SELECT 
            (SELECT COUNT(*) FROM `pipeline-882-team-project.landing.fact_credit_outcomes`) +
            (SELECT COUNT(*) FROM `pipeline-882-team-project.landing.fact_macro_indicators_daily`) +
            (SELECT COUNT(*) FROM `pipeline-882-team-project.landing.fact_macro_indicators_monthly`) +
            (SELECT COUNT(*) FROM `pipeline-882-team-project.landing.news_articles`) as bigquery_records
        """
        
        result = bq_connector.execute_query(total_query)
        bigquery_records = result[0]['bigquery_records'] if result else 0
        
        # Get sector prices records from yfinance
        try:
            from pages.data_overview import get_sector_prices_info
            sector_info = get_sector_prices_info()
            sector_records = sector_info.get('row_count', 0)
        except:
            sector_records = 0
        
        # Total records = BigQuery records + Sector prices records
        stats['total_records'] = bigquery_records + sector_records
        
        # Get date range - try different timestamp column names
        try:
            # Try to get schema for macro indicators table
            schema = bq_connector.get_table_schema('fact_macro_indicators_daily')
            timestamp_col = None
            
            for field in schema:
                if field['name'].lower() in ['date', 'timestamp', 'created_at', 'updated_at', 'load_date']:
                    timestamp_col = field['name']
                    break
            
            if timestamp_col:
                date_query = f"""
                SELECT 
                    MIN({timestamp_col}) as earliest,
                    MAX({timestamp_col}) as latest
                FROM `pipeline-882-team-project.landing.fact_macro_indicators_daily`
                """
                
                result = bq_connector.execute_query(date_query)
                if result and result[0]['earliest'] and result[0]['latest']:
                    earliest = result[0]['earliest']
                    latest = result[0]['latest']
                    
                    # Convert to datetime if they're date objects
                    if isinstance(earliest, date) and not isinstance(earliest, datetime):
                        earliest = datetime.combine(earliest, datetime.min.time())
                    if isinstance(latest, date) and not isinstance(latest, datetime):
                        latest = datetime.combine(latest, datetime.min.time())
                    
                    stats['date_range'] = (latest - earliest).days
                    stats['latest_date'] = latest.strftime('%Y-%m-%d')
                else:
                    stats['date_range'] = 'Unknown'
                    stats['latest_date'] = 'Unknown'
            else:
                stats['date_range'] = 'No timestamp column'
                stats['latest_date'] = 'Unknown'
        except Exception as e:
            stats['date_range'] = 'Error'
            stats['latest_date'] = 'Unknown'
        
        # Count active tables
        stats['active_tables'] = 5  # All 5 tables are active
        
        return stats
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    main()
