import streamlit as st

st.set_page_config(page_title="Credit Risk Intelligence Platform", layout="wide")

st.title("💳 Credit Risk Intelligence Platform (Phase 1)")
st.markdown("""
Welcome to the **Credit Risk Intelligence Platform**, built on live data from:
- **Federal Reserve (FRED)** – macroeconomic indicators  
- **Yahoo Finance** – market sector benchmarks  
- **Tavily API** – news relevance & sentiment signals  

This Phase 1 prototype provides real-time monitoring of macro, market, and news factors
that influence credit risk for banks.

Use the sidebar to navigate:
- 📂 Data Overview  
- 📊 Macro EDA  
- 💹 Market Analysis  
- 📰 News Sentiment  
- ⚠️ Risk Dashboard  
""")

st.info("✅ Connected to GCP BigQuery for live data feeds.")
