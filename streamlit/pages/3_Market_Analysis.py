import streamlit as st
from utils.gcp_connect import query_bq
from utils.charts import line_chart

st.title("💹 Market Sector Analysis")

sql = """
SELECT date, Financials_Close, Technology_Close, Energy_Close
FROM `pipeline-882-team-project.fact_market_sectors`
ORDER BY date
"""
df = query_bq(sql)

st.plotly_chart(line_chart(df,"date","Financials_Close","Financial Sector ETF (XLF)"), use_container_width=True)
st.plotly_chart(line_chart(df,"date","Technology_Close","Technology Sector ETF (XLK)"), use_container_width=True)
st.plotly_chart(line_chart(df,"date","Energy_Close","Energy Sector ETF (XLE)"), use_container_width=True)
