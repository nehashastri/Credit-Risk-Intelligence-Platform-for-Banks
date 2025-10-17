import streamlit as st
from utils.gcp_connect import query_bq
from utils.charts import line_chart

st.title("📊 Macro-Economic EDA")

sql = """
SELECT date, fedfunds, unrate, gdpc1, cpiaucsl, drcclacbs
FROM `pipeline-882-team-project.fact_macro_indicators`
ORDER BY date
"""
df = query_bq(sql)

st.plotly_chart(line_chart(df,"date","fedfunds","Federal Funds Rate (%)"), use_container_width=True)
st.plotly_chart(line_chart(df,"date","unrate","Unemployment Rate (%)"), use_container_width=True)
st.plotly_chart(line_chart(df,"date","drcclacbs","Credit Card Delinquency Rate (%)"), use_container_width=True)
