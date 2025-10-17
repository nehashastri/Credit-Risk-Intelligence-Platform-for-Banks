import streamlit as st
from utils.gcp_connect import query_bq
from utils.charts import line_chart
import plotly.express as px

st.title("📰 News Sentiment Monitor")

sql = """
SELECT date, relevance_fed_policy, relevance_cpi, relevance_labor,
       relevance_markets, relevance_energy, relevance_real_estate
FROM `pipeline-882-team-project.fact_newspaper_outputs`
ORDER BY date
"""
df = query_bq(sql)

fig = px.area(df, x="date",
              y=["relevance_fed_policy","relevance_markets","relevance_labor"],
              title="Average News Relevance by Topic")
st.plotly_chart(fig, use_container_width=True)
