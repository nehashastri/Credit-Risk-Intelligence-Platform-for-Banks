import streamlit as st
from utils.gcp_connect import query_bq
from utils.cmri import compute_cmri
from utils.charts import line_chart
import plotly.express as px

st.title("⚠️ Risk Dashboard – Composite Macro Risk Index")

sql = """
SELECT date, unrate, fedfunds, cpiaucsl, t5yie, drcclacbs
FROM `pipeline-882-team-project.fact_macro_indicators`
ORDER BY date
"""
df = query_bq(sql)
df = compute_cmri(df)

st.metric("Current CMRI", round(df["cmri"].iloc[-1],2))
fig = px.line(df, x="date", y="cmri", color="regime_label",
              title="Composite Macro Risk Index (CMRI)")
st.plotly_chart(fig, use_container_width=True)

st.caption("High CMRI = macro stress environment (e.g., 2008, 2020).")
