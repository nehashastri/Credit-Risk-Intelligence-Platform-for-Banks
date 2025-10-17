import streamlit as st
from utils.gcp_connect import query_bq
import pandas as pd

st.title("📂 Data Overview")

st.markdown("Summary of available datasets and latest update timestamps.")

tables = {
    "Macro Indicators":"fact_macro_indicators",
    "Market Sectors":"fact_market_sectors",
    "News Outputs":"fact_newspaper_outputs"
}

for name, table in tables.items():
    st.subheader(name)
    sql = f"SELECT MAX(date) AS latest_date, COUNT(*) AS rows FROM `pipeline-882-team-project.{table}`"
    df = query_bq(sql)
    st.dataframe(df)
st.success("Data check complete.")
