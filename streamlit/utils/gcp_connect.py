from google.cloud import bigquery
import pandas as pd
import streamlit as st
import json

@st.cache_resource
def get_bq_client():
    key_dict = st.secrets["gcp_service_account"]
    client = bigquery.Client.from_service_account_info(dict(key_dict))
    return client

@st.cache_data(ttl=3600)
def query_bq(sql: str):
    client = get_bq_client()
    df = client.query(sql).to_dataframe()
    return df
