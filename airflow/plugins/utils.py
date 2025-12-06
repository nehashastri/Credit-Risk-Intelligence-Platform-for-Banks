from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
import pandas as pd


def run_sql(query: str, conn_id: str = "google_cloud_connection"):
    """
    Executes a SQL query and returns the result as a list of tuples.
    Equivalent to hook.get_records(), but wrapped for convenience.
    """
    hook = BigQueryHook(gcp_conn_id=conn_id, use_legacy_sql=False)
    client = hook.get_client()
    job = client.query(query)
    result = job.result()
    return [tuple(row.values()) for row in result]


def run_sql_df(query: str, conn_id: str = "google_cloud_connection") -> pd.DataFrame:
    """
    Executes SQL and returns a pandas DataFrame.
    """
    hook = BigQueryHook(gcp_conn_id=conn_id, use_legacy_sql=False)
    return hook.get_pandas_df(query)


def run_execute(query: str, conn_id: str = "google_cloud_connection"):
    """
    Executes SQL that does not return rows (e.g. CREATE, INSERT, DELETE).
    """
    hook = BigQueryHook(gcp_conn_id=conn_id, use_legacy_sql=False)
    client = hook.get_client()
    job = client.query(query)
    job.result()  # Wait for execution
    return True
