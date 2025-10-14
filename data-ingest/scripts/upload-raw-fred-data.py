import os
import yaml
from google.cloud import bigquery

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/fred_series.yaml")

def load_to_bq(project_id, dataset_id, bucket_name, series_id):
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_id}.{series_id.lower()}"

    uri = f"gs://{bucket_name}/fred/raw/{series_id}/*.csv"

    job_config = bigquery.LoadJobConfig(
        autodetect=True,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )

    print(f"Loading {series_id} from {uri} → {table_id}")
    load_job = client.load_table_from_uri(uri, table_id, job_config=job_config)
    load_job.result()
    print(f"Loaded {load_job.output_rows} rows into {table_id}")

def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    project_id = config["project_id"]
    dataset_id = config["dataset_id"]
    bucket_name = config["bucket_name"]

    for s in config["series"]:
        load_to_bq(project_id, dataset_id, bucket_name, s)

if __name__ == "__main__":
    main()
