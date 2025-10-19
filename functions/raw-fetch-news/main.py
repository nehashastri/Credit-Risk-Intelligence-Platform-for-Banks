import os, json, datetime as dt
from typing import List, Dict

import functions_framework
import requests
from google.cloud import bigquery

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
RAW_DATASET    = os.getenv("RAW_DATASET")
RAW_TABLE      = os.getenv("RAW_TABLE")
BQ_TABLE       = f"{GCP_PROJECT_ID}.{RAW_DATASET}.{RAW_TABLE}"
TAVILY_KEY     = os.getenv("TAVILY_API_KEY")

DEFAULT_PRESETS = [
    {"topic":"fed_policy",  "query":"Federal Reserve FOMC rate decision OR Fed policy"},
    {"topic":"cpi",         "query":"US CPI inflation BLS report"},
    {"topic":"labor",       "query":"US unemployment labor market JOLTS NFP"},
    {"topic":"markets",     "query":"US stock market treasury yields credit spreads"},
    {"topic":"energy",      "query":"oil prices gasoline diesel energy market"},
    {"topic":"real_estate", "query":"US housing market mortgage delinquency"},
]

def _now_iso():    return dt.datetime.utcnow().isoformat()
def _today_iso():  return dt.date.today().isoformat()

def _safe_ts(val):
    if not val: return None
    try:
        s = str(val).replace("Z", "+00:00")
        return dt.datetime.fromisoformat(s).isoformat()
    except Exception:
        return None

def tavily_search(query:str, days:int=3, max_results:int=20):
    if not TAVILY_KEY:
        raise ValueError("Missing TAVILY_API_KEY")
    r = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {TAVILY_KEY}", "Content-Type":"application/json"},
        json={"query":query,"search_depth":"advanced","include_answers":False,
              "max_results":max_results,"days":days},
        timeout=30
    )
    r.raise_for_status()
    return r.json().get("results", [])

def normalize_rows(results:List[Dict], topic:str)->List[Dict]:
    now, today = _now_iso(), _today_iso()
    rows=[]
    for it in results:
        url = it.get("url")
        if not url: continue
        src = it.get("source") or (url.split("/")[2] if "://" in url else None)
        pub = it.get("published_date") or it.get("date")
        rows.append({
            "ingest_datetime": now,           # TIMESTAMP
            "ingest_date": today,             # DATE
            "topic": topic,                   # STRING
            "title": it.get("title"),         # STRING
            "url": url,                       # STRING
            "source_domain": src,             # STRING
            "published_at": _safe_ts(pub),    # TIMESTAMP
            "score": it.get("score"),         # FLOAT
            "snippet": it.get("content") or it.get("snippet"),  # STRING
            "raw_payload": it                 # JSON
        })
    return rows

def bq_merge(client: bigquery.Client, rows: List[Dict]) -> int:
    if not rows: return 0
    tmp = f"{GCP_PROJECT_ID}.{RAW_DATASET}._tmp_news_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    schema = [
        bigquery.SchemaField("ingest_datetime","TIMESTAMP"),
        bigquery.SchemaField("ingest_date","DATE"),
        bigquery.SchemaField("topic","STRING"),
        bigquery.SchemaField("title","STRING"),
        bigquery.SchemaField("url","STRING"),
        bigquery.SchemaField("source_domain","STRING"),
        bigquery.SchemaField("published_at","TIMESTAMP"),
        bigquery.SchemaField("score","FLOAT"),
        bigquery.SchemaField("snippet","STRING"),
        bigquery.SchemaField("raw_payload","JSON"),
    ]
    client.create_table(bigquery.Table(tmp, schema=schema))
    client.load_table_from_json(rows, tmp).result()

    merge_sql = f"""
    MERGE `{BQ_TABLE}` T
    USING `{tmp}` S
    ON T.url = S.url
       AND (T.published_at IS NOT DISTINCT FROM S.published_at)
    WHEN MATCHED THEN UPDATE SET
      ingest_datetime = S.ingest_datetime,
      ingest_date     = S.ingest_date,
      topic           = COALESCE(S.topic, T.topic),
      title           = COALESCE(S.title, T.title),
      source_domain   = COALESCE(S.source_domain, T.source_domain),
      score           = COALESCE(S.score, T.score),
      snippet         = COALESCE(S.snippet, T.snippet),
      raw_payload     = S.raw_payload
    WHEN NOT MATCHED BY TARGET THEN
      INSERT ROW
    """
    client.query(merge_sql).result()
    client.delete_table(tmp, not_found_ok=True)
    return len(rows)

@functions_framework.http
def raw_fetch_news(request):
    client = bigquery.Client(project=GCP_PROJECT_ID)
    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    presets = body.get("presets") or DEFAULT_PRESETS
    days = int(body.get("days", 3))
    max_results = int(body.get("max_results", 20))

    total = 0
    for p in presets:
        results = tavily_search(p["query"], days=days, max_results=max_results)
        rows = normalize_rows(results, p["topic"])
        total += bq_merge(client, rows)

    if total == 0:
        return ("no new rows", 204)
    return (json.dumps({"upserted": total}), 200)
