import os, json, datetime as dt, email.utils as eut, time
from typing import List, Dict

import functions_framework
import requests
from google.cloud import bigquery

# NEW: YAML & (optional) GCS loading
import yaml
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

try:
    from google.cloud import storage
except Exception:
    storage = None

# Load .env during local development; in production, use --set-env-vars
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Environment Variables ----------
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
RAW_DATASET    = os.getenv("RAW_DATASET")
RAW_TABLE      = os.getenv("RAW_TABLE")
BQ_TABLE       = f"{GCP_PROJECT_ID}.{RAW_DATASET}.{RAW_TABLE}"
TAVILY_KEY     = os.getenv("TAVILY_API_KEY")

# YAML path (optional): used as a fallback when body.presets is not provided
QUERIES_YAML_PATH = os.getenv("QUERIES_YAML_PATH")      # e.g. /workspace/include/queries.yaml
QUERIES_YAML_GCS  = os.getenv("QUERIES_YAML_GCS")       # e.g. gs://my-bucket/include/queries.yaml

# Default parameters (fallback to 3 / 20 if not set)
DEFAULT_DAYS = int(os.getenv("RAW_DAYS", 3))
DEFAULT_MAX  = int(os.getenv("RAW_MAX_RESULTS", 20))

# ---------- Default Presets (final fallback) ----------
DEFAULT_PRESETS = [
    {"topic": "fed_policy",  "query": "Federal Reserve FOMC rate decision OR Fed policy"},
    {"topic": "cpi",         "query": "US CPI inflation BLS report"},
    {"topic": "labor",       "query": "US unemployment labor market JOLTS NFP"},
    {"topic": "markets",     "query": "US stock market treasury yields credit spreads"},
    {"topic": "energy",      "query": "oil prices gasoline diesel energy market"},
    {"topic": "real_estate", "query": "US housing market mortgage delinquency"},
]

# ---------- Utility Functions (same as before) ----------
PUBLISH_KEYS = [
    "published_at","published_date","published_time",
    "date_published","date","pubDate","pub_date",
    "published","publishedAt"
]

def _now_iso()   -> str: return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
def _today_iso() -> str: return dt.date.today().isoformat()

def _parse_any_ts(s: str):
    """Attempt to parse multiple timestamp formats safely."""
    if not s:
        return None
    s = str(s).strip()
    try:
        return dt.datetime.fromisoformat(s.replace("Z","+00:00")).astimezone(dt.timezone.utc).isoformat()
    except Exception:
        pass
    try:
        return eut.parsedate_to_datetime(s).astimezone(dt.timezone.utc).isoformat()
    except Exception:
        return None

def _extract_published_from_item(it: Dict):
    """Extract publication time from item fields or metadata."""
    for k in PUBLISH_KEYS:
        if k in it and it[k]:
            ts = _parse_any_ts(it[k])
            if ts:
                return ts
    meta = it.get("meta") or it.get("metadata") or {}
    if isinstance(meta, dict):
        for k in PUBLISH_KEYS:
            if k in meta and meta[k]:
                ts = _parse_any_ts(meta[k])
                if ts:
                    return ts
    return None

def _last_modified_from_head(url: str, timeout=5):
    """Fallback: fetch 'Last-Modified' header from URL."""
    try:
        h = requests.head(url, allow_redirects=True, timeout=timeout)
        lm = h.headers.get("Last-Modified")
        return _parse_any_ts(lm)
    except Exception:
        return None

def _safe_published(it: Dict, url: str, fallback_ts: str):
    """Safely determine the publication timestamp with multiple fallback options."""
    ts = _extract_published_from_item(it)
    if ts:
        return ts
    ts = _last_modified_from_head(url)
    if ts:
        return ts
    return fallback_ts

# --- (Optional) Load YAML from GCS or local ---
def _load_yaml_local(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _load_yaml_gcs(uri: str):
    """Download YAML configuration directly from GCS."""
    if not storage:
        raise RuntimeError("google-cloud-storage not available")
    if not uri.startswith("gs://"):
        raise ValueError("GCS URI must start with gs://")
    _, rest = uri.split("gs://", 1)
    bucket_name, blob_path = rest.split("/", 1)
    client = storage.Client(project=GCP_PROJECT_ID)
    blob = client.bucket(bucket_name).blob(blob_path)
    text = blob.download_as_text()
    return yaml.safe_load(text) or {}

def _load_presets_from_yaml_fallback():
    """Load presets from local or GCS YAML, with environment overrides and defaults."""
    data = {}
    if QUERIES_YAML_PATH:
        try:
            data = _load_yaml_local(QUERIES_YAML_PATH)
        except Exception as e:
            print(f"[WARN] Local YAML load failed: {e}")
    if (not data) and QUERIES_YAML_GCS:
        try:
            data = _load_yaml_gcs(QUERIES_YAML_GCS)
        except Exception as e:
            print(f"[WARN] GCS YAML load failed: {e}")

    if not data:
        return [], {"days": DEFAULT_DAYS, "max_results": DEFAULT_MAX}

    defaults = data.get("defaults", {}) or {}
    presets  = data.get("presets", []) or []

    # Apply environment overrides to defaults
    d_days = int(defaults.get("days", DEFAULT_DAYS))
    d_max  = int(defaults.get("max_results", DEFAULT_MAX))
    defaults_merged = {"days": d_days, "max_results": d_max}

    merged_presets = []
    for p in presets:
        merged_presets.append({
            "topic": p["topic"],
            "query": p["query"],
            "days": int(p.get("days", d_days)),
            "max_results": int(p.get("max_results", d_max)),
        })
    return merged_presets, defaults_merged

# ---------- Tavily Search ----------
def tavily_search(query: str, days: int, max_results: int):
    """Perform search via Tavily API."""
    if not TAVILY_KEY:
        raise ValueError("Missing TAVILY_API_KEY")

    start_time = time.time()
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {TAVILY_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "search_depth": "advanced",
                "include_answers": False,
                "max_results": max_results,
                "days": days,
            },
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
        elapsed = time.time() - start_time
        print(f"[tavily_search] query='{query}' days={days} max={max_results} took {elapsed:.2f}s, results={len(data.get('results', []))}")
        return data.get("results", [])
    except Exception as e:
        print(f"[ERROR][tavily_search] query='{query}' failed: {str(e)}")
        return []

# ---------- Normalization ----------
def normalize_rows(results: List[Dict], topic: str) -> List[Dict]:
    """Normalize Tavily API results into a structured BigQuery row format."""
    now, today = _now_iso(), _today_iso()
    rows = []
    for it in results:
        url = it.get("url")
        if not url:
            continue
        src = it.get("source") or (url.split("/")[2] if "://" in url else None)
        published_ts = _safe_published(it, url, fallback_ts=now)
        rows.append({
            "ingest_datetime": now,
            "ingest_date": today,
            "topic": topic,
            "title": it.get("title"),
            "url": url,
            "source_domain": src,
            "published_at": published_ts,
            "score": it.get("score"),
            "snippet": it.get("content") or it.get("snippet"),
            "raw_payload": it
        })
    return rows

# ---------- BigQuery MERGE Upsert (same as before) ----------
def bq_merge(client: bigquery.Client, rows: List[Dict]) -> int:
    """Merge rows into BigQuery table, ensuring deduplication by URL and timestamp."""
    if not rows:
        return 0

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

# ---------- HTTP Endpoint ----------
@functions_framework.http
def raw_fetch_news(request):
    """Main entrypoint for Cloud Function: fetches news via Tavily API and loads into BigQuery."""
    client = bigquery.Client(project=GCP_PROJECT_ID)

    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    # 1) Priority: use request body.presets if provided
    presets = body.get("presets")
    defaults = body.get("defaults") or {}
    d_days = int(defaults.get("days", DEFAULT_DAYS))
    d_max  = int(defaults.get("max_results", DEFAULT_MAX))

    # 2) If not present, attempt to load from YAML (local or GCS)
    if not presets:
        presets, yaml_defaults = _load_presets_from_yaml_fallback()
        if yaml_defaults:
            d_days = int(yaml_defaults.get("days", d_days))
            d_max  = int(yaml_defaults.get("max_results", d_max))

    # 3) If still missing, use hardcoded fallback presets
    if not presets:
        presets = [{"topic": p["topic"], "query": p["query"], "days": d_days, "max_results": d_max}
                   for p in DEFAULT_PRESETS]

    total = 0
    for p in presets:
        topic = p["topic"]
        query = p["query"]
        days = int(p.get("days", d_days))
        max_results = int(p.get("max_results", d_max))

        print(f"[raw_fetch_news] Fetching topic='{topic}' days={days} max={max_results} ...")
        results = tavily_search(query, days=days, max_results=max_results)
        rows = normalize_rows(results, topic)
        upserted = bq_merge(client, rows)
        total += upserted
        print(f"[raw_fetch_news] Topic '{topic}' upserted {upserted} rows")

    # Always return 200 even if no new data
    return (json.dumps({"upserted": total, "defaults": {"days": d_days, "max_results": d_max}}), 200)