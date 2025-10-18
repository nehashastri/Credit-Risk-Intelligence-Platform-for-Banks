# file: explore_tavily_news.py
# Purpose:
# - Collect sample news data using Tavily (travily) API to inspect available fields
# - Perform basic profiling to understand data structure before designing BigQuery ERD
# - Output sample CSV and suggested table schema (for staging + normalized ERD)

import os, time, json, hashlib, urllib.parse
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
API_KEY   = os.getenv("TAVILY_API_KEY", "")
PROXY_URL = os.getenv("TRAVILY_PROXY_URL", "").strip()
USE_PROXY = bool(PROXY_URL)

# ---------- Tavily client initialization ----------
if not USE_PROXY:
    # Direct mode (local testing)
    if not API_KEY:
        raise RuntimeError("TAVILY_API_KEY is required when not using proxy.")
    from tavily import TavilyClient
    tavily = TavilyClient(api_key=API_KEY)

def tavily_search(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper for Tavily API call.
    Supports both direct API key and Cloud Run proxy mode (to hide secrets).
    """
    base = {
        "topic": "news",
        "search_depth": "advanced",    # use "basic" for cheaper exploration
        "include_raw_content": True,
        "max_results": 10,
        "time_range": "week",
        "include_answer": False,
        "include_favicon": False,      # favicon removed
    }
    base.update(params)
    if USE_PROXY:
        import urllib.request
        req = urllib.request.Request(
            url=PROXY_URL.rstrip("/") + "/search",
            data=json.dumps(base).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)
    else:
        return tavily.search(**base)

# ---------- Predefined queries ----------
QUERY_PRESETS = {
    "fed_policy": "Federal Reserve FOMC interest rate path dot plot policy statement",
    "yield_curve": "US Treasury yield curve inversion steepening WGS10YR WGS2YR weekly",
    "prime_mortgage": "US mortgage rates 30-year fixed average MORTGAGE30US weekly",
    "ff_effective": "Federal Funds Effective Rate weekly update",
    "cpi": "US CPI inflation monthly report CPIAUCSL release",
    "labor": "US unemployment rate and weekly initial claims ICSA",
    "credit_card_dq": "US credit card delinquency trend bank earnings consumer stress",
    "consumer_credit": "US consumer credit outstanding TOTALSL revolving growth",
    "wei": "Weekly Economic Index WEI update",
    "oil": "WTI crude oil weekly price drivers recession risk",
    "gas_prices": "US gasoline prices weekly update",
}

# ---------- Trusted domains ----------
TRUSTED_DOMAINS = [
    "reuters.com","wsj.com","bloomberg.com","ft.com","apnews.com","nytimes.com",
    "washingtonpost.com","marketwatch.com","cnbc.com",
    "federalreserve.gov","stlouisfed.org","bls.gov","bea.gov","treasury.gov",
    "eia.gov","imf.org","bis.org","oecd.org","conference-board.org",
    "visa.com","spglobal.com","sifma.org","adp.com","equifax.com",
]

# ---------- Utility functions ----------
def domain_of(url: str) -> str:
    """Extract domain name from URL."""
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""

def make_id(url: str, title: str) -> str:
    """Generate deterministic unique ID (SHA-256 of URL + title)."""
    key = (url or "") + "||" + (title or "")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]

# ---------- Main data collection ----------
def collect_samples(max_per_query=8) -> pd.DataFrame:
    """
    Run multiple predefined queries against Tavily API.
    Collect sample news articles across topics for schema exploration.
    """
    rows = []
    for tag, q in QUERY_PRESETS.items():
        data = tavily_search({
            "query": q,
            "include_domains": TRUSTED_DOMAINS,
            "max_results": max_per_query,
        })
        results = data.get("results", [])
        fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
        
        # Print one example JSON per topic for inspection
        if results:
            print(f"\n--- [{tag}] example ---")
            print(json.dumps(
                {k: results[0].get(k) for k in ["title","url","author","published_date","score","content"]},
                ensure_ascii=False, indent=2
            )[:1000])

        for r in results:
            title = r.get("title")
            url = r.get("url")
            pub = r.get("published_date") or r.get("published_at")
            try:
                pub_ts = pd.to_datetime(pub, errors="coerce")
            except Exception:
                pub_ts = pd.NaT

            rows.append({
                "id": make_id(url, title),
                "topic_tag": tag,
                "query": q,
                "title": title,
                "url": url,
                "source_domain": domain_of(url),
                "score": r.get("score"),
                "published_at": (pub_ts.tz_localize(None) if isinstance(pub_ts, pd.Timestamp) and not pd.isna(pub_ts) else None),
                "content_md": r.get("raw_content") or r.get("content"),
                "author": r.get("author"),
                "raw_published": pub,
            })
        time.sleep(0.5)
    df = pd.DataFrame(rows).drop_duplicates(subset=["url"]).reset_index(drop=True)
    return df

# ---------- Profiling ----------
def profile_df(df: pd.DataFrame):
    """Basic EDA summary: columns, null ratio, domain/topic counts, text lengths."""
    print("\n=== Shape ===")
    print(df.shape)

    print("\n=== Columns ===")
    print(df.columns.tolist())

    print("\n=== Null ratio ===")
    print(df.isna().mean().sort_values(ascending=False).head(20))

    print("\n=== Sample rows ===")
    with pd.option_context("display.max_colwidth", 120):
        print(df[["title","source_domain","published_at","score"]].head(10))

    print("\n=== Top domains ===")
    print(df["source_domain"].value_counts().head(20))

    print("\n=== Topic distribution ===")
    print(df["topic_tag"].value_counts())

    if "published_at" in df.columns and df["published_at"].notna().any():
        print("\n=== Published date range ===")
        print(df["published_at"].min(), " ~ ", df["published_at"].max())

    df["content_len"] = df["content_md"].fillna("").map(len)
    print("\n=== Content length statistics ===")
    print(df["content_len"].describe())

# ---------- ERD suggestion ----------
def suggest_erd_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Suggest BigQuery schema (staging + normalized) based on sample data structure.
    """
    staging_cols = [
        "id STRING",
        "fetched_at_utc TIMESTAMP",
        "topic_tag STRING",
        "query STRING",
        "title STRING",
        "url STRING",
        "source_domain STRING",
        "score FLOAT64",
        "published_at TIMESTAMP",
        "content_md STRING",
    ]
    dim_source = [
        "source_id STRING",
        "source_domain STRING",
        "brand STRING",
    ]
    news_article = [
        "article_id STRING",
        "source_id STRING",
        "title STRING",
        "url STRING",
        "published_at TIMESTAMP",
        "ingested_at TIMESTAMP",
        "score FLOAT64",
        "content_md STRING",
    ]
    return staging_cols, dim_source + news_article

# ---------- Main ----------
def main():
    print("Collecting samples from Tavily ...")
    df = collect_samples(max_per_query=8)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    df.insert(1, "fetched_at_utc", now)

    # Run EDA
    profile_df(df)

    # Save sample
    out = "tavily_news_sample.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved sample CSV: {out} ({len(df)} rows)")

    # Print ERD suggestion
    staging_cols, normalized_cols = suggest_erd_columns(df)
    print("\n=== ERD Suggestion: staging (news_raw) ===")
    print(",\n".join(staging_cols))
    print("\n=== ERD Suggestion: normalized (dim_source, news_article) ===")
    print(",\n".join(normalized_cols))

if __name__ == "__main__":
    main()
