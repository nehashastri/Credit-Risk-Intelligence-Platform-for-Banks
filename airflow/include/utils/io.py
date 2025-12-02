from pathlib import Path

def read_sql(path: str) -> str:
    return Path(path).read_text()
