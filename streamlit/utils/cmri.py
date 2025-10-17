import pandas as pd
import numpy as np

def compute_cmri(df: pd.DataFrame):
    cols = ["unrate","fedfunds","cpiaucsl","t5yie","drcclacbs"]
    df["cmri"] = df[cols].apply(lambda x: np.nanmean(x), axis=1)
    df["regime_label"] = pd.cut(df["cmri"],
                                bins=[-999,-0.5,0.5,999],
                                labels=["Expansion","Normal","Stress"])
    return df
