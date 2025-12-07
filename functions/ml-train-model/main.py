import functions_framework
from google.cloud import bigquery, storage
import pandas as pd
import json
import joblib
import io
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import math
import random

# Optional imports for XGBoost & LightGBM
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


# -------------------------------------------------------------------------
# Helper: Read SQL file
# -------------------------------------------------------------------------
def read_sql(filename: str) -> str:
    """Read a SQL file from the sql/ directory."""
    base_dir = Path(__file__).parent
    possible_paths = [base_dir / filename, base_dir / "sql" / filename]

    for path in possible_paths:
        if path.exists():
            print(f"✅ Found SQL file: {path}")
            return path.read_text(encoding="utf-8")

    raise FileNotFoundError(f"SQL file not found: {filename}")


# -------------------------------------------------------------------------
# Helper metrics
# -------------------------------------------------------------------------
def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.maximum(denom, 1e-8)
    return 100 * np.mean(diff)


def rmse(y_true, y_pred):
    return math.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def mase(y_true, y_pred):
    """Mean Absolute Scaled Error — scaled to lag-1 naïve forecast."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    return mae_model / (mae_naive + 1e-8)


# -------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------
PROJECT_ID = "pipeline-882-team-project"
BUCKET_NAME = "pipeline-882-team-project-models"
DATASET_TABLE = f"{PROJECT_ID}.gold.fact_all_indicators_weekly"


# -------------------------------------------------------------------------
# Hyperparameter search space
# -------------------------------------------------------------------------
def random_search_params(algorithm: str, n_iter: int = 20):
    """
    Return a list of randomly sampled hyperparameter dictionaries
    for the given algorithm.
    """
    search_space = {
        # No real hyperparams – just keep a dummy entry for consistency
        "linear_regression": [{}],

        "elastic_net": {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },

        "random_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 8, 12],
            "min_samples_split": [2, 5, 10],
        },

        "gradient_boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4],
        },

        "xgboost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5, 6],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },

        "lightgbm": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 63, 127],
        },

        # -------- Time series models --------
        # Plain ARIMA on the target only
        "arima": {
            "p": [0, 1, 2, 3],
            "d": [0, 1],
            "q": [0, 1, 2],
        },

        # SARIMAX with exogenous regressors (X_train, X_test)
        "sarimax": {
            "p": [0, 1, 2],
            "d": [0, 1],
            "q": [0, 1],
            # Weekly macro data – allow no seasonality or yearly (52-week) seasonality
            "sp": [0, 1],
            "sd": [0, 1],
            "sq": [0, 1],
            "s": [0, 52],
        },

        # VARMAX treating the target + some features as a multivariate system
        "varmax": {
            "order": [(1, 0), (2, 0), (1, 1)],
        },
    }

    # If algorithm not in search space or has no hyperparameters
    if algorithm not in search_space:
        return [{}]

    space = search_space[algorithm]

    # If we explicitly provided a list of param combinations
    if isinstance(space, list):
        return space

    # Linear regression has a single '{}' config
    if algorithm == "linear_regression":
        return space

    # Random cartesian sampling from dict-of-lists
    keys = list(space.keys())
    samples = []
    for _ in range(n_iter):
        params = {k: random.choice(space[k]) for k in keys}
        samples.append(params)
    return samples


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
@functions_framework.http
def task(request):
    """
    Train and evaluate model with extended metrics:
    sMAPE, RMSE_recent6, MAE, Pearson r, R², MASE,
    using internal random search (no hyperparameters passed in).
    """

    # ---------------- Parse params ----------------
    algorithm = request.args.get("algorithm")
    run_id = request.args.get("run_id")
    dataset_id = request.args.get("dataset_id")
    model_id = request.args.get("model_id")

    if not all([algorithm, run_id, dataset_id, model_id]):
        return {"error": "Missing required parameters"}, 400

    print(f"🔧 Algorithm: {algorithm}")
    print(f"🔧 Run ID: {run_id}, Dataset ID: {dataset_id}, Model ID: {model_id}")

    # ---------------- Load data ----------------
    bq = bigquery.Client(project=PROJECT_ID)
    train_query = read_sql("load-train-data.sql")
    test_query = read_sql("load-test-data.sql")

    print("🔹 Loading training data...")
    train_df = bq.query(train_query).to_dataframe()

    print("🔹 Loading test data...")
    test_df = bq.query(test_query).to_dataframe()

    if "delinq" not in train_df.columns:
        return {"error": "Target variable 'delinq' not found."}, 400

    # ---------------- Feature prep ----------------
    # We keep date/week/year out of feature set, but still use date for TS models
    feature_cols = [c for c in train_df.columns if c not in ("delinq", "date", "week", "year")]
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["delinq"].fillna(0)
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["delinq"].fillna(0)

    print("y_test (first 20):")
    print(y_test.head(20).tolist())

    # ---------------- Baseline model (no random search) ----------------
    if algorithm == "base":
        print("🧠 Using baseline rolling-mean model...")
        # Simple 3-point rolling mean on training
        baseline = y_train.rolling(window=3, min_periods=1).mean().iloc[-1]
        y_pred = np.full_like(y_test, baseline, dtype=float)
        model = {"type": "baseline", "value": baseline}
        best_hyperparams = {}
    else:
        # ---------------- Random Search ----------------
        print(f"🧠 Running Random Search for {algorithm}...")
        param_candidates = random_search_params(algorithm, n_iter=20)

        best_model = None
        best_pred = None
        best_hyperparams = None
        best_mae = float("inf")
        any_success = False

        # For time-series models, we sometimes need y as numpy or pandas
        y_train_series = y_train.astype(float)
        y_test_series = y_test.astype(float)

        for params in param_candidates:
            print(f"  🔍 Trying params: {params}")
            try:
                # -----------------------------------------------------------------
                # Classical ML models on tabular features
                # -----------------------------------------------------------------
                if algorithm == "linear_regression":
                    model_temp = LinearRegression()
                    model_temp.fit(X_train, y_train_series)
                    y_pred_temp = model_temp.predict(X_test)

                elif algorithm == "elastic_net":
                    model_temp = ElasticNet(**params, random_state=42)
                    model_temp.fit(X_train, y_train_series)
                    y_pred_temp = model_temp.predict(X_test)

                elif algorithm == "random_forest":
                    model_temp = RandomForestRegressor(**params, random_state=42)
                    model_temp.fit(X_train, y_train_series)
                    y_pred_temp = model_temp.predict(X_test)

                elif algorithm == "gradient_boosting":
                    model_temp = GradientBoostingRegressor(**params, random_state=42)
                    model_temp.fit(X_train, y_train_series)
                    y_pred_temp = model_temp.predict(X_test)

                elif algorithm == "xgboost":
                    if xgb is None:
                        return {"error": "xgboost not installed"}, 400
                    model_temp = xgb.XGBRegressor(**params, random_state=42)
                    model_temp.fit(X_train, y_train_series)
                    y_pred_temp = model_temp.predict(X_test)

                elif algorithm == "lightgbm":
                    if lgb is None:
                        return {"error": "lightgbm not installed"}, 400
                    model_temp = lgb.LGBMRegressor(**params, random_state=42)
                    model_temp.fit(X_train, y_train_series)
                    y_pred_temp = model_temp.predict(X_test)

                # -----------------------------------------------------------------
                # Time series models
                # -----------------------------------------------------------------
                elif algorithm == "arima":
                    # ARIMA on the target only (univariate)
                    from statsmodels.tsa.arima.model import ARIMA

                    order = (params["p"], params["d"], params["q"])
                    ts_model = ARIMA(y_train_series, order=order)
                    ts_fit = ts_model.fit()
                    y_pred_temp = ts_fit.forecast(steps=len(y_test_series))
                    model_temp = ts_fit  # Save fitted model

                elif algorithm == "sarimax":
                    # SARIMAX with exogenous regressors
                    from statsmodels.tsa.statespace.sarimax import SARIMAX

                    order = (params["p"], params["d"], params["q"])

                    # seasonal_period s; if s == 0 → disable seasonality
                    s = params["s"]
                    if s == 0:
                        seasonal_order = (0, 0, 0, 0)
                    else:
                        seasonal_order = (
                            params["sp"],
                            params["sd"],
                            params["sq"],
                            s,
                        )

                    ts_model = SARIMAX(
                        y_train_series,
                        exog=X_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    ts_fit = ts_model.fit(disp=False)
                    y_pred_temp = ts_fit.forecast(steps=len(y_test_series), exog=X_test)
                    model_temp = ts_fit

                elif algorithm == "varmax":
                    # VARMAX on target + features as a system
                    from statsmodels.tsa.statespace.varmax import VARMAX

                    # Combine target + some features (endogenous variables)
                    # To keep it manageable, you could optionally select a subset of key macro features.
                    var_df_train = pd.concat(
                        [y_train_series.rename("delinq"), X_train], axis=1
                    )

                    order = params["order"]  # tuple like (1, 0) or (2, 0)
                    ts_model = VARMAX(var_df_train, order=order, enforce_stationarity=False)
                    ts_fit = ts_model.fit(disp=False)

                    # Forecast future steps (same horizon as len(y_test))
                    var_forecast = ts_fit.forecast(steps=len(y_test_series))
                    # First column is 'delinq' as we named above
                    y_pred_temp = var_forecast["delinq"].values
                    model_temp = ts_fit

                else:
                    return {"error": f"Unknown algorithm: {algorithm}"}, 400

                # Evaluate MAE
                mae_temp = mean_absolute_error(y_test_series, y_pred_temp)
                print(f"    → MAE: {mae_temp:.4f}")

                if mae_temp < best_mae and np.isfinite(mae_temp):
                    best_mae = mae_temp
                    best_model = model_temp
                    best_pred = np.asarray(y_pred_temp, dtype=float)
                    best_hyperparams = params
                    any_success = True

            except Exception as e:
                print(f"❌ Failed for params {params}: {e}")
                continue

        if not any_success:
            return {"error": f"All parameter combinations failed for algorithm {algorithm}"}, 500

        print(f"🏆 Best MAE: {best_mae:.4f}")
        print(f"🏆 Best hyperparameters: {best_hyperparams}")

        model = best_model
        y_pred = best_pred
        # For metrics below, we already know best_mae

    # ---------------- Metrics ----------------
    y_test_arr = np.array(y_test, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    pearson_r = None
    if len(y_test_arr) > 1 and np.std(y_pred_arr) > 0 and np.std(y_test_arr) > 0:
        pearson_r = float(np.corrcoef(y_test_arr, y_pred_arr)[0, 1])

    smape_val = smape(y_test_arr, y_pred_arr)
    mae_val = mean_absolute_error(y_test_arr, y_pred_arr)
    mase_val = mase(y_test_arr, y_pred_arr)
    r2_val = r2_score(y_test_arr, y_pred_arr)
    if len(y_test_arr) >= 6:
        rmse_recent6_val = rmse(y_test_arr[-6:], y_pred_arr[-6:])
    else:
        rmse_recent6_val = rmse(y_test_arr, y_pred_arr)

    print(f"✅ Pearson r: {pearson_r:.4f}" if pearson_r is not None else "⚠️ Not enough data/variance for correlation")
    print(f"✅ sMAPE: {smape_val:.2f}%")
    print(f"✅ MAE: {mae_val:.4f}")
    print(f"✅ MASE: {mase_val:.4f}")
    print(f"✅ R²: {r2_val:.4f}")
    print(f"✅ RMSE_recent6: {rmse_recent6_val:.4f}")

    # ---------------- Save model ----------------
    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes)
    model_bytes.seek(0)

    gcs_path = f"models/model_id={model_id}/dataset_id={dataset_id}/run_id={run_id}/model.pkl"
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    bucket.blob(gcs_path).upload_from_file(
        model_bytes, content_type="application/octet-stream"
    )

    # ---------------- Return response ----------------
    result = {
        "run_id": run_id,
        "dataset_id": dataset_id, 
        "algorithm": algorithm,
        "best_hyperparameters": best_hyperparams if algorithm != "base" else {},
        "gcs_path": f"gs://{BUCKET_NAME}/{gcs_path}",
        "metrics": {
            "pearson_r": round(float(pearson_r), 4) if pearson_r is not None else None,
            "smape": round(float(smape_val), 2),
            "mae": round(float(mae_val), 4),
            "mase": round(float(mase_val), 4),
            "r2": round(float(r2_val), 4),
            "rmse_recent6": round(float(rmse_recent6_val), 4),
            "test_count": int(len(y_test_arr)),
        },
        "feature_count": len(feature_cols),
    }

    print(json.dumps(result, indent=2))
    return result, 200
