import os
import asyncio
import logging
import mlflow
import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from mlflow import MlflowClient
from dotenv import load_dotenv

from inference import Inference
from database import create_tables
from monitoring import (
    _run_drift_report,
    _classify_severity,
    _run_performance_report,
    SERIOUS_DRIFT_THRESHOLD,
    NEGLIGIBLE_DRIFT_THRESHOLD,
)

logger = logging.getLogger(__name__)
load_dotenv()

# ── DB init ───────────────────────────────────────────────────────────────────
try:
    create_tables()
except Exception as e:
    logger.warning(f"DB table creation skipped: {e}")

PREFECT_API_URL       = os.getenv("PREFECT_API_URL")
RETRAINING_DEPLOYMENT = os.getenv("RETRAINING_DEPLOYMENT")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# ── Module-level globals ──────────────────────────────────────────────────────
# Assigned by lifespan at startup, refreshed every 5 min by background loop
inference             = None
current_model_name    = None
current_model_version = None


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    transaction_id:               str
    customer_id:                  str
    transaction_amount:           float
    transaction_type:             str
    transaction_time:             str
    transaction_location:         str
    device_type:                  str
    previous_transactions_count:  int


class PredictionRequest(BaseModel):
    data: list[TransactionRequest]


class MonitorRequest(BaseModel):
    reference_source:   str
    current_source:     str
    reference_query:    str
    current_query:      str
    include_performance: bool = False
    postgres_config:    dict | None = None
    bigquery_config:    dict | None = None


# ── MLflow helpers ────────────────────────────────────────────────────────────

def get_best_model_info() -> tuple[str, str]:
    client = MlflowClient()
    for model in client.search_registered_models():
        try:
            version = client.get_model_version_by_alias(model.name, "production")
            if version:
                return version.name, version.version
        except Exception:
            continue
    raise ValueError("No model found with production alias")


# ── Inference loader ──────────────────────────────────────────────────────────

def _load_inference() -> tuple:
    """
    Instantiates Inference with the current production model.
    Inference lazy loads model + preprocessor + explainer from MinIO.
    No preprocessor injection needed — Inference owns its own loading.
    """
    best_model_name, best_model_version = get_best_model_info()
    inf = Inference(best_model_name=best_model_name)
    return inf, best_model_name, best_model_version


# ── Background model reload ───────────────────────────────────────────────────

async def _background_model_reload(interval_seconds: int = 300):
    """
    Polls MLflow every 5 minutes for a new production alias.
    Reloads silently in background — no user request ever waits
    for a MinIO download.
    """
    global inference, current_model_name, current_model_version
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            latest_name, latest_version = get_best_model_info()
            if latest_name != current_model_name or latest_version != current_model_version:
                logger.info(f"New model detected: {latest_name} v{latest_version} — reloading.")
                inference, current_model_name, current_model_version = _load_inference()
                logger.info("Background model reload complete.")
        except Exception as e:
            logger.warning(f"Background reload check failed: {e}")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup, launch background reload loop."""
    global inference, current_model_name, current_model_version
    inference, current_model_name, current_model_version = _load_inference()
    logger.info(f"Model loaded at startup: {current_model_name} v{current_model_version}")
    asyncio.create_task(_background_model_reload())
    yield


# ── App — defined BEFORE middleware ───────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/predict")
def predict(request: PredictionRequest):
    global inference, current_model_name, current_model_version

    if inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet — try again shortly")

    try:
        df = pd.DataFrame([r.dict() for r in request.data])

        predictions       = inference.predict(df)
        confidence_scores = inference.predict_proba(df)
        explanation       = inference.explain(df)   # ← SHAP explanation

        # ── Log predictions to DB ─────────────────────────────────────────────
        try:
            from database import get_engine, Prediction as PredictionRecord
            from sqlalchemy.orm import Session

            with Session(get_engine()) as session:
                for txn, pred, conf in zip(request.data, predictions, confidence_scores):
                    session.add(PredictionRecord(
                        transaction_id=txn.transaction_id,
                        predicted_label=int(pred),
                        confidence_score=float(conf),
                        model_name=current_model_name,
                        model_version=str(current_model_version),
                    ))
                session.commit()
        except Exception as db_err:
            # DB failure must never break the prediction response
            logger.warning(f"Failed to log predictions to DB: {db_err}")

        return {
            "prediction":    predictions.tolist(),
            "confidence":    confidence_scores.tolist(),
            "explanation":   explanation,              
            "model_used":    current_model_name,
            "model_version": current_model_version,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monitor")
def run_monitor(request: MonitorRequest):
    """
    On-demand drift check.
    """
    try:
        from preprocessing import FeatureEngineer
        from retrain_flow import get_extractor

        # Load reference data
        ref_extractor = get_extractor(
            request.reference_source,
            request.postgres_config or request.bigquery_config
        )
        ref_df = ref_extractor.extract_and_validate(request.reference_query)
        ref_fe = FeatureEngineer(ref_df)
        ref_fe.cleaning()
        reference_df = ref_fe.transform()

        # Load current data
        curr_extractor = get_extractor(
            request.current_source,
            request.postgres_config or request.bigquery_config
        )
        curr_df = curr_extractor.extract_and_validate(request.current_query)
        curr_fe = FeatureEngineer(curr_df)
        curr_fe.cleaning()
        current_df = curr_fe.transform()

        # Run drift report — same functions as monitoring.py
        data_dict     = _run_drift_report(reference_df, current_df)
        drift_metrics = data_dict["metrics"][0]["result"]
        share_drifted = drift_metrics["share_of_drifted_columns"]

        target_drift     = None
        prediction_drift = None
        for metric in data_dict["metrics"][1:]:
            col      = metric["result"].get("column_name")
            detected = metric["result"].get("drift_detected", False)
            if col == "is_fraud":
                target_drift = detected
            elif col == "predicted_fraud":
                prediction_drift = detected

        # Performance monitoring
        recall = None
        performance_result = {"available": False, "reason": "Not requested"}
        if request.include_performance:
            try:
                from database import get_engine
                import sqlalchemy as sa
                engine = get_engine()
                query = sa.text("""
                    SELECT p.transaction_id, p.predicted_label,
                           f.actual_label, f.confirmed_at
                    FROM predictions p
                    INNER JOIN fraud_feedback f
                        ON p.transaction_id = f.transaction_id
                    ORDER BY f.confirmed_at DESC
                """)
                with engine.connect() as conn:
                    perf_df = pd.read_sql(query, conn)

                if not perf_df.empty:
                    perf_dict      = _run_performance_report(perf_df)
                    metrics_result = perf_dict["metrics"][0]["result"]["current"]
                    recall         = metrics_result.get("recall", 1.0)
                    performance_result = {
                        "available":           True,
                        "precision":           metrics_result.get("precision"),
                        "recall":              recall,
                        "f1":                  metrics_result.get("f1"),
                        "low_recall_detected": recall < 0.70,
                    }
            except Exception as e:
                performance_result = {"available": False, "reason": str(e)}

        severity = _classify_severity(share_drifted, prediction_drift, target_drift, recall)
        retraining_recommended = severity == "serious"

        # Trigger retraining if serious
        retraining_triggered = False
        if retraining_recommended:
            try:
                response = httpx.post(
                    f"{PREFECT_API_URL}/deployments/{RETRAINING_DEPLOYMENT}/create_flow_run",
                    json={"parameters": {
                        "source":           request.current_source,
                        "query":            request.current_query,
                        "extractor_config": request.postgres_config or request.bigquery_config,
                        "trigger_reason":   "drift"
                    }},
                    timeout=15,
                )
                retraining_triggered = response.status_code in [200, 201]
            except Exception as e:
                logger.error(f"Failed to trigger retraining: {e}")

        return {
            "status":                "ok",
            "drift_severity":        severity,
            "data_drift": {
                "drift_score":      share_drifted,
                "drifted_features": drift_metrics["number_of_drifted_columns"],
            },
            "target_drift":          {"drift_detected": target_drift},
            "prediction_drift":      {"drift_detected": prediction_drift},
            "performance":           performance_result,
            "retraining_recommended": retraining_recommended,
            "retraining_triggered":  retraining_triggered,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status":                 "ok",
        "model":                  current_model_name,
        "model_version":          current_model_version,
        "model_loaded":           inference is not None,
        "reload_interval_seconds": 300,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)