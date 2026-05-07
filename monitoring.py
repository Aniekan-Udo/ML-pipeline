import os
import logging
import pandas as pd
import mlflow
import httpx
from prefect import flow, task
from dotenv import load_dotenv

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import ColumnDriftMetric
from evidently.pipeline.column_mapping import ColumnMapping

from minio_storage import StorageSingleton, preprocessor_key, model_key

load_dotenv()

logger = logging.getLogger(__name__)

PREFECT_API_URL       = os.getenv("PREFECT_API_URL")
RETRAINING_DEPLOYMENT = os.getenv("RETRAINING_DEPLOYMENT")
SLACK_WEBHOOK_URL     = os.getenv("SLACK_WEBHOOK_URL")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

SERIOUS_DRIFT_THRESHOLD    = 0.50
NEGLIGIBLE_DRIFT_THRESHOLD = 0.15


# ── Core report functions — Evidently IS the abstraction ─────────────────────

def _run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """
    Runs Evidently data drift + target drift + prediction drift report.
    Layer 1 — DataDriftPreset    : feature distribution shift
    Layer 2 — ColumnDriftMetric  : target drift (is_fraud label shift)
    Layer 2 — ColumnDriftMetric  : prediction drift (model output shift)
    """
    metrics = [
        DataDriftPreset(),
        ColumnDriftMetric(column_name="is_fraud"),   # target drift — always measured
    ]

    # Prediction drift — only if model has already scored both datasets
    if "predicted_fraud" in reference.columns and "predicted_fraud" in current.columns:
        metrics.append(ColumnDriftMetric(column_name="predicted_fraud"))

    report = Report(metrics=metrics)
    report.run(reference_data=reference, current_data=current)
    return report.as_dict()


def _run_performance_report(performance_df: pd.DataFrame) -> dict:
    """Runs Evidently classification performance report directly."""
    col_mapping = ColumnMapping(target="actual_label", prediction="predicted_label")
    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=None, current_data=performance_df, column_mapping=col_mapping)
    return report.as_dict()


def _classify_severity(
    share_drifted: float,
    prediction_drift: bool | None,
    target_drift: bool | None,
    recall: float | None,
) -> str:
    """
    Severity classification:
      serious    — ≥50% features drifted OR prediction drift OR target drift OR recall < 0.70
      negligible — 15–49% features drifted, no prediction/target drift
      none       — < 15% drifted, all healthy
    """
    if (share_drifted >= SERIOUS_DRIFT_THRESHOLD
            or bool(prediction_drift)
            or bool(target_drift)           # ← target drift always serious
            or (recall is not None and recall < 0.70)):
        return "serious"
    if share_drifted >= NEGLIGIBLE_DRIFT_THRESHOLD:
        return "negligible"
    return "none"


# ── Prefect Tasks ─────────────────────────────────────────────────────────────

@task(name="load reference data")
def load_reference() -> pd.DataFrame:
    """
    Loads post-feature-engineering training snapshot from MinIO.
    Saved during every training/retraining run as reference_data.parquet.
    """
    try:
        return StorageSingleton.get().load_parquet("reference/reference_data.parquet")
    except Exception as e:
        ref_path = os.getenv("REFERENCE_DATA_PATH", "reference_data.parquet")
        if os.path.exists(ref_path):
            return pd.read_parquet(ref_path)
        raise RuntimeError(f"Failed to load reference data: {e}")


@task(name="load current data")
def load_current(source: str, query: str, extractor_config: dict | None = None) -> pd.DataFrame:
    """
    Pulls fresh transactions via ExtractorPort and applies
    the same feature engineering as training — so columns match reference.
    NOTE: Preprocessor scaling intentionally NOT applied — raw feature
    distributions must be compared for drift to be meaningful.
    """
    from retrain_flow import get_extractor
    from preprocessing import FeatureEngineer

    extractor = get_extractor(source, extractor_config)
    df = extractor.extract_and_validate(query)
    fe = FeatureEngineer(df)
    fe.cleaning()
    return fe.transform()


@task(name="score for prediction drift")
def score_for_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads production model from MinIO and adds 'predicted_fraud' column
    to both datasets so prediction drift can be measured in Layer 2.
    Skips silently if no production model found.
    """
    try:
        from mlflow import MlflowClient

        client = MlflowClient()
        best_model_name = None

        for model in client.search_registered_models():
            try:
                v = client.get_model_version_by_alias(model.name, "production")
                if v:
                    best_model_name = model.name.replace("fraud-", "")
                    break
            except Exception:
                continue

        if best_model_name is None:
            logger.warning("No production model found — skipping prediction drift scoring.")
            return reference_df, current_df

        storage = StorageSingleton.get()
        preprocessor = storage.load(preprocessor_key())
        model = storage.load(model_key(best_model_name))

        for df in [reference_df, current_df]:
            features = df.drop(columns=["is_fraud"], errors="ignore")
            X = preprocessor.transform(features)
            df["predicted_fraud"] = model.predict(X)

    except Exception as e:
        logger.warning(f"score_for_drift skipped: {e}")

    return reference_df, current_df


@task(name="load performance data")
def load_performance_data() -> pd.DataFrame | None:
    """
    Option B — joins predictions with fraud_feedback table.
    Returns None gracefully if no feedback data has arrived yet.
    Activates automatically once dispute/chargeback data starts flowing.
    """
    try:
        from database import get_engine
        import sqlalchemy as sa

        engine = get_engine()
        query = sa.text("""
            SELECT
                p.transaction_id,
                p.predicted_label,
                p.confidence_score,
                f.actual_label,
                f.source,
                f.confirmed_at
            FROM predictions p
            INNER JOIN fraud_feedback f
                ON p.transaction_id = f.transaction_id
            ORDER BY f.confirmed_at DESC
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            logger.info("No feedback rows yet — Option B activates once feedback arrives.")
            return None

        logger.info(f"Loaded {len(df)} matched prediction+feedback rows.")
        return df

    except Exception as e:
        logger.warning(f"load_performance_data failed: {e}")
        return None


@task(name="run drift check")
def run_drift_check(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    include_performance: bool = False,
    performance_df: pd.DataFrame | None = None,
) -> dict:
    """
    Orchestrates all three monitoring layers:
      Layer 1 — Data drift       : feature distribution shift
      Layer 2 — Target drift     : is_fraud label distribution shift
      Layer 2 — Prediction drift : model output distribution shift
      Layer 3 — Performance      : recall/precision from feedback data
    """
    # ── Layers 1 + 2 ─────────────────────────────────────────────────────────
    data_dict     = _run_drift_report(reference_df, current_df)
    drift_metrics = data_dict["metrics"][0]["result"]
    share_drifted = drift_metrics["share_of_drifted_columns"]
    dataset_drift = drift_metrics["dataset_drift"]

    # Extract target drift and prediction drift from ColumnDriftMetric results
    target_drift     = None
    prediction_drift = None

    for metric in data_dict["metrics"][1:]:
        col  = metric["result"].get("column_name")
        detected = metric["result"].get("drift_detected", False)
        if col == "is_fraud":
            target_drift = detected
        elif col == "predicted_fraud":
            prediction_drift = detected

    # ── Layer 3 — Performance ─────────────────────────────────────────────────
    recall = None
    performance_result = {"available": False, "reason": "No feedback data yet"}

    if include_performance and performance_df is not None and not performance_df.empty:
        try:
            perf_dict      = _run_performance_report(performance_df)
            metrics_result = perf_dict["metrics"][0]["result"]["current"]
            recall         = metrics_result.get("recall", 1.0)
            performance_result = {
                "available":           True,
                "precision":           metrics_result.get("precision"),
                "recall":              recall,
                "f1":                  metrics_result.get("f1"),
                "roc_auc":             metrics_result.get("roc_auc"),
                "low_recall_detected": recall < 0.70,
            }
        except Exception as e:
            performance_result = {"available": False, "reason": str(e)}

    severity = _classify_severity(share_drifted, prediction_drift, target_drift, recall)

    return {
        "data_drift": {
            "drift_score":            share_drifted,
            "drifted_features":       drift_metrics["number_of_drifted_columns"],
            "dataset_drift_detected": dataset_drift,
        },
        "target_drift": {
            "drift_detected": target_drift,
            "available":      target_drift is not None,
        },
        "prediction_drift": {
            "drift_detected": prediction_drift,
            "available":      prediction_drift is not None,
        },
        "performance":            performance_result,
        "drift_severity":         severity,
        "retraining_recommended": severity == "serious",
    }


@task(name="notify slack")
def notify_slack(result: dict, retraining_triggered: bool):
    """
    Sends Slack summary for every monitoring run regardless of severity.
    Skips silently if SLACK_WEBHOOK_URL is not configured.
    """
    if not SLACK_WEBHOOK_URL:
        logger.warning("SLACK_WEBHOOK_URL not set — skipping Slack notification.")
        return

    severity    = result.get("drift_severity", "none")
    drift_score = result["data_drift"]["drift_score"]
    pred_drift  = result["prediction_drift"].get("drift_detected")
    tgt_drift   = result["target_drift"].get("drift_detected")
    perf        = result.get("performance", {})

    severity_map = {
        "serious":    ("🚨", "*SERIOUS DRIFT DETECTED — Immediate retraining triggered*"),
        "negligible": ("⚠️",  "*Negligible drift — Scheduled for Sunday retraining*"),
        "none":       ("✅", "*No significant drift — Pipeline healthy*"),
    }
    emoji, header = severity_map.get(severity, ("ℹ️", "*Drift check complete*"))

    lines = [
        header,
        f"{emoji} *Severity*: `{severity.upper()}`",
        f"📊 *Feature drift score*: `{drift_score:.1%}` of columns drifted",
        f"🎯 *Target drift* (is_fraud): `{'Yes' if tgt_drift else 'No' if tgt_drift is not None else 'Not measured'}`",
        f"🤖 *Prediction drift*: `{'Yes' if pred_drift else 'No' if pred_drift is not None else 'Not measured'}`",
    ]

    if perf.get("available"):
        lines.append(
            f"📉 *Performance*: recall=`{perf['recall']:.2%}` | "
            f"precision=`{perf['precision']:.2%}` | f1=`{perf['f1']:.2%}`"
        )
        if perf.get("low_recall_detected"):
            lines.append("⚠️ *Recall below 0.70 — model may be missing fraud cases*")

    lines.append(f"🔄 *Immediate retraining triggered*: `{'Yes' if retraining_triggered else 'No'}`")
    if severity == "negligible" and not retraining_triggered:
        lines.append("📅 Retraining will run on Sunday midnight schedule.")

    try:
        httpx.post(
            SLACK_WEBHOOK_URL,
            json={"text": "\n".join(lines)},
            timeout=10
        ).raise_for_status()
        logger.info("Slack notification sent.")
    except Exception as e:
        logger.error(f"Slack notification failed: {e}")


@task(name="trigger retraining if needed")
def trigger_retraining_if_needed(
    result: dict,
    source: str,
    query: str,
    extractor_config: dict | None = None,
) -> bool:
    """
    Triggers immediate retraining ONLY for serious drift.
    Negligible drift is intentionally left to the Sunday scheduled run.
    """
    severity = result.get("drift_severity", "none")

    if severity == "serious":
        logger.info("Serious drift — triggering immediate retraining.")
        try:
            response = httpx.post(
                f"{PREFECT_API_URL}/deployments/{RETRAINING_DEPLOYMENT}/create_flow_run",
                json={"parameters": {
                    "source":           source,
                    "query":            query,
                    "extractor_config": extractor_config,
                    "trigger_reason":   "serious_drift",
                }},
                timeout=15,
            )
            success = response.status_code in [200, 201]
            logger.info("Retraining triggered." if success else f"Trigger failed: HTTP {response.status_code}")
            return success
        except Exception as e:
            logger.error(f"Failed to trigger retraining: {e}")
            return False

    elif severity == "negligible":
        logger.info("Negligible drift — Sunday retrain will handle this.")
        return False

    logger.info("No drift — pipeline healthy.")
    return False



@flow(name="Daily Drift Monitoring")
def daily_monitoring_flow(
    source: str = "postgres",
    query: str = "SELECT * FROM transactions WHERE transaction_time >= NOW() - INTERVAL '1 day'",
    extractor_config: dict | None = None,
    include_performance: bool = True,
):
    """
    Runs every day at 9am via Prefect cron schedule.
    Three monitoring layers with tiered retraining response:

      Serious    (≥50% features OR target drift OR prediction drift OR recall<0.70)
                 → retrain immediately + Slack alert
      Negligible (15–49% features, no target/prediction drift)
                 → Slack alert, Sunday retrain
      None       (<15% features)
                 → Slack OK confirmation
    """
    reference_df = load_reference()
    current_df   = load_current(source, query, extractor_config)

    # Score both datasets so prediction drift can be measured
    reference_df, current_df = score_for_drift(reference_df, current_df)

    # Load feedback data for performance monitoring (None if empty)
    performance_df = load_performance_data() if include_performance else None

    result = run_drift_check(reference_df, current_df, include_performance, performance_df)

    retraining_triggered = trigger_retraining_if_needed(result, source, query, extractor_config)
    notify_slack(result, retraining_triggered)

    logger.info(f"Severity        : {result['drift_severity'].upper()}")
    logger.info(f"Feature drift   : {result['data_drift']['drift_score']:.2%}")
    logger.info(f"Target drift    : {result['target_drift']}")
    logger.info(f"Prediction drift: {result['prediction_drift']}")
    logger.info(f"Performance     : {result['performance']}")
    logger.info(f"Retrain         : {retraining_triggered}")

    return {
        "drift_summary":       result,
        "retraining_triggered": retraining_triggered,
    }


if __name__ == "__main__":
    daily_monitoring_flow.serve(
        name="daily-monitoring-deployment",
        cron="0 9 * * *"
    )