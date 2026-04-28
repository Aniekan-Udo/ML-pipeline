import os
import io
import mlflow
import pandas as pd
from mlflow import MlflowClient
from prefect import flow, task
from dotenv import load_dotenv

import boto3
from botocore.client import Config

from extraction import CSVExtractor, PostgresExtractor, BigQueryExtractor
from preprocessing import FeatureEngineer, Preprocessor
from train import Trainer
from validate import Validator
from model_config import MODEL_CONFIG

from minio_storage import save, preprocessor_key

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))


def get_extractor(source: str, config: dict | None = None):
    if source == "csv":
        return CSVExtractor()
    elif source == "postgres":
        return PostgresExtractor(**config)
    elif source == "bigquery":
        return BigQueryExtractor(**config)
    else:
        raise ValueError(f"Unknown source: {source}")


@task(name="retrain extract")
def retrain_extract(source: str, query: str, config: dict | None = None):
    extractor = get_extractor(source, config)
    return extractor.extract_and_validate(query)


@task(name="retrain feature engineering")
def retrain_feature_engineer(df):
    fe = FeatureEngineer(df)
    fe.cleaning()
    return fe.transform()


@task(name="retrain preprocess")
def retrain_preprocess(df, target_col):
    preprocessor = Preprocessor(df)

    X_train, X_test, X_val, y_train, y_test, y_val = preprocessor.run(target_col=target_col)
    feature_names = preprocessor.get_feature_names_out()

    mlflow.set_experiment("fraud-detection-retraining")

    with mlflow.start_run(run_name="preprocessor"):
        preprocessor_uri = save(preprocessor, preprocessor_key())
        mlflow.log_param("preprocessor_minio_uri", preprocessor_uri)

    return X_train, X_test, X_val, y_train, y_test, y_val, preprocessor, feature_names


@task(name="update reference data")
def update_reference_data(df: pd.DataFrame):
    """
    Overwrites the reference baseline in MinIO with the current training data.

    Called ONLY after a successful model promotion so the baseline always
    reflects the data distribution the live production model was trained on.

    This prevents the drift monitor from using a stale baseline that would
    cause false positives (drift always triggered) as real-world data evolves.
    """
    bucket = os.getenv("MINIO_BUCKET", "mlflow-artifacts")
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.upload_fileobj(buf, bucket, "reference/reference_data.parquet")

    uri = f"s3://{bucket}/reference/reference_data.parquet"
    mlflow.set_experiment("fraud-detection-retraining")
    with mlflow.start_run(run_name="reference-data-refresh"):
        mlflow.log_param("reference_data_minio_uri", uri)
        mlflow.log_param("reference_row_count", len(df))

    return uri


@task(name="retrain train")
def retrain_train(X_train, y_train, feature_names):
    trainer = Trainer(X_train=X_train, y_train=y_train, feature_names=feature_names)
    trainer.train()


@task(name="retrain validate and promote")
def retrain_validate_and_promote(X_val, y_val):
    mlflow.set_experiment("fraud-detection-retraining")

    with mlflow.start_run(run_name="validation"):
        validator = Validator(
            model_names=list(MODEL_CONFIG.keys()),
            X_val=X_val,
            y_val=y_val
        )
        results = validator.evaluate()

        for name, metrics in results.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_metrics(metrics)

    best = validator.best_model(results)

    client = MlflowClient()
    latest_version = client.get_latest_versions(f"fraud-{best}")[0].version

    # Save previous production for rollback before promoting new version
    try:
        current_prod = client.get_model_version_by_alias(f"fraud-{best}", "production")
        client.set_registered_model_alias(
            name=f"fraud-{best}",
            alias="previous-production",
            version=current_prod.version
        )
    except Exception:
        pass  # no current production version exists yet — first retrain

    # Promote new version to production
    client.set_registered_model_alias(
        name=f"fraud-{best}",
        alias="production",
        version=latest_version
    )

    return best


@flow(name="Retraining Pipeline")
def retrain_pipeline(
    source: str,
    query: str,
    target_col: str = "is_fraud",
    extractor_config: dict | None = None,
    trigger_reason: str = "manual"
):
    mlflow.set_experiment("fraud-detection-retraining")
    # Set trigger metadata as experiment tags (available outside of a run context)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("fraud-detection-retraining")
    if experiment:
        client.set_experiment_tag(experiment.experiment_id, "trigger", trigger_reason)
        client.set_experiment_tag(experiment.experiment_id, "source", source)

    df = retrain_extract(source, query, extractor_config)
    df = retrain_feature_engineer(df)  # df is now feature-engineered

    X_train, X_test, X_val, y_train, y_test, y_val, preprocessor, feature_names = retrain_preprocess(df, target_col)

    retrain_train(X_train, y_train, feature_names)
    best = retrain_validate_and_promote(X_val, y_val)

    # Refresh reference baseline ONLY after successful promotion.
    # This ensures the monitoring system always compares live data against
    # the same data distribution the current production model was trained on.
    update_reference_data(df)

    return best


if __name__ == "__main__":
    retrain_pipeline.serve(name="retraining-deployment")
