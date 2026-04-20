import logging
import pandas as pd
from sklearn.metrics import classification_report

from extraction import CSVExtractor, PostgresExtractor, BigQueryExtractor
from preprocessing import FeatureEngineer, Preprocessor
from train import Trainer, models
from validate import Validator
from inference import Inference
from prefect import flow, task
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.sklearn
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task(name="extract data", description="Extract data from source")
def extract(extractor, query, **kwargs):
    return extractor.extract_and_validate(query, **kwargs)


@task(name="feature engineering", description="Cleans and transforms raw data into model-ready features")
def feature_engineer(df):
    fe = FeatureEngineer(df)
    fe.cleaning()
    return fe.transform()


@task(name="preprocess data", description="Preprocesses the data for training")
def preprocess(df, target_col):
    preprocessor = Preprocessor(df)
    X_train, X_test, X_val = preprocessor.run(target_col=target_col)
    feature_names = preprocessor.get_feature_names_out()

    # Save artifact here — correct responsibility
    with mlflow.start_run(run_name="preprocessor"):
        joblib.dump(preprocessor, "preprocessor.pkl")
        mlflow.log_artifact("preprocessor.pkl")

    return X_train, X_test, X_val, preprocessor, feature_names


@task(name="Training", description="Trains all models and logs to MLflow")
def train(X_train, y_train, feature_names):
    trainer = Trainer(X_train=X_train, y_train=y_train, feature_names=feature_names)
    trainer.train()
    


@task(name="Validation Pipeline", description="Evaluates models, logs metrics to MLflow and selects the best one")
def validate(model_names, X_val, y_val):
    mlflow.set_experiment("fraud-detection")
    validator = Validator(model_names=model_names, X_val=X_val, y_val=y_val)
    results = validator.evaluate()

    for name, metrics in results.items():
        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_metrics(metrics)

    best = validator.best_model(results)

    client = MlflowClient()
    client.transition_model_version_stage(
        name=f"fraud-{best}",
        version=client.get_latest_versions(f"fraud-{best}")[0].version,
        stage="Production"
    )

    return best, results


@task(name="inference", description="Runs inference on the test set using the best model")
def evaluate(best, preprocessor, X_test):
    inference = Inference(best_model_name=best, preprocessor=preprocessor)
    return inference.predict(X_test)


@flow(name="ML Data Pipeline", description="Extract → Preprocess → Train → Validate → Inference")
def run_pipeline(extractor, query, target_col="is_fraud", **kwargs):
    df = extract(extractor, query, **kwargs)
    df = feature_engineer(df)
    X_train, X_test, X_val, preprocessor, feature_names = preprocess(df, target_col)
    train(models, X_train, preprocessor.y_train, preprocessor, feature_names)
    best, results = validate(list(models.keys()), X_val, preprocessor.y_val)
    evaluate(best, preprocessor, X_test)
    return best, preprocessor


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")

    # --- Option 1: CSV ---
    # best, preprocessor = run_pipeline(
    #     extractor=CSVExtractor(),
    #     query="data/transactions.csv",
    #     target_col="is_fraud"
    # )

    # --- Option 2: PostgreSQL ---
    # best, preprocessor = run_pipeline(
    #     extractor=PostgresExtractor(
    #         host="localhost", port=5432,
    #         database="ml_db", user="admin", password="secret"
    #     ),
    #     query="SELECT * FROM transactions WHERE created_at > %(start)s",
    #     target_col="is_fraud",
    #     params={"start": "2024-01-01"}
    # )

    # --- Option 3: BigQuery ---
    # best, preprocessor = run_pipeline(
    #     extractor=BigQueryExtractor(
    #         project_id="my-gcp-project",
    #         dataset="ml_dataset"
    #     ),
    #     query="""
    #         SELECT user_id, amount, label
    #         FROM `my-gcp-project.ml_dataset.transactions`
    #         WHERE DATE(created_at) >= '2024-01-01'
    #         LIMIT 1000000
    #     """,
    #     target_col="is_fraud"
    # )

    print("Swap any extractor — the pipeline code never changes.")