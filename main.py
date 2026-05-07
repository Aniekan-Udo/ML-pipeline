import subprocess 
import logging
from datetime import datetime

from extraction import CSVExtractor, PostgresExtractor, BigQueryExtractor
from model_config import  ModelFactory
from preprocessing import FeatureEngineer, Preprocessor
from train import MLTrainer
from validate import Validator
from inference import Inference
from prefect import flow, task
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.sklearn


from minio_storage import save, load, preprocessor_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any

from dotenv import load_dotenv
load_dotenv()

class PipelineConfig(BaseModel):
    """
    Validated pipeline configuration.
    Pydantic enforces types and constraints automatically.
    Fails immediately on instantiation — never mid-pipeline.
    """
    extractor: Any
    query: str
    target_col: str = "is_fraud"
    do_tuning: bool = False
    n_trials: int = Field(default=50, ge=1)        # ge=1 → must be >= 1
    mlflow_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment: str = "fraud-detection"

    model_config = {"arbitrary_types_allowed": True}  # allows extractor objects

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v

    @field_validator("mlflow_uri")
    @classmethod
    def valid_mlflow_uri(cls, v):
        if not v.startswith("http"):
            raise ValueError(f"Invalid MLflow URI: {v}")
        return v

    @model_validator(mode="after")
    def tuning_requires_trials(self):
        if self.do_tuning and self.n_trials < 1:
            raise ValueError("n_trials must be positive when tuning is enabled")
        return self
    


class MLflowClientSingleton:
    _instance = None

    @classmethod
    def get(cls) -> MlflowClient:
        if cls._instance is None:
            cls._instance = MlflowClient()
        return cls._instance
    
    
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
    mlflow.set_experiment("fraud-detection")

    preprocessor = Preprocessor(df)
    X_train, X_test, X_val, y_train, y_test, y_val = preprocessor.run(
        target_col=target_col
    )
    feature_names = preprocessor.get_feature_names_out()
   
    # Upload preprocessor to MinIO and log the URI as an MLflow artifact reference
    with mlflow.start_run(run_name="preprocessor"):
        uri = save(preprocessor, preprocessor_key())
        mlflow.log_param("preprocessor_minio_uri", uri)
        logger.info(f"Preprocessor saved to MinIO: {uri}")

    return X_train, X_test, X_val, y_train, y_test, y_val, preprocessor, feature_names


@task(name="Training", description="Trains all models and logs to MLflow")
def train(X_train, y_train, feature_names):
    trainer = MLTrainer(X_train=X_train, y_train=y_train, feature_names=feature_names)
    trainer.train()


@task(name="Validation Pipeline", description="Evaluates models, logs metrics to MLflow and selects the best one")
def validate(model_names, X_val, y_val, X_test, y_test):
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name="validation"):
        validator = Validator(model_names=model_names, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
        results = validator.evaluate()

        for name, metrics in results.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_metrics(metrics)

    best = validator.best_model(results)
    

    client = MLflowClientSingleton.get()
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
        pass

    # Use model aliases (replaces deprecated transition_model_version_stage)
    client.set_registered_model_alias(
        name=f"fraud-{best}",
        alias="production",
        version=latest_version
    )

    return best, results


@task(name="inference", description="Runs inference on the test set using the best model")
def evaluate(best, X_test):
    inference = Inference(best_model_name=best)
    return inference.predict(X_test)



@task(name="Hyperparameter Tuning", description="Tunes the hyperparameters of the best model")
def tune_best_model(best_model_name, X_train, y_train, X_val, y_val):
    mlflow.set_experiment("fraud-detection")
    model_factory = ModelFactory()

    tuner = model_factory.create_tuner(best_model_name)
    best_params, best_score, study, tuned_model = tuner.tune(
        X_train, y_train, X_val, y_val, n_trials=50
    )


    with mlflow.start_run(run_name=f"{best_model_name}_tuned"):
        mlflow.log_params(best_params)
        mlflow.log_metric("tuned_roc_auc", best_score)
        mlflow.sklearn.log_model(
            tuned_model,
            artifact_path="model",
            registered_model_name=f"fraud-{best_model_name}-tuned"
        )
        # Also persist tuned model to MinIO for direct loading
        from minio_storage import model_key
        uri = save(tuned_model, model_key(f"{best_model_name}_tuned"))
        mlflow.log_param("tuned_model_minio_uri", uri)
        logger.info(f"Tuned model saved to MinIO: {uri}")

    return tuned_model, best_params



@task(name="version data")
def version_data(df, source: str):
    if source != "csv":
        # Save extracted data locally first, then version it
        df.to_csv("data/transactions.csv", index=False)
    
    subprocess.run(["dvc", "add", "data/transactions.csv"], shell=True)
    subprocess.run(["git", "add", "data/transactions.csv.dvc"], shell=True)
    subprocess.run(["git", "commit", "-m",
        f"auto-version: {source} snapshot {datetime.now().isoformat()}"])
    subprocess.run(["dvc", "push"], shell=True)


@flow(name="ML Data Pipeline")
def run_pipeline(
    extractor,
    query,
    target_col: str = "is_fraud",
    do_tuning: bool = False,
    n_trials: int = 50,
    mlflow_uri: str = "http://localhost:5000",
    mlflow_experiment: str = "fraud-detection",
    **kwargs
):
    # Pydantic validates everything instantly
    config = PipelineConfig(
        extractor=extractor,
        query=query,
        target_col=target_col,
        do_tuning=do_tuning,
        n_trials=n_trials,
        mlflow_uri=mlflow_uri,
        mlflow_experiment=mlflow_experiment
    )

    # Rest of pipeline unchanged
    model_factory = ModelFactory()
    mlflow.set_tracking_uri(config.mlflow_uri)
    
    model_factory = ModelFactory()
    #version_data()

    df = extract(extractor, query, **kwargs)
    
    version_data(df, source=extractor.source)
    df = feature_engineer(df)

    X_train, X_test, X_val, y_train, y_test, y_val, preprocessor, feature_names = preprocess(df, target_col)

    train(X_train, y_train, feature_names)
    best, results = validate(model_factory.available_models(), X_val, y_val, X_test, y_test)

    tuned_model, best_params = None, {}
    if do_tuning:
        tuned_model, best_params = tune_best_model(
            best, X_train, y_train, X_val, y_val
        )

    predictions = evaluate(best, X_test)
    return best, preprocessor, predictions, results, best_params, tuned_model


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
    run_pipeline(
            extractor=BigQueryExtractor(
        project_id="fraud-detection-495412",
        dataset="fraud_dataset",
        location="US"
        ),

        query="""
            SELECT * FROM `fraud-detection-495412.fraud_dataset.transactions`
        """,

        target_col="is_fraud",
        
    )

    
    
    

