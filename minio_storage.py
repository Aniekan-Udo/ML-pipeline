"""
minio_storage.py
----------------
Thin helper for saving and loading joblib artifacts (models, preprocessors)
directly to/from MinIO (S3-compatible). All other MLflow artifact logging
(log_model, log_artifact) already goes to MinIO via the MLflow tracking
server — this module handles the joblib side-channel used by Validator and
Inference to load models by name.

Required environment variables (already set in docker-compose.yml):
    MLFLOW_S3_ENDPOINT_URL  - e.g. http://minio:9000
    AWS_ACCESS_KEY_ID       - MinIO root user
    AWS_SECRET_ACCESS_KEY   - MinIO root password

Optional:
    MINIO_BUCKET            - defaults to "mlflow-artifacts"
"""

import io
import os
import joblib
import boto3
from abc import ABC, abstractmethod
from botocore.client import Config
import pandas as pd

BUCKET = os.getenv("MINIO_BUCKET", "mlflow-artifacts")


# ─────────────────────────────────────────────
# PORT
# Stable contract. Pipeline depends on this.
# Never on MinIO or boto3 directly.
# Swap MinIO for S3/GCS = new adapter only.
# ─────────────────────────────────────────────

class StoragePort(ABC):
    """
    Port — stable storage contract.
    Core pipeline depends on this abstraction only.
    Never on MinIO, boto3, or any concrete storage directly.
    """

    @abstractmethod
    def save(self, obj, key: str) -> str:
        """Serialize and persist obj. Returns storage URI."""
        ...

    @abstractmethod
    def load(self, key: str):
        """Retrieve and deserialize artifact by key."""
        ...
    
    @abstractmethod
    def save_parquet(self, df: pd.DataFrame, key: str) -> str:
        """Serialize DataFrame as parquet and persist. Returns storage URI."""
    @abstractmethod
    def load_parquet(self, key: str) -> pd.DataFrame:
        """Retrieve parquet artifact by key and deserialize as DataFrame."""
        ...


# ─────────────────────────────────────────────
# ADAPTER
# Concrete MinIO implementation of StoragePort.
# Only place boto3 and MinIO exist in the codebase.
# ─────────────────────────────────────────────

class MinIOAdapter(StoragePort):
    """
    Adapter — MinIO implementation of StoragePort.
    All boto3 and MinIO details live here and nowhere else.
    """

    def __init__(self, bucket: str = BUCKET):
        self.bucket = bucket

    def _client(self):
        """Return a boto3 S3 client pointed at MinIO."""
        return boto3.client(
            "s3",
            endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

    def save(self, obj, key: str) -> str:
        """
        Serialize obj with joblib and upload to MinIO.

        Parameters
        ----------
        obj : any joblib-serializable object (model, preprocessor, ...)
        key : S3 key, e.g. "models/RandomForest.pkl"

        Returns
        -------
        s3_uri : str  -  s3://<bucket>/<key>
        """
        buf = io.BytesIO()
        joblib.dump(obj, buf)
        buf.seek(0)
        self._client().upload_fileobj(buf, self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def load(self, key: str):
        """
        Download and deserialize a joblib artifact from MinIO.

        Parameters
        ----------
        key : S3 key, e.g. "models/RandomForest.pkl"

        Returns
        -------
        Deserialized Python object.
        """
        buf = io.BytesIO()
        self._client().download_fileobj(self.bucket, key, buf)
        buf.seek(0)
        return joblib.load(buf)
    
    def save_parquet(self, df: pd.DataFrame, key: str) -> str:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        self._client().upload_fileobj(buf, self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def save_parquet(df, key: str) -> str:
        return StorageSingleton.get().save_parquet(df, key)
    
    def load_parquet(self, key: str) -> pd.DataFrame:
        import io
        buf = io.BytesIO()
        self._client().download_fileobj(self.bucket, key, buf)
        buf.seek(0)
        return pd.read_parquet(buf)
    
    def load_parquet(self, key: str) -> pd.DataFrame:
        return StorageSingleton.get().load_parquet(key)

# ─────────────────────────────────────────────
# SINGLETON
# One storage instance for the entire pipeline.
# Expensive boto3 client never duplicated.
# ─────────────────────────────────────────────

class StorageSingleton:
    _instance: StoragePort = None

    @classmethod
    def get(cls) -> StoragePort:
        if cls._instance is None:
            cls._instance = MinIOAdapter()
        return cls._instance


# ─────────────────────────────────────────────
# CONVENIENCE HELPERS
# Keeps key format consistent across modules.
# ─────────────────────────────────────────────

def model_key(name: str) -> str:
    return f"models/{name}.pkl"

def preprocessor_key() -> str:
    return "preprocessor/preprocessor.pkl"

def clip_boundary_key() -> str:
    return "clip_boundary/clip_boundary.pkl"


# ─────────────────────────────────────────────
# MODULE-LEVEL API
# Backward compatible — pipeline.py unchanged.
# All calls route through Singleton → Adapter → MinIO
# ─────────────────────────────────────────────

def save(obj, key: str) -> str:
    return StorageSingleton.get().save(obj, key)

def load(key: str):
    return StorageSingleton.get().load(key)