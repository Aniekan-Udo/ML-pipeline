from abc import ABC, abstractmethod
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    average_precision_score
)
import numpy as np
import pandas as pd
import logging

from minio_storage import StorageSingleton, model_key, preprocessor_key, clip_boundary_key
from preprocessing import InferenceFeatureEngineer 

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SENTINEL
# Marks a failed explainer load so we never
# retry MinIO on every request.
# ─────────────────────────────────────────────

_EXPLAINER_MISSING = object()


# ─────────────────────────────────────────────
# PORT
# ─────────────────────────────────────────────

class InferencePort(ABC):
    """
    Port — inference contract.
    Implement for sklearn, PyTorch, TensorFlow, or any future framework.
    app.py depends on this abstraction only — never on Inference directly.
    """

    @abstractmethod
    def predict(self, new_data) -> np.ndarray:
        """Returns binary predictions."""
        ...

    @abstractmethod
    def predict_proba(self, new_data) -> np.ndarray:
        """Returns fraud probability scores."""
        ...

    @abstractmethod
    def explain(self, new_data) -> dict:
        """Returns SHAP explanation for a single prediction."""
        ...

    @abstractmethod
    def evaluate(self, X_test, y_test) -> dict:
        """Returns full evaluation metrics on test set."""
        ...


# ─────────────────────────────────────────────
# IMPLEMENTATION
# ─────────────────────────────────────────────

class Inference(InferencePort):
    """
    Sklearn inference implementation.
    Loads model, preprocessor, and SHAP explainer from MinIO.
    Lazy loads — nothing loaded until first call.
    Explainer failure is cached — MinIO is only hit once even if
    the explainer file is missing.
    """

    def __init__(self, best_model_name: str):
        self.best_model_name = best_model_name
        self._model        = None           # lazy loaded
        self._preprocessor = None           # lazy loaded
        self._explainer    = None           # lazy loaded; _EXPLAINER_MISSING if not found
        self._clip_bounds  = None           # lazy loaded

    # ── Lazy loaders — load once, cache forever ───────────────────────────────

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading model: {self.best_model_name}")
            self._model = StorageSingleton.get().load(model_key(self.best_model_name))
        return self._model

    @property
    def preprocessor(self):
        if self._preprocessor is None:
            logger.info("Loading preprocessor")
            self._preprocessor = StorageSingleton.get().load(preprocessor_key())
        return self._preprocessor

    @property
    def explainer(self):
        if self._explainer is None:
            try:
                logger.info(f"Loading SHAP explainer: {self.best_model_name}")
                self._explainer = StorageSingleton.get().load(
                    f"explainers/{self.best_model_name}_explainer.pkl"
                )
                logger.info("SHAP explainer loaded successfully")
            except Exception as e:
                logger.warning(f"SHAP explainer not found — explain() unavailable: {e}")
                self._explainer = _EXPLAINER_MISSING  # cache the failure — never retry

        if self._explainer is _EXPLAINER_MISSING:
            return None
        return self._explainer

    @property
    def clip_bounds(self):
        if self._clip_bounds is None:
            try:
                logger.info("Loading clip bounds")
                from minio_storage import clip_boundary_key
                self._clip_bounds = StorageSingleton.get().load(clip_boundary_key())
            except Exception as e:
                logger.warning(f"Clip bounds not found in MinIO: {e}. Using fallback (no clipping).")
                self._clip_bounds = {"lower": 0.0, "upper": float('inf')}
        return self._clip_bounds

    # ── Internal transform ────────────────────────────────────────────────────

    def _transform(self, new_data: pd.DataFrame):
        # If already transformed (sparse matrix or numpy array), return as-is
        import scipy.sparse as sp
        if sp.issparse(new_data) or isinstance(new_data, np.ndarray):
            return new_data

        fe = InferenceFeatureEngineer(new_data, clip_bounds=self.clip_bounds)
        fe.cleaning()
        fe.transform()
        return self.preprocessor.transform(fe.df)

    # ── Public interface ──────────────────────────────────────────────────────

    def predict(self, new_data) -> np.ndarray:
        return self.model.predict(self._transform(new_data))

    def predict_proba(self, new_data) -> np.ndarray:
        return self.model.predict_proba(self._transform(new_data))[:, 1]

    def explain(self, new_data, feature_names: list | None = None) -> dict:
        """
        Returns local SHAP explanation for new_data.
        Answers: why was THIS transaction flagged?

        Returns top 10 features by absolute SHAP impact.
        Returns {"available": False} if explainer not available —
        never raises, never retries MinIO.
        """
        if self.explainer is None:
            return {"available": False, "reason": "SHAP explainer not loaded"}

        try:
            X = self._transform(new_data)
            shap_values = self.explainer.shap_values(X)

            # Binary classification RandomForest returns [class0, class1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Use feature names from preprocessor if not provided
            if feature_names is None:
                try:
                    feature_names = self.preprocessor.get_feature_names_out()
                except Exception:
                    feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

            # Sort by absolute impact — top 10
            impacts = list(zip(feature_names, shap_values[0]))
            top_features = sorted(impacts, key=lambda x: abs(x[1]), reverse=True)[:10]

            return {
                "available": True,
                "top_reasons": [
                    {
                        "feature":   f,
                        "impact":    round(float(v), 4),
                        "direction": "toward_fraud" if v > 0 else "away_from_fraud"
                    }
                    for f, v in top_features
                ]
            }

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return {"available": False, "reason": str(e)}

    def evaluate(self, X_test, y_test) -> dict:
        """
        Full evaluation on test set.
        Returns classification report, ROC-AUC, PR-AUC, F1.
        """
        y_pred  = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        report  = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc  = average_precision_score(y_test, y_proba)

        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

        return {
            "classification_report": report,
            "roc_auc":               round(roc_auc, 4),
            "pr_auc":                round(pr_auc, 4),
            "f1_score":              round(f1_score(y_test, y_pred), 4),
            "precision_curve":       precision.tolist(),
            "recall_curve":          recall.tolist(),
            "thresholds":            thresholds.tolist(),
        }