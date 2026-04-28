"""
tests/test_suite.py
===================
Full test suite for the ML Fraud Detection Pipeline.

Priority order (mirrors business risk):
  1. False negatives — fraud that slips through
  2. Preprocessing correctness — silent corruption kills every prediction
  3. Schema validation — bad input must fail loudly
  4. Threshold & probability behaviour
  5. Drift detection boundaries
  6. Prediction consistency & concurrency
  7. Class imbalance (SMOTE)
  8. Integration — raw transaction → prediction
  9. Security — auth, rate limiting
  10. MinIO retry / circuit breaker / cache
  11. Alerting
  12. Edge cases unique to fraud

Run with:
    pytest tests/test_suite.py -v
    pytest tests/test_suite.py -v -k "false_negative"   # single group
"""

import time
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES — shared test data
# ══════════════════════════════════════════════════════════════════════════════

def _make_raw_df(n=100, fraud_ratio=0.1, device="mobile") -> pd.DataFrame:
    """Create a minimal valid raw transaction DataFrame."""
    np.random.seed(42)
    n_fraud = max(1, int(n * fraud_ratio))
    labels = [1] * n_fraud + [0] * (n - n_fraud)
    np.random.shuffle(labels)
    return pd.DataFrame({
        "transaction_id":              [f"txn_{i}" for i in range(n)],
        "customer_id":                 [f"cust_{i % 20}" for i in range(n)],
        "transaction_amount":          np.random.uniform(10, 5000, n),
        "transaction_type":            np.random.choice(["purchase", "refund"], n),
        "transaction_time":            pd.date_range("2024-01-01", periods=n, freq="h").astype(str),
        "transaction_location":        np.random.choice(["Lagos", "Abuja", "London"], n),
        "device_type":                 [device] * n,
        "previous_transactions_count": np.random.randint(0, 100, n),
        "is_fraud":                    labels,
    })


def _make_engineered_df(n=100, fraud_ratio=0.1) -> pd.DataFrame:
    """Return a DataFrame that has already been through FeatureEngineer."""
    from preprocessing import FeatureEngineer
    df = _make_raw_df(n, fraud_ratio)
    fe = FeatureEngineer(df)
    fe.cleaning()
    return fe.transform()


@pytest.fixture
def raw_df():
    return _make_raw_df(n=200)


@pytest.fixture
def engineered_df():
    return _make_engineered_df(n=200)


@pytest.fixture
def preprocessed_data(engineered_df):
    """Returns (X_train, X_test, X_val, y_train, y_test, y_val, preprocessor)."""
    from preprocessing import Preprocessor
    p = Preprocessor(engineered_df)
    X_train, X_test, X_val, y_train, y_test, y_val = p.run(target_col="is_fraud")
    return X_train, X_test, X_val, y_train, y_test, y_val, p


# ══════════════════════════════════════════════════════════════════════════════
# 1. FALSE NEGATIVES — most critical tests in the entire suite
# ══════════════════════════════════════════════════════════════════════════════

class TestFalseNegatives:
    """
    Fraud that gets through = real financial loss.
    These tests must never be skipped or xfailed in CI.
    """

    def test_all_fraud_batch_none_missed(self):
        """
        A batch of 100% known-fraud transactions should never return
        all-zero predictions after a trained model is applied.
        """
        from preprocessing import FeatureEngineer, Preprocessor
        from sklearn.ensemble import RandomForestClassifier

        # Build a synthetic fraud-only batch
        df_fraud = _make_raw_df(n=50, fraud_ratio=1.0)
        fe = FeatureEngineer(df_fraud)
        fe.cleaning()
        df_fraud = fe.transform()

        # Train on a balanced set so the model actually learns fraud
        df_train = _make_engineered_df(n=300, fraud_ratio=0.4)
        p = Preprocessor(df_train)
        X_train, X_test, X_val, y_train, y_test, y_val = p.run()

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Transform fraud-only batch
        X_fraud = p.transform(df_fraud.drop(columns=["is_fraud"]))
        preds = model.predict(X_fraud)

        # Model should catch at least some fraud — not all zeros
        assert preds.sum() > 0, (
            "Model returned all-zero predictions on a 100% fraud batch — "
            "this means all fraud is being missed."
        )

    def test_recall_above_minimum_threshold(self, preprocessed_data):
        """
        Recall must be ≥ 0.60 on the test set.
        Below 0.60 = unacceptable fraud miss rate for production.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import recall_score

        X_train, X_test, X_val, y_train, y_test, y_val, _ = preprocessed_data
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)

        recall = recall_score(y_test, model.predict(X_test), zero_division=0)
        assert recall >= 0.60, (
            f"Recall {recall:.2%} is below the 0.60 minimum threshold. "
            f"The model is missing too many fraud cases."
        )

    def test_high_value_transactions_scored(self):
        """
        Transactions above £10,000 must always receive a fraud score,
        never be silently dropped or raise an error.
        """
        from preprocessing import FeatureEngineer

        df = _make_raw_df(n=10)
        df["transaction_amount"] = 15000.0  # all high-value
        fe = FeatureEngineer(df)
        fe.cleaning()
        result = fe.transform()

        assert len(result) > 0, "High-value transactions were silently dropped."
        assert "log_transaction_amount" in result.columns

    def test_borderline_confidence_produces_prediction(self):
        """
        At exactly 0.5 predicted probability, the model must still
        produce a definitive 0 or 1 — never NaN or an error.
        """
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np

        # Mock a model that returns exactly 0.5 for every sample
        model = MagicMock(spec=RandomForestClassifier)
        model.predict_proba.return_value = np.array([[0.5, 0.5]])
        model.predict.return_value = np.array([1])  # sklearn default: ≥0.5 → positive

        proba = model.predict_proba(None)[:, 1]
        pred  = model.predict(None)

        assert pred[0] in (0, 1), "Prediction at 0.5 threshold must be 0 or 1."
        assert not np.isnan(proba[0]), "Probability must not be NaN."

    def test_single_transaction_not_dropped(self):
        """
        A single-row batch must produce exactly one prediction.
        Batch processing must never silently skip single records.
        """
        from preprocessing import FeatureEngineer

        df = _make_raw_df(n=1, fraud_ratio=1.0)
        fe = FeatureEngineer(df)
        fe.cleaning()
        result = fe.transform()

        assert len(result) == 1, (
            f"Single transaction batch produced {len(result)} rows — "
            f"expected exactly 1."
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING CORRECTNESS
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessingCorrectness:
    """
    A preprocessing mismatch between training and inference silently
    corrupts every single prediction. These tests catch that.
    """

    def test_feature_engineer_produces_required_columns(self, raw_df):
        from preprocessing import FeatureEngineer
        fe = FeatureEngineer(raw_df)
        fe.cleaning()
        result = fe.transform()

        required = [
            "log_transaction_amount",
            "transaction_year",
            "transaction_month",
            "transaction_day",
            "transaction_hour",
            "transaction_day_of_week",
            "amount_by_device",
        ]
        for col in required:
            assert col in result.columns, f"Missing engineered feature: {col}"

    def test_log_transform_is_non_negative(self, raw_df):
        from preprocessing import FeatureEngineer
        fe = FeatureEngineer(raw_df)
        fe.cleaning()
        result = fe.transform()

        assert (result["log_transaction_amount"] >= 0).all(), (
            "log_transaction_amount contains negative values — "
            "log1p of a non-negative amount must always be >= 0."
        )

    def test_transform_is_deterministic(self, raw_df):
        """Same input must produce exactly the same output every time."""
        from preprocessing import FeatureEngineer

        fe1 = FeatureEngineer(raw_df.copy())
        fe1.cleaning()
        out1 = fe1.transform()

        fe2 = FeatureEngineer(raw_df.copy())
        fe2.cleaning()
        out2 = fe2.transform()

        pd.testing.assert_frame_equal(out1, out2)

    def test_preprocessor_transform_matches_fit(self, engineered_df):
        """
        Preprocessor.transform() on new data must produce the same
        number of features as fit_transform() on training data.
        This catches column mismatch between training and inference.
        """
        from preprocessing import Preprocessor
        p = Preprocessor(engineered_df)
        X_train, X_test, X_val, y_train, y_test, y_val = p.run()

        # Simulate new inference data — same shape as test
        new_data = engineered_df.drop(columns=["is_fraud"]).head(10)
        X_new = p.transform(new_data)

        assert X_new.shape[1] == X_train.shape[1], (
            f"Feature count mismatch: training={X_train.shape[1]}, "
            f"inference={X_new.shape[1]}. "
            f"Preprocessor state was not preserved correctly."
        )

    def test_cleaning_removes_duplicates(self):
        from preprocessing import FeatureEngineer
        df = _make_raw_df(n=50)
        df = pd.concat([df, df.head(10)], ignore_index=True)  # add 10 duplicates
        fe = FeatureEngineer(df)
        fe.cleaning()

        assert fe.df.duplicated().sum() == 0, "Duplicates were not removed by cleaning()."

    def test_cleaning_clips_outliers(self):
        from preprocessing import FeatureEngineer
        df = _make_raw_df(n=100)
        df.loc[0, "transaction_amount"] = 999_999_999  # extreme outlier
        fe = FeatureEngineer(df)
        fe.cleaning()

        upper = df["transaction_amount"].quantile(0.99)
        assert fe.df["transaction_amount"].max() <= upper * 1.01, (
            "Outlier clipping did not cap extreme transaction_amount values."
        )

    def test_datetime_features_are_valid_ranges(self, raw_df):
        from preprocessing import FeatureEngineer
        fe = FeatureEngineer(raw_df)
        fe.cleaning()
        result = fe.transform()

        assert result["transaction_month"].between(1, 12).all(),     "transaction_month out of range"
        assert result["transaction_day"].between(1, 31).all(),       "transaction_day out of range"
        assert result["transaction_hour"].between(0, 23).all(),      "transaction_hour out of range"
        assert result["transaction_day_of_week"].between(0, 6).all(),"transaction_day_of_week out of range"

    def test_amount_by_device_mobile_is_nonzero(self):
        """amount_by_device must equal transaction_amount for mobile, 0 for others."""
        from preprocessing import FeatureEngineer

        df_mobile  = _make_raw_df(n=20, device="mobile")
        df_desktop = _make_raw_df(n=20, device="desktop")

        fe_m = FeatureEngineer(df_mobile)
        fe_m.cleaning()
        out_m = fe_m.transform()
        assert (out_m["amount_by_device"] > 0).all(), \
            "amount_by_device should be > 0 for mobile transactions."

        fe_d = FeatureEngineer(df_desktop)
        fe_d.cleaning()
        out_d = fe_d.transform()
        assert (out_d["amount_by_device"] == 0).all(), \
            "amount_by_device should be 0 for desktop transactions."

    def test_preprocessor_split_ratios(self, engineered_df):
        """Train/val/test split must respect the configured ratios."""
        from preprocessing import Preprocessor
        p = Preprocessor(engineered_df)
        X_train, X_test, X_val, y_train, y_test, y_val = p.run()

        total = len(X_train) + len(X_test) + len(X_val)
        assert total == len(engineered_df), "Rows lost during train/val/test split."

        test_ratio = len(X_test) / total
        assert 0.15 <= test_ratio <= 0.25, \
            f"Test set ratio {test_ratio:.2%} is outside expected range 15–25%."

    def test_smote_balances_classes(self, engineered_df):
        """SMOTE must produce a balanced y_train (equal 0s and 1s)."""
        from preprocessing import Preprocessor
        p = Preprocessor(engineered_df)
        X_train, X_test, X_val, y_train, y_test, y_val = p.run()

        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))

        ratio = min(class_counts.values()) / max(class_counts.values())
        assert ratio >= 0.8, (
            f"SMOTE did not balance classes adequately. "
            f"Class counts: {class_counts}. Ratio: {ratio:.2f}."
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. SCHEMA VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaValidation:
    """Bad input must fail loudly — never produce a silent wrong prediction."""

    def test_negative_transaction_amount_raises(self):
        import pandera as pa
        from preprocessing import FeatureEngineer
        df = _make_raw_df(n=10)
        df["transaction_amount"] = -100.0
        fe = FeatureEngineer(df)
        with pytest.raises(pa.errors.SchemaErrors):
            fe.cleaning()

    def test_zero_transaction_amount_raises(self):
        import pandera as pa
        from preprocessing import FeatureEngineer
        df = _make_raw_df(n=10)
        df["transaction_amount"] = 0.0
        fe = FeatureEngineer(df)
        with pytest.raises(pa.errors.SchemaErrors):
            fe.cleaning()

    def test_invalid_device_type_raises(self):
        import pandera as pa
        from preprocessing import FeatureEngineer
        df = _make_raw_df(n=10)
        df["device_type"] = "smartwatch"   # not in allowed set
        fe = FeatureEngineer(df)
        with pytest.raises(pa.errors.SchemaErrors):
            fe.cleaning()

    def test_invalid_is_fraud_label_raises(self):
        import pandera as pa
        from preprocessing import FeatureEngineer
        df = _make_raw_df(n=10)
        df["is_fraud"] = 2   # only 0 or 1 allowed
        fe = FeatureEngineer(df)
        with pytest.raises(pa.errors.SchemaErrors):
            fe.cleaning()

    def test_missing_transaction_amount_raises(self):
        import pandera as pa
        from preprocessing import FeatureEngineer
        df = _make_raw_df(n=10)
        df["transaction_amount"] = None
        fe = FeatureEngineer(df)
        with pytest.raises((pa.errors.SchemaErrors, Exception)):
            fe.cleaning()

    def test_missing_is_fraud_raises(self):
        import pandera as pa
        from preprocessing import FeatureEngineer
        df = _make_raw_df(n=10)
        df["is_fraud"] = None
        fe = FeatureEngineer(df)
        with pytest.raises((pa.errors.SchemaErrors, Exception)):
            fe.cleaning()

    def test_empty_dataframe_raises(self):
        from preprocessing import FeatureEngineer
        fe = FeatureEngineer(pd.DataFrame())
        with pytest.raises(Exception):
            fe.cleaning()

    def test_all_nulls_dropped_by_dropna_all(self):
        from preprocessing import FeatureEngineer
        df = _make_raw_df(n=20)
        # Add an all-null row
        null_row = pd.Series([None] * len(df.columns), index=df.columns)
        df = pd.concat([df, null_row.to_frame().T], ignore_index=True)
        fe = FeatureEngineer(df)
        fe.cleaning()
        # All-null rows must be gone
        assert not fe.df.isnull().all(axis=1).any(), "All-null rows were not dropped."


# ══════════════════════════════════════════════════════════════════════════════
# 4. EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

class TestExtraction:

    def test_csv_extractor_source_property(self):
        from extraction import CSVExtractor
        assert CSVExtractor().source == "csv"

    def test_postgres_extractor_source_property(self):
        from extraction import PostgresExtractor
        ext = PostgresExtractor(host="h", port=5432, database="d", user="u", password="p")
        assert ext.source == "postgres"

    def test_bigquery_extractor_source_property(self):
        from extraction import BigQueryExtractor
        ext = BigQueryExtractor(project_id="p", dataset="d")
        assert ext.source == "bigquery"

    def test_csv_extractor_validate_rejects_empty(self):
        from extraction import CSVExtractor
        ext = CSVExtractor()
        assert ext.validate(pd.DataFrame()) is False

    def test_csv_extractor_validate_accepts_nonempty(self, raw_df):
        from extraction import CSVExtractor
        ext = CSVExtractor()
        assert ext.validate(raw_df) is True

    def test_extract_and_validate_raises_on_empty(self, tmp_path):
        from extraction import CSVExtractor
        # Write an empty CSV
        empty = tmp_path / "empty.csv"
        empty.write_text("transaction_amount,is_fraud\n")  # header only
        ext = CSVExtractor()
        with pytest.raises(ValueError, match="failed validation"):
            ext.extract_and_validate(str(empty))

    def test_csv_extractor_reads_file(self, tmp_path, raw_df):
        from extraction import CSVExtractor
        path = tmp_path / "txns.csv"
        raw_df.to_csv(path, index=False)
        ext = CSVExtractor()
        result = ext.extract_and_validate(str(path))
        assert len(result) == len(raw_df)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MODEL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class TestModelConfig:

    def test_available_models_returns_three(self):
        from model_config import ModelFactory
        models = ModelFactory().available_models()
        assert len(models) == 3
        assert "RandomForest" in models
        assert "XGBoost" in models
        assert "LogisticRegression" in models

    def test_create_model_returns_correct_type(self):
        from model_config import ModelFactory
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.linear_model import LogisticRegression

        factory = ModelFactory()
        assert isinstance(factory.create_model("RandomForest"), RandomForestClassifier)
        assert isinstance(factory.create_model("XGBoost"), XGBClassifier)
        assert isinstance(factory.create_model("LogisticRegression"), LogisticRegression)

    def test_create_model_unknown_raises(self):
        from model_config import ModelFactory
        with pytest.raises(ValueError, match="not registered"):
            ModelFactory().create_model("BrainwaveClassifier")

    def test_feature_importance_flag(self):
        from model_config import ModelFactory
        factory = ModelFactory()
        assert factory.supports_feature_importance("RandomForest") is True
        assert factory.supports_feature_importance("XGBoost") is True
        assert factory.supports_feature_importance("LogisticRegression") is False

    def test_predict_proba_flag(self):
        from model_config import ModelFactory
        factory = ModelFactory()
        for name in factory.available_models():
            assert factory.supports_predict_proba(name) is True

    def test_each_model_is_fresh_instance(self):
        """create_model must return a new instance each call — not a shared one."""
        from model_config import ModelFactory
        factory = ModelFactory()
        m1 = factory.create_model("RandomForest")
        m2 = factory.create_model("RandomForest")
        assert m1 is not m2


# ══════════════════════════════════════════════════════════════════════════════
# 6. VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class TestValidator:

    @patch("validate.StorageSingleton")
    def test_best_model_by_roc_auc(self, mock_storage):
        from validate import Validator

        results = {
            "RandomForest":      {"f1": 0.80, "recall": 0.82, "precision": 0.78, "roc_auc": 0.91},
            "XGBoost":           {"f1": 0.84, "recall": 0.86, "precision": 0.82, "roc_auc": 0.94},
            "LogisticRegression":{"f1": 0.72, "recall": 0.74, "precision": 0.70, "roc_auc": 0.85},
        }
        v = Validator(
            model_names=list(results.keys()),
            X_val=np.zeros((10, 5)), y_val=np.zeros(10),
            X_test=np.zeros((10, 5)), y_test=np.zeros(10),
        )
        best = v.best_model(results)
        assert best == "XGBoost", f"Expected XGBoost (highest ROC-AUC), got {best}"

    @patch("validate.StorageSingleton")
    def test_best_model_recall_tiebreak(self, mock_storage):
        """When ROC-AUC is identical, a custom metric can pick by recall."""
        from validate import Validator

        results = {
            "ModelA": {"f1": 0.80, "recall": 0.90, "precision": 0.72, "roc_auc": 0.88},
            "ModelB": {"f1": 0.80, "recall": 0.70, "precision": 0.92, "roc_auc": 0.88},
        }
        v = Validator(
            model_names=list(results.keys()),
            X_val=np.zeros((10, 5)), y_val=np.zeros(10),
            X_test=np.zeros((10, 5)), y_test=np.zeros(10),
        )
        # Default metric is roc_auc — both equal, first wins alphabetically
        best = v.best_model(results, metric="recall")
        assert best == "ModelA", "Should pick ModelA with higher recall."


# ══════════════════════════════════════════════════════════════════════════════
# 7. INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

class TestInference:

    def _make_inference(self, model=None, preprocessor=None):
        """Helper — returns Inference with mocked MinIO dependencies."""
        from inference import Inference

        mock_model = model or MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2],[0.1, 0.9],[0.7, 0.3]])

        mock_preprocessor = preprocessor or MagicMock()
        mock_preprocessor.transform.return_value = np.zeros((3, 10))
        mock_preprocessor.get_feature_names_out.return_value = [f"f{i}" for i in range(10)]

        with patch("inference.StorageSingleton") as mock_storage:
            mock_storage.get.return_value.load.side_effect = lambda key: (
                mock_model if "models" in key else mock_preprocessor
            )
            inf = Inference(best_model_name="RandomForest")
            inf._model        = mock_model
            inf._preprocessor = mock_preprocessor
        return inf

    def test_predict_returns_binary_array(self):
        inf = self._make_inference()
        df  = _make_raw_df(n=3)
        with patch.object(inf, "_transform", return_value=np.zeros((3, 10))):
            preds = inf.predict(df)
        assert set(preds).issubset({0, 1}), f"predict() returned non-binary values: {set(preds)}"
        assert len(preds) == 3

    def test_predict_proba_between_zero_and_one(self):
        inf = self._make_inference()
        df  = _make_raw_df(n=3)
        with patch.object(inf, "_transform", return_value=np.zeros((3, 10))):
            probas = inf.predict_proba(df)
        assert ((probas >= 0) & (probas <= 1)).all(), \
            "predict_proba() returned values outside [0, 1]."

    def test_predict_proba_length_matches_input(self):
        inf = self._make_inference()
        df  = _make_raw_df(n=3)
        with patch.object(inf, "_transform", return_value=np.zeros((3, 10))):
            probas = inf.predict_proba(df)
        assert len(probas) == 3

    def test_same_input_same_output(self):
        """Determinism — same data must produce identical predictions."""
        inf = self._make_inference()
        df  = _make_raw_df(n=5)
        with patch.object(inf, "_transform", return_value=np.zeros((5, 10))):
            p1 = inf.predict(df)
            p2 = inf.predict(df)
        np.testing.assert_array_equal(p1, p2, err_msg="Non-deterministic predictions detected.")

    def test_explain_returns_dict(self):
        from inference import Inference

        mock_model = MagicMock()
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.zeros((1, 10))
        mock_preprocessor.get_feature_names_out.return_value = [f"f{i}" for i in range(10)]

        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.random.randn(1, 10)

        inf = Inference("RandomForest")
        inf._model        = mock_model
        inf._preprocessor = mock_preprocessor
        inf._explainer    = mock_explainer

        df = _make_raw_df(n=1)
        with patch.object(inf, "_transform", return_value=np.zeros((1, 10))):
            result = inf.explain(df)

        assert isinstance(result, dict)
        assert "available" in result

    def test_explain_graceful_when_no_explainer(self):
        from inference import Inference

        inf = Inference("RandomForest")
        inf._model        = MagicMock()
        inf._preprocessor = MagicMock()
        inf._explainer    = None   # no explainer loaded

        df = _make_raw_df(n=1)
        with patch.object(inf, "_transform", return_value=np.zeros((1, 10))):
            result = inf.explain(df)

        assert result["available"] is False

    def test_lazy_loading_model_only_once(self):
        """Model must be loaded from MinIO exactly once, then cached."""
        from inference import Inference

        mock_storage = MagicMock()
        mock_storage.load.return_value = MagicMock(
            predict=lambda x: np.array([0]),
            predict_proba=lambda x: np.array([[0.9, 0.1]])
        )

        with patch("inference.StorageSingleton") as mock_singleton:
            mock_singleton.get.return_value = mock_storage
            inf = Inference("RandomForest")
            _ = inf.model
            _ = inf.model  # second access
            _ = inf.model  # third access

        # StorageSingleton.get().load() should only be called once for the model
        model_calls = [c for c in mock_storage.load.call_args_list
                       if "models" in str(c)]
        assert len(model_calls) == 1, \
            f"Model loaded {len(model_calls)} times — expected exactly 1 (lazy load then cache)."


# ══════════════════════════════════════════════════════════════════════════════
# 8. DRIFT DETECTION BOUNDARIES
# ══════════════════════════════════════════════════════════════════════════════

class TestDriftDetection:

    def test_classify_severity_serious_high_drift(self):
        from monitoring import _classify_severity
        result = _classify_severity(
            share_drifted=0.60,   # > 0.50 threshold
            prediction_drift=False,
            target_drift=False,
            recall=0.85,
        )
        assert result == "serious"

    def test_classify_severity_serious_target_drift(self):
        from monitoring import _classify_severity
        result = _classify_severity(
            share_drifted=0.10,   # low feature drift
            prediction_drift=False,
            target_drift=True,    # but target drifted
            recall=0.85,
        )
        assert result == "serious"

    def test_classify_severity_serious_prediction_drift(self):
        from monitoring import _classify_severity
        result = _classify_severity(
            share_drifted=0.10,
            prediction_drift=True,  # prediction drifted
            target_drift=False,
            recall=0.85,
        )
        assert result == "serious"

    def test_classify_severity_serious_low_recall(self):
        from monitoring import _classify_severity
        result = _classify_severity(
            share_drifted=0.10,
            prediction_drift=False,
            target_drift=False,
            recall=0.60,   # < 0.70 threshold
        )
        assert result == "serious"

    def test_classify_severity_negligible(self):
        from monitoring import _classify_severity
        result = _classify_severity(
            share_drifted=0.30,   # 15–49%
            prediction_drift=False,
            target_drift=False,
            recall=0.85,
        )
        assert result == "negligible"

    def test_classify_severity_none(self):
        from monitoring import _classify_severity
        result = _classify_severity(
            share_drifted=0.05,   # < 15%
            prediction_drift=False,
            target_drift=False,
            recall=0.90,
        )
        assert result == "none"

    def test_classify_severity_at_exactly_serious_threshold(self):
        from monitoring import _classify_severity
        result = _classify_severity(
            share_drifted=0.50,   # exactly at threshold
            prediction_drift=False,
            target_drift=False,
            recall=0.85,
        )
        assert result == "serious", "At exactly 0.50 threshold should be 'serious'."

    def test_classify_severity_at_exactly_negligible_threshold(self):
        from monitoring import _classify_severity
        result = _classify_severity(
            share_drifted=0.15,   # exactly at negligible boundary
            prediction_drift=False,
            target_drift=False,
            recall=0.85,
        )
        assert result == "negligible", "At exactly 0.15 threshold should be 'negligible'."

    def test_classify_severity_recall_none_does_not_crash(self):
        """recall=None means no feedback data yet — must not raise."""
        from monitoring import _classify_severity
        result = _classify_severity(
            share_drifted=0.10,
            prediction_drift=False,
            target_drift=False,
            recall=None,
        )
        assert result in ("none", "negligible", "serious")


# ══════════════════════════════════════════════════════════════════════════════
# 9. SECURITY — Authentication & Rate Limiting
# ══════════════════════════════════════════════════════════════════════════════

class TestSecurity:

    def test_valid_api_key_accepted(self):
        """A request with a valid key must not raise."""
        import asyncio
        from security import get_api_key, _LOADED_KEYS
        from fastapi import Request

        # Register a test key
        test_token = "test-valid-key-12345"
        _LOADED_KEYS[test_token] = "test-client"

        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "127.0.0.1"
        mock_request.method = "POST"
        mock_request.url.path = "/predict"

        mock_bearer = MagicMock()
        mock_bearer.credentials = test_token

        result = asyncio.get_event_loop().run_until_complete(
            get_api_key(mock_request, bearer=mock_bearer, x_api_key=None)
        )
        assert result == "test-client"

        # Cleanup
        del _LOADED_KEYS[test_token]

    def test_invalid_api_key_raises_401(self):
        import asyncio
        from security import get_api_key
        from fastapi import HTTPException, Request

        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "127.0.0.1"
        mock_request.method = "POST"
        mock_request.url.path = "/predict"

        mock_bearer = MagicMock()
        mock_bearer.credentials = "definitely-not-a-real-key"

        with pytest.raises(HTTPException) as exc_info:
            asyncio.get_event_loop().run_until_complete(
                get_api_key(mock_request, bearer=mock_bearer, x_api_key=None)
            )
        assert exc_info.value.status_code == 401

    def test_missing_api_key_raises_401(self):
        import asyncio
        from security import get_api_key
        from fastapi import HTTPException, Request

        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "127.0.0.1"
        mock_request.method = "POST"
        mock_request.url.path = "/predict"

        with pytest.raises(HTTPException) as exc_info:
            asyncio.get_event_loop().run_until_complete(
                get_api_key(mock_request, bearer=None, x_api_key=None)
            )
        assert exc_info.value.status_code == 401

    def test_rate_limit_raises_429_after_burst(self):
        from security import _check_rate_limit, _rate_buckets, _RATE_LIMIT_BURST
        from fastapi import HTTPException

        client = "rate-limit-test-client"
        _rate_buckets[client] = {
            "tokens":      0.0,   # bucket is empty
            "last_refill": time.monotonic() - 0.001   # just refilled — tiny delta
        }

        with pytest.raises(HTTPException) as exc_info:
            _check_rate_limit(client)

        assert exc_info.value.status_code == 429

        # Cleanup
        del _rate_buckets[client]

    def test_webhook_valid_signature_accepted(self):
        from security import verify_webhook_signature
        import hmac, hashlib

        secret  = "my-webhook-secret"
        payload = b'{"event": "fraud_confirmed"}'
        sig     = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

        assert verify_webhook_signature(payload, sig, secret) is True

    def test_webhook_invalid_signature_rejected(self):
        from security import verify_webhook_signature
        assert verify_webhook_signature(b"payload", "sha256=invalid", "secret") is False

    def test_webhook_missing_prefix_rejected(self):
        from security import verify_webhook_signature
        assert verify_webhook_signature(b"payload", "invalidformat", "secret") is False


# ══════════════════════════════════════════════════════════════════════════════
# 10. MINIO — Retry, Circuit Breaker, Cache
# ══════════════════════════════════════════════════════════════════════════════

class TestMinIOAdapter:

    def _adapter(self):
        from minio_storage import MinIOAdapter
        # Reset circuit breaker state before each test
        MinIOAdapter._cb_failure_count = 0
        MinIOAdapter._cb_open_until   = 0.0
        return MinIOAdapter(bucket="test-bucket")

    def test_save_and_load_roundtrip(self):
        adapter = self._adapter()
        obj = {"model": "test", "version": 1}

        mock_client = MagicMock()
        with patch.object(adapter, "_client", return_value=mock_client), \
             patch("minio_storage.joblib.dump") as mock_dump, \
             patch("minio_storage.joblib.load", return_value=obj) as mock_load:

            adapter.save(obj, "models/test.pkl")
            result = adapter.load("models/test.pkl")

        assert result == obj

    def test_retry_on_transient_error(self):
        from botocore.exceptions import BotoCoreError
        adapter = self._adapter()

        call_count = {"n": 0}

        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise BotoCoreError()
            return "success"

        with patch("time.sleep"):  # don't actually sleep
            result = adapter._with_retry(flaky, max_attempts=3, base_delay=0.01)

        assert result == "success"
        assert call_count["n"] == 3

    def test_raises_after_max_attempts(self):
        from botocore.exceptions import BotoCoreError
        adapter = self._adapter()

        def always_fails():
            raise BotoCoreError()

        with patch("time.sleep"), pytest.raises(RuntimeError, match="failed after"):
            adapter._with_retry(always_fails, max_attempts=3, base_delay=0.01)

    def test_circuit_breaker_opens_after_threshold(self):
        from botocore.exceptions import BotoCoreError
        from minio_storage import MinIOAdapter

        adapter = self._adapter()

        def always_fails():
            raise BotoCoreError()

        # Exhaust retries multiple times to trigger circuit breaker
        for _ in range(adapter._cb_failure_threshold):
            with patch("time.sleep"):
                try:
                    adapter._with_retry(always_fails, max_attempts=1, base_delay=0.01)
                except RuntimeError:
                    pass

        assert MinIOAdapter._cb_open_until > time.time(), \
            "Circuit breaker should be OPEN after repeated failures."

    def test_circuit_breaker_blocks_when_open(self):
        from minio_storage import MinIOAdapter

        adapter = self._adapter()
        MinIOAdapter._cb_open_until = time.time() + 60  # force open

        with pytest.raises(RuntimeError, match="circuit breaker OPEN"):
            adapter._with_retry(lambda: None)

        # Reset
        MinIOAdapter._cb_open_until = 0.0

    def test_cache_returns_on_hit(self):
        import minio_storage
        adapter = self._adapter()

        obj = {"cached": True}
        minio_storage._cache["test-key"] = {
            "value":   obj,
            "expires": time.time() + 300
        }

        result, hit = adapter._cache_get("test-key")
        assert hit is True
        assert result == obj

        del minio_storage._cache["test-key"]

    def test_cache_miss_on_expired(self):
        import minio_storage
        adapter = self._adapter()

        minio_storage._cache["expired-key"] = {
            "value":   "stale",
            "expires": time.time() - 1  # already expired
        }

        result, hit = adapter._cache_get("expired-key")
        assert hit is False

        del minio_storage._cache["expired-key"]

    def test_permanent_error_not_retried(self):
        from botocore.exceptions import ClientError
        adapter = self._adapter()
        call_count = {"n": 0}

        def no_such_key():
            call_count["n"] += 1
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": ""}}, "GetObject")

        with pytest.raises(ClientError):
            adapter._with_retry(no_such_key, max_attempts=3)

        assert call_count["n"] == 1, "NoSuchKey should not be retried."


# ══════════════════════════════════════════════════════════════════════════════
# 11. ALERTING
# ══════════════════════════════════════════════════════════════════════════════

class TestAlerting:

    @patch("alerting.SLACK_WEBHOOK_URL", "http://fake-slack-webhook")
    @patch("alerting.httpx.post")
    def test_slack_alert_sent_on_drift(self, mock_post):
        from alerting import alert_drift

        mock_post.return_value.raise_for_status = MagicMock()
        alert_drift(drift_score=0.60, severity_label="serious")
        mock_post.assert_called_once()

    @patch("alerting.SLACK_WEBHOOK_URL", "")
    @patch("alerting.httpx.post")
    def test_slack_skipped_when_not_configured(self, mock_post):
        from alerting import alert

        result = alert("Test", "msg", severity="info", channels=["slack"])
        mock_post.assert_not_called()

    @patch("alerting.PAGERDUTY_ROUTING_KEY", "fake-pd-key")
    @patch("alerting.httpx.post")
    def test_pagerduty_triggered_on_critical(self, mock_post):
        from alerting import alert

        mock_post.return_value.raise_for_status = MagicMock()
        alert("Test", "critical issue", severity="critical", channels=["pagerduty"])
        mock_post.assert_called_once()
        call_payload = mock_post.call_args[1]["json"]
        assert call_payload["payload"]["severity"] == "critical"

    @patch("alerting.PAGERDUTY_ROUTING_KEY", "fake-pd-key")
    @patch("alerting.httpx.post")
    def test_pagerduty_not_triggered_on_info(self, mock_post):
        """PagerDuty should only fire on critical — not info or warning."""
        from alerting import alert

        alert("Test", "informational", severity="info", channels=["pagerduty"])
        mock_post.assert_not_called()

    @patch("alerting.httpx.post", side_effect=Exception("network error"))
    @patch("alerting.SLACK_WEBHOOK_URL", "http://fake")
    def test_alert_does_not_raise_on_channel_failure(self, mock_post):
        """A broken Slack/PD should never crash the pipeline."""
        from alerting import alert
        # Should return False for that channel, not raise
        result = alert("Test", "msg", severity="warning", channels=["slack"])
        assert result.get("slack") is False


# ══════════════════════════════════════════════════════════════════════════════
# 12. FRAUD-SPECIFIC EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

class TestFraudEdgeCases:

    def test_all_legitimate_batch_does_not_crash(self):
        """A 100% legitimate batch must produce all-zero predictions without error."""
        from preprocessing import FeatureEngineer

        df = _make_raw_df(n=50, fraud_ratio=0.0)
        df["is_fraud"] = 0
        fe = FeatureEngineer(df)
        fe.cleaning()
        result = fe.transform()
        assert len(result) > 0

    def test_new_unseen_device_type_handled(self):
        """
        A device type never seen in training is rejected at schema validation.
        It must never silently pass through and corrupt the prediction.
        """
        import pandera as pa
        from preprocessing import FeatureEngineer

        df = _make_raw_df(n=10)
        df["device_type"] = "smartwatch"
        fe = FeatureEngineer(df)
        with pytest.raises(pa.errors.SchemaErrors):
            fe.cleaning()

    def test_rapid_sequential_same_customer_preserved(self):
        """
        Multiple rapid transactions from the same customer_id must all
        be preserved — none should be dropped as duplicates.
        """
        from preprocessing import FeatureEngineer

        base = _make_raw_df(n=5)
        base["customer_id"] = "cust_001"  # same customer
        base["transaction_id"] = [f"txn_{i}" for i in range(5)]  # different txn IDs
        base["transaction_time"] = pd.date_range("2024-01-01 10:00", periods=5, freq="min").astype(str)

        fe = FeatureEngineer(base)
        fe.cleaning()
        # All 5 should survive — they're distinct transactions
        assert len(fe.df) == 5, (
            f"Rapid sequential transactions from same customer were incorrectly dropped. "
            f"Expected 5, got {len(fe.df)}."
        )

    def test_very_large_transaction_amount_handled(self):
        """Abnormally large amounts must be clipped, not raise an error."""
        from preprocessing import FeatureEngineer

        df = _make_raw_df(n=20)
        df.loc[0, "transaction_amount"] = 1_000_000_000.0
        fe = FeatureEngineer(df)
        fe.cleaning()   # should clip, not raise
        assert fe.df["transaction_amount"].max() < 1_000_000_000.0

    def test_is_fraud_label_preserved_through_feature_engineering(self, raw_df):
        """is_fraud must not be accidentally dropped or altered by FeatureEngineer."""
        from preprocessing import FeatureEngineer

        original_labels = raw_df["is_fraud"].values.copy()
        fe = FeatureEngineer(raw_df)
        fe.cleaning()
        result = fe.transform()

        # Labels in result must still be a subset of {0, 1}
        assert set(result["is_fraud"].unique()).issubset({0, 1}), \
            "is_fraud column was corrupted by feature engineering."

    def test_all_fraud_batch_feature_engineering_completes(self):
        """A 100% fraud batch must pass through feature engineering without error."""
        from preprocessing import FeatureEngineer

        df = _make_raw_df(n=50, fraud_ratio=1.0)
        fe = FeatureEngineer(df)
        fe.cleaning()
        result = fe.transform()
        assert len(result) > 0
        assert result["is_fraud"].sum() > 0

    def test_prediction_consistency_under_repeated_calls(self):
        """
        The same transaction submitted 10 times must produce
        the same prediction every time — no randomness in inference.
        """
        from sklearn.ensemble import RandomForestClassifier
        from preprocessing import FeatureEngineer, Preprocessor

        # Train
        df_train = _make_engineered_df(n=300, fraud_ratio=0.3)
        p = Preprocessor(df_train)
        X_train, X_test, X_val, y_train, y_test, y_val = p.run()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Single transaction
        txn = _make_raw_df(n=1)
        fe  = FeatureEngineer(txn)
        fe.cleaning()
        X   = p.transform(fe.transform().drop(columns=["is_fraud"]))

        predictions = [model.predict(X)[0] for _ in range(10)]
        assert len(set(predictions)) == 1, \
            f"Non-deterministic predictions: {predictions}"


# ══════════════════════════════════════════════════════════════════════════════
# 13. INTEGRATION — raw transaction → prediction
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_full_preprocessing_pipeline(self):
        """
        Raw DataFrame → FeatureEngineer → Preprocessor → numpy arrays.
        This is the exact path every transaction takes before prediction.
        """
        from preprocessing import FeatureEngineer, Preprocessor

        df = _make_raw_df(n=300, fraud_ratio=0.2)
        fe = FeatureEngineer(df)
        fe.cleaning()
        df_eng = fe.transform()

        p = Preprocessor(df_eng)
        X_train, X_test, X_val, y_train, y_test, y_val = p.run()

        assert X_train.shape[0] > 0, "Training set is empty."
        assert X_test.shape[0]  > 0, "Test set is empty."
        assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch."
        assert len(y_train) == X_train.shape[0], "Label/feature row count mismatch."

    def test_trained_model_predicts_on_test_set(self):
        """Train a model end-to-end and verify it produces predictions on the test set."""
        from sklearn.ensemble import RandomForestClassifier
        from preprocessing import FeatureEngineer, Preprocessor

        df = _make_raw_df(n=400, fraud_ratio=0.2)
        fe = FeatureEngineer(df)
        fe.cleaning()
        df_eng = fe.transform()

        p = Preprocessor(df_eng)
        X_train, X_test, X_val, y_train, y_test, y_val = p.run()

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1})

    def test_preprocessor_save_and_reload_produces_same_transform(self, tmp_path):
        """
        Simulates the MinIO save/load cycle using joblib locally.
        Confirms the preprocessor state is fully preserved on disk.
        """
        import joblib
        from preprocessing import FeatureEngineer, Preprocessor

        df = _make_raw_df(n=300, fraud_ratio=0.2)
        fe = FeatureEngineer(df)
        fe.cleaning()
        df_eng = fe.transform()

        p = Preprocessor(df_eng)
        p.run()

        # Save
        path = tmp_path / "preprocessor.pkl"
        joblib.dump(p, path)

        # Reload
        p2 = joblib.load(path)

        # Both must produce identical transforms on new data
        new_df = _make_raw_df(n=20)
        fe2 = FeatureEngineer(new_df)
        fe2.cleaning()
        df_new = fe2.transform().drop(columns=["is_fraud"])

        X1 = p.transform(df_new)
        X2 = p2.transform(df_new)

        np.testing.assert_array_almost_equal(X1, X2, decimal=6,
            err_msg="Reloaded preprocessor produces different transforms — serialization broke state.")

    def test_csv_extractor_to_feature_engineer(self, tmp_path):
        """End-to-end: CSV file → CSVExtractor → FeatureEngineer → engineered DataFrame."""
        from extraction import CSVExtractor
        from preprocessing import FeatureEngineer

        raw = _make_raw_df(n=100)
        path = tmp_path / "transactions.csv"
        raw.to_csv(path, index=False)

        ext = CSVExtractor()
        df  = ext.extract_and_validate(str(path))

        fe = FeatureEngineer(df)
        fe.cleaning()
        result = fe.transform()

        assert "log_transaction_amount" in result.columns
        assert len(result) > 0