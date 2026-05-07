from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
import pandera.pandas as pa
from pandera import Column, DataFrameSchema, Check
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



raw_transaction_schema = DataFrameSchema(
    columns={
        "transaction_amount": Column(
            float,
            checks=[
                Check.greater_than(0, error="transaction_amount must be positive"),
            ],
            nullable=False,
        ),
        "transaction_time": Column(
            str,
            nullable=False,
        ),
        "device_type": Column(
            str,
            checks=Check.isin(["mobile", "desktop", "tablet", "pos_terminal"],
                              error="device_type must be mobile, desktop, tablet or pos_terminal"),
            nullable=False,
        ),
        "is_fraud": Column(
            int,
            checks=Check.isin([0, 1], error="is_fraud must be 0 or 1"),
            nullable=False,
        ),
    },
    coerce=True,         
    strict=False,        # allow extra columns beyond the declared ones
)

engineered_schema = DataFrameSchema(
    columns={
        "log_transaction_amount": Column(
            float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "transaction_year": Column(int, nullable=False),
        "transaction_month": Column(
            int,
            checks=Check.in_range(1, 12),
            nullable=False,
        ),
        "transaction_day": Column(
            int,
            checks=Check.in_range(1, 31),
            nullable=False,
        ),
        "transaction_hour": Column(
            int,
            checks=Check.in_range(0, 23),
            nullable=False,
        ),
        "transaction_day_of_week": Column(
            int,
            checks=Check.in_range(0, 6),
            nullable=False,
        ),
        "amount_by_device": Column(float, nullable=False),
    },
    coerce=True,
    strict=False,        # raw columns still present alongside engineered ones
)




class FeatureEngineerPort(ABC):
    @abstractmethod
    def cleaning(self, amount_col: str = 'transaction_amount') -> pd.DataFrame: ...

    @abstractmethod
    def transform(self) -> pd.DataFrame: ...


class PreprocessorPort(ABC):
    @abstractmethod
    def split(self, target_col: str = 'is_fraud'): ...

    @abstractmethod
    def preprocess(self): ...

    @abstractmethod
    def inbalance_handling(self): ...

    @abstractmethod
    def get_feature_names_out(self) -> list: ...

    @abstractmethod
    def transform(self, X: pd.DataFrame): ...

    @abstractmethod
    def run(self, target_col: str = 'is_fraud'):
        """Returns X_train, X_test, X_val, y_train, y_test, y_val"""
        ...



class FeatureEngineer(FeatureEngineerPort):
    """Sklearn/pandas feature engineering implementation with Pandera validation."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _validate_raw(self) -> None:
        """Validates raw input against the raw transaction schema."""
        try:
            raw_transaction_schema.validate(self.df, lazy=True)
            logger.info("Raw schema validation passed.")
        except pa.errors.SchemaErrors as e:
            logger.error(f"Raw schema validation failed:\n{e.failure_cases}")
            raise

    def _validate_engineered(self) -> None:
        """Validates that all engineered features were created correctly."""
        try:
            engineered_schema.validate(self.df, lazy=True)
            logger.info("Engineered schema validation passed.")
        except pa.errors.SchemaErrors as e:
            logger.error(f"Engineered schema validation failed:\n{e.failure_cases}")
            raise

    def cleaning(self, amount_col: str = 'transaction_amount') -> pd.DataFrame:
        self._validate_raw()                        

        self.df = self.df.drop_duplicates()
        self.df = self.df.dropna(how="all")
        self.df.columns = [c.lower().strip() for c in self.df.columns]

        for col in self.df.columns:
            if self.df[col].isnull().mean() > 0.10:
                self.df = self.df.dropna(subset=[col])

        lower = self.df[amount_col].quantile(0.01)
        upper = self.df[amount_col].quantile(0.99)
        self.df[amount_col] = self.df[amount_col].clip(lower, upper)
        return self.df

    def transform(self) -> pd.DataFrame:
        self.df['log_transaction_amount'] = np.log1p(self.df['transaction_amount'])
        self.df['transaction_time'] = pd.to_datetime(self.df['transaction_time'])
        self.df['transaction_year'] = self.df['transaction_time'].dt.year
        self.df['transaction_month'] = self.df['transaction_time'].dt.month
        self.df['transaction_day'] = self.df['transaction_time'].dt.day
        self.df['transaction_hour'] = self.df['transaction_time'].dt.hour
        self.df['transaction_day_of_week'] = self.df['transaction_time'].dt.dayofweek
        self.df['amount_by_device'] = self.df['transaction_amount'] * (
            self.df['device_type'] == 'mobile').astype(int)

        self._validate_engineered()                  
        return self.df


class Preprocessor(PreprocessorPort):
    """Sklearn preprocessing — returns numpy arrays."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.column_transformer = None

    def split(self, target_col: str = 'is_fraud'):
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.1, random_state=42)
        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test  = self.X_test.reset_index(drop=True)
        self.X_val   = self.X_val.reset_index(drop=True)

    def preprocess(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer

        numeric_features     = [col for col in self.X_train.columns if self.X_train[col].dtype in ['int64', 'float64']]
        categorical_features = [col for col in self.X_train.columns if self.X_train[col].dtype == 'object']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.column_transformer = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ])

        self.X_train = self.column_transformer.fit_transform(self.X_train)
        self.X_test  = self.column_transformer.transform(self.X_test)
        self.X_val   = self.column_transformer.transform(self.X_val)
        return self.X_train, self.X_test, self.X_val

    def inbalance_handling(self):
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        return self.X_train, self.y_train

    def get_feature_names_out(self) -> list:
        num_features = self.column_transformer.named_transformers_['num'].get_feature_names_out().tolist()
        cat_transformer = self.column_transformer.named_transformers_.get('cat')
        if cat_transformer is not None and hasattr(cat_transformer, 'named_steps'):
            cat_features = cat_transformer['onehot'].get_feature_names_out().tolist()
        else:
            cat_features = []
        return num_features + cat_features

    def transform(self, X: pd.DataFrame):
        """Transform new data using the already-fitted column_transformer."""
        return self.column_transformer.transform(X)

    def run(self, target_col: str = 'is_fraud'):
        self.split(target_col)
        X_train, X_test, X_val = self.preprocess()
        X_train, self.y_train = self.inbalance_handling()
        return X_train, X_test, X_val, self.y_train, self.y_test, self.y_val