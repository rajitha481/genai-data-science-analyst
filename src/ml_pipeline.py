import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBClassifier, XGBRegressor


def detect_problem_type(y):
    """
    Automatically determine whether the target
    represents classification or regression.
    """

    if y.dtype == "object" or y.dtype.name == "category":
        return "Classification"

    unique_values = y.nunique()

    # Binary or small number of classes
    if unique_values <= 10:
        return "Classification"

    return "Regression"


def prepare_features(X):
    """
    Identify numerical and categorical columns.
    """

    numerical_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


def get_classification_models():
    """
    Return classification models.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss"
        )
    }

    return models


def get_regression_models():
    """
    Return regression models.
    """

    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            objective="reg:squarederror"
        )
    }

    return models


def train_classification_models(X, y):
    """
    Train and compare classification models.
    """

    preprocessor = prepare_features(X)

    models = get_classification_models()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    results = []
    trained_models = {}

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        y_pred = pipeline.predict(X_test)

        try:
            y_probability = pipeline.predict_proba(
                X_test
            )[:, 1]

            roc_auc = roc_auc_score(
                y_test,
                y_probability
            )

        except Exception:
            roc_auc = np.nan

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4)
            if not np.isnan(roc_auc)
            else None
        })

        trained_models[name] = pipeline

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    ).reset_index(drop=True)

    return (
        results_df,
        trained_models,
        X_test,
        y_test
    )


def train_regression_models(X, y):
    """
    Train and compare regression models.
    """

    preprocessor = prepare_features(X)

    models = get_regression_models()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    results = []
    trained_models = {}

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        y_pred = pipeline.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R² Score": round(r2, 4)
        })

        trained_models[name] = pipeline

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="R² Score",
        ascending=False
    ).reset_index(drop=True)

    return (
        results_df,
        trained_models,
        X_test,
        y_test
    )