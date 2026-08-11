import numpy as np
import pandas as pd
import shap


def get_feature_names(preprocessor):
    """
    Get feature names after preprocessing.
    """

    feature_names = []

    # Numerical features
    if "numerical" in preprocessor.named_transformers_:

        numerical_pipeline = (
            preprocessor
            .named_transformers_["numerical"]
        )

        numerical_features = (
            numerical_pipeline
            .named_steps["imputer"]
            .feature_names_in_
        )

        feature_names.extend(
            numerical_features
        )

    # Categorical features
    if "categorical" in preprocessor.named_transformers_:

        categorical_pipeline = (
            preprocessor
            .named_transformers_["categorical"]
        )

        encoder = (
            categorical_pipeline
            .named_steps["encoder"]
        )

        categorical_features = (
            encoder.get_feature_names_out()
        )

        feature_names.extend(
            categorical_features
        )

    return feature_names


def calculate_shap_values(
    pipeline,
    X
):
    """
    Calculate SHAP values for tree-based models.
    """

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    X_transformed = (
        preprocessor.transform(X)
    )

    # Convert sparse matrix if necessary
    if hasattr(
        X_transformed,
        "toarray"
    ):
        X_transformed = X_transformed.toarray()

    feature_names = get_feature_names(
        preprocessor
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_transformed
    )

    return (
        shap_values,
        X_transformed,
        feature_names
    )


def get_feature_importance(
    shap_values,
    feature_names
):
    """
    Calculate mean absolute SHAP importance.
    """

    # Handle binary classification
    if isinstance(
        shap_values,
        list
    ):

        values = shap_values[-1]

    else:

        values = shap_values

        # New SHAP versions can return
        # 3D arrays for classification
        if len(values.shape) == 3:

            values = values[:, :, -1]

    importance = np.abs(
        values
    ).mean(
        axis=0
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return importance_df