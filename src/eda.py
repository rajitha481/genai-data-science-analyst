import pandas as pd
import numpy as np


def numerical_summary(df):
    """
    Generate descriptive statistics for numerical columns.
    """
    numerical_df = df.select_dtypes(include=np.number)

    if numerical_df.empty:
        return pd.DataFrame()

    return numerical_df.describe().T


def categorical_summary(df):
    """
    Generate summary for categorical columns.
    """
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    results = []

    for column in categorical_columns:
        results.append({
            "Column": column,
            "Unique Values": df[column].nunique(),
            "Missing Values": df[column].isnull().sum(),
            "Most Frequent": df[column].mode().iloc[0]
            if not df[column].mode().empty else None
        })

    return pd.DataFrame(results)


def detect_outliers(df):
    """
    Detect outliers using the IQR method.
    """

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    results = []

    for column in numerical_columns:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[
            (df[column] < lower_bound) |
            (df[column] > upper_bound)
        ]

        results.append({
            "Column": column,
            "Outliers": len(outliers),
            "Outlier %": round(
                len(outliers) / len(df) * 100, 2
            )
        })

    return pd.DataFrame(results)