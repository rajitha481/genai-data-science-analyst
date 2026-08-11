import pandas as pd
import numpy as np

from scipy.stats import (
    pearsonr,
    spearmanr,
    chi2_contingency,
    ttest_ind,
    f_oneway
)


def correlation_analysis(df):
    """
    Calculate Pearson and Spearman correlation
    for numerical variables.
    """

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    results = []

    for i in range(len(numerical_columns)):

        for j in range(i + 1, len(numerical_columns)):

            col1 = numerical_columns[i]
            col2 = numerical_columns[j]

            data = df[[col1, col2]].dropna()

            if len(data) < 3:
                continue

            pearson_corr, pearson_p = pearsonr(
                data[col1],
                data[col2]
            )

            spearman_corr, spearman_p = spearmanr(
                data[col1],
                data[col2]
            )

            results.append({
                "Variable 1": col1,
                "Variable 2": col2,
                "Pearson Correlation": round(
                    pearson_corr, 4
                ),
                "Pearson p-value": round(
                    pearson_p, 4
                ),
                "Spearman Correlation": round(
                    spearman_corr, 4
                ),
                "Spearman p-value": round(
                    spearman_p, 4
                )
            })

    return pd.DataFrame(results)


def chi_square_test(df, column1, column2):
    """
    Perform Chi-Square test of independence
    between two categorical variables.
    """

    contingency_table = pd.crosstab(
        df[column1],
        df[column2]
    )

    chi2, p_value, degrees_of_freedom, expected = (
        chi2_contingency(contingency_table)
    )

    if p_value < 0.05:
        conclusion = (
            "Statistically significant association "
            "between the variables."
        )
    else:
        conclusion = (
            "No statistically significant association "
            "was detected."
        )

    return {
        "Chi-Square Statistic": round(chi2, 4),
        "p-value": round(p_value, 6),
        "Degrees of Freedom": degrees_of_freedom,
        "Conclusion": conclusion
    }


def t_test(df, column, group_column, group1, group2):
    """
    Independent two-sample t-test.
    """

    data1 = df[
        df[group_column] == group1
    ][column].dropna()

    data2 = df[
        df[group_column] == group2
    ][column].dropna()

    statistic, p_value = ttest_ind(
        data1,
        data2,
        equal_var=False
    )

    if p_value < 0.05:
        conclusion = (
            "The group means are statistically "
            "significantly different."
        )
    else:
        conclusion = (
            "No statistically significant difference "
            "between the group means was detected."
        )

    return {
        "T-statistic": round(statistic, 4),
        "p-value": round(p_value, 6),
        "Conclusion": conclusion
    }


def anova_test(df, numerical_column, categorical_column):
    """
    One-way ANOVA test.
    """

    groups = []

    for group in df[categorical_column].dropna().unique():

        values = df[
            df[categorical_column] == group
        ][numerical_column].dropna()

        if len(values) > 1:
            groups.append(values)

    if len(groups) < 2:
        return None

    statistic, p_value = f_oneway(*groups)

    if p_value < 0.05:
        conclusion = (
            "At least one group has a statistically "
            "different mean."
        )
    else:
        conclusion = (
            "No statistically significant difference "
            "between group means was detected."
        )

    return {
        "F-statistic": round(statistic, 4),
        "p-value": round(p_value, 6),
        "Conclusion": conclusion
    }