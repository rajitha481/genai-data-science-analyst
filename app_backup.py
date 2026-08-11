import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.eda import (
    numerical_summary,
    categorical_summary,
    detect_outliers
)
from src.statistics import (
    correlation_analysis,
    chi_square_test,
    t_test,
    anova_test
)
from src.ml_pipeline import (
    detect_problem_type,
    train_classification_models,
    train_regression_models
)
from src.explainability import (
    calculate_shap_values,
    get_feature_importance
)
from src.llm_analyst import generate_analysis
# ==================================================
# SESSION STATE
# ==================================================

if "best_model_pipeline" not in st.session_state:
    st.session_state.best_model_pipeline = None

if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None

if "X_test" not in st.session_state:
    st.session_state.X_test = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None

if "ml_results" not in st.session_state:
    st.session_state.ml_results = None

if "problem_type" not in st.session_state:
    st.session_state.problem_type = None

if "shap_importance" not in st.session_state:
    st.session_state.shap_importance = None
if "best_model_pipeline" not in st.session_state:
    st.session_state.best_model_pipeline = None

if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None

if "X_test" not in st.session_state:
    st.session_state.X_test = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="GenAI Data Science Analyst",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🤖 GenAI-Powered Data Science Analyst")

st.write(
    "Upload a dataset and automatically perform "
    "data profiling, exploratory analysis, statistics "
    "and machine learning."
)

st.divider()


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📂 Upload your dataset",
    type=["csv", "xlsx"]
)


if uploaded_file is not None:

    # --------------------------------------------------
    # READ DATASET
    # --------------------------------------------------

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)


    st.success("✅ Dataset uploaded successfully!")


    # --------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------

    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )


    st.divider()


    # --------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------

    st.header("🔍 Data Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    st.divider()


    # --------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------

    st.header("🧾 Data Types")

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        dtype_df,
        use_container_width=True
    )


    st.divider()


    # ==================================================
    # EXPLORATORY DATA ANALYSIS
    # ==================================================

    st.header("📈 Exploratory Data Analysis")


    # --------------------------------------------------
    # NUMERICAL SUMMARY
    # --------------------------------------------------

    st.subheader("🔢 Numerical Statistics")

    num_summary = numerical_summary(df)

    if not num_summary.empty:

        st.dataframe(
            num_summary,
            use_container_width=True
        )

    else:

        st.info(
            "No numerical columns found."
        )


    # --------------------------------------------------
    # CATEGORICAL SUMMARY
    # --------------------------------------------------

    st.subheader("🔤 Categorical Analysis")

    cat_summary = categorical_summary(df)

    if not cat_summary.empty:

        st.dataframe(
            cat_summary,
            use_container_width=True
        )

    else:

        st.info(
            "No categorical columns found."
        )


    # --------------------------------------------------
    # HISTOGRAM
    # --------------------------------------------------

    st.subheader("📊 Distribution Analysis")

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if numerical_columns:

        selected_column = st.selectbox(
            "Select a numerical column",
            numerical_columns
        )

        fig, ax = plt.subplots()

        ax.hist(
            df[selected_column].dropna(),
            bins=30
        )

        ax.set_title(
            f"Distribution of {selected_column}"
        )

        ax.set_xlabel(
            selected_column
        )

        ax.set_ylabel(
            "Frequency"
        )

        st.pyplot(fig)


    # --------------------------------------------------
    # BOXPLOT
    # --------------------------------------------------

    st.subheader("📦 Outlier Analysis")

    if numerical_columns:

        selected_box_column = st.selectbox(
            "Select column for boxplot",
            numerical_columns,
            key="boxplot"
        )

        fig, ax = plt.subplots()

        ax.boxplot(
            df[selected_box_column].dropna()
        )

        ax.set_title(
            f"Boxplot of {selected_box_column}"
        )

        ax.set_ylabel(
            selected_box_column
        )

        st.pyplot(fig)


    # --------------------------------------------------
    # OUTLIER TABLE
    # --------------------------------------------------

    outlier_df = detect_outliers(df)

    if not outlier_df.empty:

        st.dataframe(
            outlier_df,
            use_container_width=True
        )


    # --------------------------------------------------
    # CORRELATION
    # --------------------------------------------------

    st.subheader("🔗 Correlation Analysis")

    if len(numerical_columns) >= 2:

        correlation = df[
            numerical_columns
        ].corr()

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            ax=ax
        )

        ax.set_title(
            "Correlation Matrix"
        )

        st.pyplot(fig)

    else:

        st.info(
            "At least two numerical columns "
            "are required for correlation analysis."
        )
            # ==================================================
    # STATISTICAL ANALYSIS
    # ==================================================

    st.divider()

    st.header("📐 Statistical Analysis")

    st.write(
        "Use statistical tests to identify relationships "
        "and determine whether observed patterns are "
        "statistically significant."
    )


    # --------------------------------------------------
    # CORRELATION SIGNIFICANCE
    # --------------------------------------------------

    st.subheader("🔗 Correlation Significance")

    correlation_results = correlation_analysis(df)

    if not correlation_results.empty:

        st.dataframe(
            correlation_results,
            use_container_width=True
        )

        st.info(
            "A p-value below 0.05 indicates statistically "
            "significant evidence of a relationship."
        )

    else:

        st.info(
            "Not enough numerical variables for correlation analysis."
        )


    # --------------------------------------------------
    # CHI-SQUARE TEST
    # --------------------------------------------------

    st.subheader("🧪 Chi-Square Test")

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if len(categorical_columns) >= 2:

        col1 = st.selectbox(
            "Select first categorical variable",
            categorical_columns,
            key="chi_col1"
        )

        col2 = st.selectbox(
            "Select second categorical variable",
            categorical_columns,
            key="chi_col2"
        )

        if col1 != col2:

            if st.button(
                "Run Chi-Square Test"
            ):

                result = chi_square_test(
                    df,
                    col1,
                    col2
                )

                st.metric(
                    "Chi-Square Statistic",
                    result["Chi-Square Statistic"]
                )

                st.metric(
                    "p-value",
                    result["p-value"]
                )

                st.write(
                    f"**Conclusion:** {result['Conclusion']}"
                )

        else:

            st.warning(
                "Please select two different variables."
            )

    else:

        st.info(
            "At least two categorical variables "
            "are required."
        )


    # --------------------------------------------------
    # T-TEST
    # --------------------------------------------------

    st.subheader("📊 Two-Group Comparison")

    if categorical_columns and numerical_columns:

        group_column = st.selectbox(
            "Select grouping variable",
            categorical_columns,
            key="ttest_group"
        )

        numerical_column = st.selectbox(
            "Select numerical variable",
            numerical_columns,
            key="ttest_num"
        )

        groups = (
            df[group_column]
            .dropna()
            .unique()
            .tolist()
        )

        if len(groups) >= 2:

            group1 = st.selectbox(
                "Select Group 1",
                groups,
                key="group1"
            )

            group2 = st.selectbox(
                "Select Group 2",
                groups,
                key="group2"
            )

            if group1 != group2:

                if st.button(
                    "Run T-Test"
                ):

                    result = t_test(
                        df,
                        numerical_column,
                        group_column,
                        group1,
                        group2
                    )

                    st.metric(
                        "T-statistic",
                        result["T-statistic"]
                    )

                    st.metric(
                        "p-value",
                        result["p-value"]
                    )

                    st.write(
                        f"**Conclusion:** {result['Conclusion']}"
                    )

            else:

                st.warning(
                    "Please select two different groups."
                )


    # --------------------------------------------------
    # ANOVA
    # --------------------------------------------------

    st.subheader("📈 ANOVA — Multiple Group Comparison")

    if categorical_columns and numerical_columns:

        anova_category = st.selectbox(
            "Select categorical variable",
            categorical_columns,
            key="anova_cat"
        )

        anova_numeric = st.selectbox(
            "Select numerical variable",
            numerical_columns,
            key="anova_num"
        )

        if st.button(
            "Run ANOVA"
        ):

            result = anova_test(
                df,
                anova_numeric,
                anova_category
            )

            if result is not None:

                st.metric(
                    "F-statistic",
                    result["F-statistic"]
                )

                st.metric(
                    "p-value",
                    result["p-value"]
                )

                st.write(
                    f"**Conclusion:** {result['Conclusion']}"
                )
                    # ==================================================
    # MACHINE LEARNING
    # ==================================================

    st.divider()

    st.header("🤖 Machine Learning")

    st.write(
        "Select a target variable and the system will "
        "automatically determine the problem type and "
        "compare multiple machine learning models."
    )

    target_column = st.selectbox(
        "🎯 Select Target Column",
        df.columns,
        key="ml_target"
    )

    if st.button(
        "🚀 Train Machine Learning Models"
    ):

        X = df.drop(
            columns=[target_column]
        )

        y = df[target_column]

        # Remove rows where target is missing
        valid_rows = y.notna()

        X = X.loc[valid_rows]
        y = y.loc[valid_rows]

        problem_type = detect_problem_type(y)
        st.session_state.problem_type = problem_type
        st.success(
            f"Detected Problem Type: "
            f"**{problem_type}**"
        )

        # --------------------------------------------------
        # CLASSIFICATION
        # --------------------------------------------------

        if problem_type == "Classification":

            # Remove very high-cardinality text columns
            # such as passenger names or IDs
            categorical_columns_ml = X.select_dtypes(
                include=["object", "category"]
            ).columns

            columns_to_drop = []

            for column in categorical_columns_ml:

                if X[column].nunique() > 50:

                    columns_to_drop.append(column)

            if columns_to_drop:

                X = X.drop(
                    columns=columns_to_drop
                )

                st.info(
                    "Automatically removed high-cardinality "
                    f"columns: {', '.join(columns_to_drop)}"
                )

            results_df, trained_models, X_test, y_test = (
                train_classification_models(
                    X,
                    y
                )
            )
            st.session_state.ml_results = results_df
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.subheader(
                "📊 Model Comparison"
            )

            st.dataframe(
                results_df,
                use_container_width=True
            )

            best_model = results_df.iloc[0]["Model"]
            st.session_state.best_model_name = best_model
            st.session_state.best_model_pipeline = (
    trained_models[best_model]
)
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.success(
                f"🏆 Best Model: **{best_model}**"
            )

            best_row = results_df.iloc[0]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Accuracy",
                    best_row["Accuracy"]
                )

            with col2:
                st.metric(
                    "Precision",
                    best_row["Precision"]
                )

            with col3:
                st.metric(
                    "Recall",
                    best_row["Recall"]
                )

            with col4:
                st.metric(
                    "F1 Score",
                    best_row["F1 Score"]
                )


        # --------------------------------------------------
        # REGRESSION
        # --------------------------------------------------

        else:

            results_df, trained_models, X_test, y_test = (
                train_regression_models(
                    X,
                    y
                )
            )
            st.session_state.ml_results = results_df
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            st.subheader(
                "📊 Model Comparison"
            )

            st.dataframe(
                results_df,
                use_container_width=True
            )

            best_model = results_df.iloc[0]["Model"]
            st.session_state.best_model_name = best_model
            st.session_state.best_model_pipeline = (
                trained_models[best_model]
            )
            st.success(
                f"🏆 Best Model: **{best_model}**"
            )

            best_row = results_df.iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "MAE",
                    best_row["MAE"]
                )

            with col2:
                st.metric(
                    "RMSE",
                    best_row["RMSE"]
                )

            with col3:
                st.metric(
                    "R² Score",
                    best_row["R² Score"]
                )
                # ==================================================
# EXPLAINABLE AI
# ==================================================

st.divider()

st.header("🔍 Explainable AI")

st.write(
    "Understand which features have the greatest "
    "influence on the machine learning model."
)

if (
    st.session_state.best_model_pipeline
    is not None
):

    pipeline = (
        st.session_state
        .best_model_pipeline
    )

    X_test = (
        st.session_state.X_test
    )

    model_name = (
        st.session_state.best_model_name
    )

    # SHAP currently works best with
    # tree-based models
    if (
        "RandomForest"
        in pipeline.named_steps["model"].__class__.__name__
        or
        "XGB"
        in pipeline.named_steps["model"].__class__.__name__
        or
        "DecisionTree"
        in pipeline.named_steps["model"].__class__.__name__
    ):

        try:

            shap_values, X_transformed, feature_names = (
                calculate_shap_values(
                    pipeline,
                    X_test
                )
            )

            importance_df = (
                get_feature_importance(
                    shap_values,
                    feature_names
                )
            )
            st.session_state.shap_importance = importance_df
            st.subheader(
                "📊 Feature Importance"
            )

            st.dataframe(
                importance_df.head(15),
                use_container_width=True
            )

            st.subheader(
                "📈 Top Features"
            )

            top_features = (
                importance_df
                .head(10)
                .sort_values(
                    "Importance"
                )
            )

            st.bar_chart(
                top_features.set_index(
                    "Feature"
                )
            )

            st.success(
                f"Model explained: {model_name}"
            )

        except Exception as e:

            st.error(
                f"SHAP explanation failed: {e}"
            )

    else:

        st.info(
            "The selected model is not currently "
            "supported by this SHAP visualization. "
            "Run the analysis again and select a "
            "tree-based model such as Random Forest "
            "or XGBoost."
        )

else:

    st.info(
        "Train a machine learning model first "
        "to enable Explainable AI."
    )
    # ==================================================
# GENAI DATA SCIENCE ANALYST
# ==================================================

st.divider()

st.header("🧠 AI Data Science Analyst")

st.write(
    "Ask questions about the dataset, EDA, statistics, "
    "machine learning performance, and model explainability."
)


if st.session_state.best_model_pipeline is not None:

    question = st.text_area(
        "💬 Ask the AI Data Scientist",
        placeholder=(
            "Example: Why is this model performing well?"
        ),
        key="ai_question"
    )

    if st.button(
        "✨ Ask AI Analyst",
        key="ask_ai"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            # ==========================================
            # BUILD COMPLETE ANALYSIS CONTEXT
            # ==========================================

            context_parts = []


            # ------------------------------------------
            # DATASET INFORMATION
            # ------------------------------------------

            context_parts.append(
                f"""
DATASET INFORMATION
-------------------

Rows: {df.shape[0]}
Columns: {df.shape[1]}

Column names:
{list(df.columns)}
"""
            )


            # ------------------------------------------
            # DATA QUALITY
            # ------------------------------------------

            missing_values = (
                df.isnull()
                .sum()
            )

            duplicate_count = (
                df.duplicated()
                .sum()
            )

            context_parts.append(
                f"""
DATA QUALITY
------------

Total missing values:
{int(missing_values.sum())}

Duplicate rows:
{int(duplicate_count)}

Missing values by column:
{missing_values.to_string()}
"""
            )


            # ------------------------------------------
            # NUMERICAL STATISTICS
            # ------------------------------------------

            try:

                num_summary = (
                    numerical_summary(df)
                )

                context_parts.append(
                    f"""
NUMERICAL STATISTICS
--------------------

{num_summary.to_string()}
"""
                )

            except Exception:
                pass


            # ------------------------------------------
            # CATEGORICAL ANALYSIS
            # ------------------------------------------

            try:

                cat_summary = (
                    categorical_summary(df)
                )

                if not cat_summary.empty:

                    context_parts.append(
                        f"""
CATEGORICAL ANALYSIS
--------------------

{cat_summary.to_string(
    index=False
)}
"""
                    )

            except Exception:
                pass


            # ------------------------------------------
            # CORRELATION ANALYSIS
            # ------------------------------------------

            try:

                correlation_results = (
                    correlation_analysis(df)
                )

                if not correlation_results.empty:

                    context_parts.append(
                        f"""
CORRELATION ANALYSIS
--------------------

{correlation_results.to_string(
    index=False
)}
"""
                    )

            except Exception:
                pass


            # ------------------------------------------
            # MACHINE LEARNING
            # ------------------------------------------

            if (
                st.session_state.ml_results
                is not None
            ):

                context_parts.append(
                    f"""
MACHINE LEARNING RESULTS
------------------------

Problem Type:
{st.session_state.problem_type}

Best Model:
{st.session_state.best_model_name}

Model Comparison:

{st.session_state.ml_results.to_string(
    index=False
)}
"""
                )


            # ------------------------------------------
            # SHAP EXPLAINABILITY
            # ------------------------------------------

            if (
                st.session_state.shap_importance
                is not None
            ):

                context_parts.append(
                    f"""
SHAP FEATURE IMPORTANCE
-----------------------

Top predictive features:

{st.session_state.shap_importance
.head(15)
.to_string(index=False)}
"""
                )


            # ------------------------------------------
            # COMBINE EVERYTHING
            # ------------------------------------------

            analysis_context = "\n".join(
                context_parts
            )


            # ==========================================
            # SEND VERIFIED RESULTS TO GEMINI
            # ==========================================

            with st.spinner(
                "🧠 AI is analyzing the Data Science results..."
            ):

                try:

                    answer = generate_analysis(
                        question,
                        analysis_context
                    )

                    st.subheader(
                        "🤖 AI Analyst Response"
                    )

                    st.markdown(
                        answer
                    )

                except Exception as e:

                    st.error(
                        f"AI analysis failed: {e}"
                    )

else:

    st.info(
        "Train a machine learning model first "
        "to activate the AI Data Scientist."
    )