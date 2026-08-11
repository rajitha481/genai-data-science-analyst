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


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="GenAI Data Science Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
# =========================================================
# PROFESSIONAL UI STYLING
# =========================================================

st.markdown(
    """
    <style>

    /* Main application */

    .main {
        padding-top: 1rem;
    }


    /* Main title */

    h1 {
        font-size: 2.4rem;
        font-weight: 700;
    }


    /* Section headings */

    h2 {
        font-size: 1.8rem;
        font-weight: 650;
    }

    h3 {
        font-size: 1.35rem;
        font-weight: 600;
    }


    /* Metric cards */

    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
    }


    /* Buttons */

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }


    /* File uploader */

    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }


    /* Dataframes */

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
    }


    /* Sidebar */

    section[data-testid="stSidebar"] {
        border-right: 1px solid #e2e8f0;
    }


    /* Info boxes */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }


    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SESSION STATE
# =========================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

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

if "target_column" not in st.session_state:
    st.session_state.target_column = None


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🤖 Data Science Analyst")
st.sidebar.caption(
    "Intelligent analytics workspace"
)
st.sidebar.write(
    "AI-powered Data Science platform"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Exploratory Data Analysis",
        "📐 Statistical Analysis",
        "🤖 Machine Learning",
        "🔍 Explainable AI",
        "🧠 AI Data Scientist",
        "📄 Report"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Python • Streamlit • Scikit-learn • SHAP • Gemini"
)


# =========================================================
# HEADER
# =========================================================

st.title("🤖 GenAI-Powered Data Science Analyst")
st.markdown(
    """
    **Transform raw data into insights, predictions,
    explanations, and AI-powered recommendations.**
    """
)
st.caption(
    "From raw data to statistical insights, machine learning "
    "and GenAI-powered explanations."
)
st.info(
    "📌 Upload a dataset to automatically explore "
    "data quality, statistics, machine learning performance, "
    "model explainability, and GenAI insights."
)

# =========================================================
# DATASET UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload your dataset",
    type=["csv", "xlsx"],
    help="Upload a CSV or Excel dataset."
)


# =========================================================
# READ DATASET
# =========================================================

if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.df = df
        st.session_state.file_name = uploaded_file.name

        st.success(
            f"✅ Successfully loaded {uploaded_file.name}"
        )

    except Exception as e:

        st.error(
            f"Could not read the dataset: {e}"
        )

        st.stop()


elif st.session_state.df is not None:

    df = st.session_state.df

else:

    df = None


# =========================================================
# NO DATASET
# =========================================================

if df is None:

    st.info(
        "👆 Upload a CSV or Excel dataset to begin."
    )

    st.markdown(
        """
        ### What this application can do

        **📋 Data Profiling**
        - Dataset dimensions
        - Missing values
        - Duplicate detection
        - Data types

        **📊 Exploratory Data Analysis**
        - Descriptive statistics
        - Distributions
        - Boxplots
        - Correlation analysis

        **📐 Statistics**
        - Pearson correlation
        - Spearman correlation
        - Chi-Square
        - T-Test
        - ANOVA

        **🤖 Machine Learning**
        - Classification
        - Regression
        - Multiple model comparison

        **🔍 Explainable AI**
        - SHAP feature importance

        **🧠 GenAI**
        - AI-powered data interpretation
        - Business insights
        - Recommendations
        """
    )

    st.stop()


# =========================================================
# COMMON DATA INFORMATION
# =========================================================

total_rows = df.shape[0]
total_columns = df.shape[1]

total_missing = int(
    df.isnull().sum().sum()
)

total_duplicates = int(
    df.duplicated().sum()
)

numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.header("🏠 Analytics Dashboard")
    st.caption(
    "A high-level view of your dataset and analysis results."
)
    st.caption(
        f"Currently analyzing: **{st.session_state.file_name}**"
    )
    st.markdown(
    """
    ### 🔄 Data Science Workflow

    **Upload → Explore → Analyze → Predict → Explain → Recommend**
    """
)

    st.divider()

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📊 Total Rows",
            f"{total_rows:,}"
        )

    with col2:

        st.metric(
            "📋 Total Columns",
            total_columns
        )

    with col3:

        st.metric(
            "⚠️ Missing Values",
            f"{total_missing:,}"
        )

    with col4:

        st.metric(
            "♻️ Duplicate Rows",
            f"{total_duplicates:,}"
        )

    st.divider()

    # -----------------------------------------------------
    # DATASET HEALTH
    # -----------------------------------------------------

    st.subheader("🩺 Dataset Health")

    if (
        total_missing == 0
        and total_duplicates == 0
    ):

        st.success(
            "✅ Excellent! No missing values or "
            "duplicate rows detected."
        )

    elif total_missing == 0:

        st.warning(
            f"⚠️ No missing values detected, but "
            f"{total_duplicates} duplicate rows were found."
        )

    elif total_duplicates == 0:

        st.warning(
            f"⚠️ {total_missing} missing values were detected."
        )

    else:

        st.warning(
            f"⚠️ Dataset contains {total_missing} "
            f"missing values and {total_duplicates} "
            f"duplicate rows."
        )

    st.divider()

    # -----------------------------------------------------
    # DATA COMPOSITION
    # -----------------------------------------------------

    st.subheader("🧾 Data Composition")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🔢 Numerical Columns",
            len(numerical_columns)
        )

    with col2:

        st.metric(
            "🔤 Categorical Columns",
            len(categorical_columns)
        )

    with col3:

        other_columns = (
            total_columns
            - len(numerical_columns)
            - len(categorical_columns)
        )

        st.metric(
            "📦 Other Columns",
            other_columns
        )

    st.divider()

    # -----------------------------------------------------
    # DATA PREVIEW
    # -----------------------------------------------------

    st.subheader("👀 Data Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------------
    # COLUMN INFORMATION
    # -----------------------------------------------------

    st.subheader("🧾 Column Information")

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

    # -----------------------------------------------------
    # MACHINE LEARNING SUMMARY
    # -----------------------------------------------------

    st.subheader("🤖 Machine Learning Summary")

    if st.session_state.ml_results is not None:

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🎯 Problem Type",
                st.session_state.problem_type
            )

        with col2:
            st.metric(
                "🏆 Best Model",
                st.session_state.best_model_name
            )

        st.write("### 📊 Model Comparison")

        st.dataframe(
            st.session_state.ml_results,
            use_container_width=True
        )

    else:

        st.info(
            "Train a machine learning model from "
            "the Machine Learning section to see "
            "model performance here."
        )
        st.divider()

    # -----------------------------------------------------
    # SHAP SUMMARY
    # -----------------------------------------------------

    st.subheader(
        "🔍 Explainable AI Summary"
    )

    if (
        st.session_state.shap_importance
        is not None
    ):

        st.write(
            "Top features influencing the model:"
        )

        top_features = (
            st.session_state.shap_importance
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

    else:

        st.info(
            "Generate SHAP explanations from the "
            "Explainable AI section to see feature "
            "importance here."
        )

# =========================================================
# EDA
# =========================================================

elif page == "📊 Exploratory Data Analysis":

    st.header("📊 Exploratory Data Analysis")

    st.write(
        "Explore distributions, descriptive statistics, "
        "outliers and correlations."
    )

    st.divider()

    # -----------------------------------------------------
    # NUMERICAL STATISTICS
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # CATEGORICAL ANALYSIS
    # -----------------------------------------------------

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

    st.divider()

    # -----------------------------------------------------
    # DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("📈 Distribution Analysis")

    if numerical_columns:

        selected_column = st.selectbox(
            "Select a numerical column",
            numerical_columns,
            key="eda_distribution"
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

    else:

        st.info(
            "No numerical columns available."
        )

    # -----------------------------------------------------
    # BOXPLOT
    # -----------------------------------------------------

    st.subheader("📦 Outlier Analysis")

    if numerical_columns:

        selected_box_column = st.selectbox(
            "Select column for boxplot",
            numerical_columns,
            key="eda_boxplot"
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

        outlier_df = detect_outliers(df)

        if not outlier_df.empty:

            st.dataframe(
                outlier_df,
                use_container_width=True
            )

    # -----------------------------------------------------
    # CORRELATION
    # -----------------------------------------------------

    st.subheader("🔗 Correlation Matrix")

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
            "At least two numerical columns are required."
        )


# =========================================================
# STATISTICAL ANALYSIS
# =========================================================

elif page == "📐 Statistical Analysis":

    st.header("📐 Statistical Analysis")

    st.write(
        "Use statistical tests to identify relationships "
        "and determine whether observed patterns are "
        "statistically significant."
    )

    st.divider()

    # -----------------------------------------------------
    # CORRELATION SIGNIFICANCE
    # -----------------------------------------------------

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
            "Not enough numerical variables."
        )

    st.divider()

    # -----------------------------------------------------
    # CHI-SQUARE
    # -----------------------------------------------------

    st.subheader("🧪 Chi-Square Test")

    if len(categorical_columns) >= 2:

        col1, col2 = st.columns(2)

        with col1:

            chi_col1 = st.selectbox(
                "First categorical variable",
                categorical_columns,
                key="chi_col1"
            )

        with col2:

            chi_col2 = st.selectbox(
                "Second categorical variable",
                categorical_columns,
                key="chi_col2"
            )

        if chi_col1 != chi_col2:

            if st.button(
                "Run Chi-Square Test",
                key="run_chi"
            ):

                result = chi_square_test(
                    df,
                    chi_col1,
                    chi_col2
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Chi-Square",
                        result[
                            "Chi-Square Statistic"
                        ]
                    )

                with col2:

                    st.metric(
                        "p-value",
                        result["p-value"]
                    )

                with col3:

                    st.metric(
                        "Degrees of Freedom",
                        result[
                            "Degrees of Freedom"
                        ]
                    )

                st.write(
                    f"**Conclusion:** "
                    f"{result['Conclusion']}"
                )

        else:

            st.warning(
                "Select two different variables."
            )

    else:

        st.info(
            "At least two categorical variables are required."
        )

    st.divider()

    # -----------------------------------------------------
    # T-TEST
    # -----------------------------------------------------

    st.subheader("📊 Two-Group Comparison")

    if (
        categorical_columns
        and numerical_columns
    ):

        group_column = st.selectbox(
            "Grouping variable",
            categorical_columns,
            key="ttest_group"
        )

        numerical_column = st.selectbox(
            "Numerical variable",
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

            col1, col2 = st.columns(2)

            with col1:

                group1 = st.selectbox(
                    "Group 1",
                    groups,
                    key="group1"
                )

            with col2:

                group2 = st.selectbox(
                    "Group 2",
                    groups,
                    key="group2"
                )

            if group1 != group2:

                if st.button(
                    "Run T-Test",
                    key="run_ttest"
                ):

                    result = t_test(
                        df,
                        numerical_column,
                        group_column,
                        group1,
                        group2
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "T-statistic",
                            result["T-statistic"]
                        )

                    with col2:

                        st.metric(
                            "p-value",
                            result["p-value"]
                        )

                    st.write(
                        f"**Conclusion:** "
                        f"{result['Conclusion']}"
                    )

    # -----------------------------------------------------
    # ANOVA
    # -----------------------------------------------------

    st.subheader("📈 ANOVA — Multiple Groups")

    if (
        categorical_columns
        and numerical_columns
    ):

        anova_category = st.selectbox(
            "Categorical variable",
            categorical_columns,
            key="anova_cat"
        )

        anova_numeric = st.selectbox(
            "Numerical variable",
            numerical_columns,
            key="anova_num"
        )

        if st.button(
            "Run ANOVA",
            key="run_anova"
        ):

            result = anova_test(
                df,
                anova_numeric,
                anova_category
            )

            if result is not None:

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "F-statistic",
                        result["F-statistic"]
                    )

                with col2:

                    st.metric(
                        "p-value",
                        result["p-value"]
                    )

                st.write(
                    f"**Conclusion:** "
                    f"{result['Conclusion']}"
                )


# =========================================================
# MACHINE LEARNING
# =========================================================

elif page == "🤖 Machine Learning":

    st.header("🤖 Machine Learning")

    st.write(
        "Select a target column. The system automatically "
        "detects classification or regression and compares "
        "multiple machine learning models."
    )

    st.divider()

    target_column = st.selectbox(
        "🎯 Select Target Column",
        df.columns,
        key="ml_target"
    )

    st.session_state.target_column = target_column

    if st.button(
        "🚀 Train Machine Learning Models",
        key="train_ml"
    ):

        X = df.drop(
            columns=[target_column]
        )

        y = df[target_column]

        # Remove rows with missing target
        valid_rows = y.notna()

        X = X.loc[valid_rows]
        y = y.loc[valid_rows]

        problem_type = detect_problem_type(y)

        st.session_state.problem_type = (
            problem_type
        )

        # ---------------------------------------------
        # REMOVE HIGH CARDINALITY COLUMNS
        # ---------------------------------------------

        categorical_columns_ml = X.select_dtypes(
            include=["object", "category"]
        ).columns

        columns_to_drop = []

        for column in categorical_columns_ml:

            if X[column].nunique() > 50:

                columns_to_drop.append(
                    column
                )

        if columns_to_drop:

            X = X.drop(
                columns=columns_to_drop
            )

            st.info(
                "Automatically removed high-cardinality "
                f"columns: {', '.join(columns_to_drop)}"
            )

        st.success(
            f"Detected Problem Type: **{problem_type}**"
        )

        # ---------------------------------------------
        # CLASSIFICATION
        # ---------------------------------------------

        if problem_type == "Classification":

            (
                results_df,
                trained_models,
                X_test,
                y_test
            ) = train_classification_models(
                X,
                y
            )

        # ---------------------------------------------
        # REGRESSION
        # ---------------------------------------------

        else:

            (
                results_df,
                trained_models,
                X_test,
                y_test
            ) = train_regression_models(
                X,
                y
            )

        # ---------------------------------------------
        # SAVE RESULTS
        # ---------------------------------------------

        st.session_state.ml_results = (
            results_df
        )

        st.session_state.X_test = (
            X_test
        )

        st.session_state.y_test = (
            y_test
        )

        best_model = (
            results_df.iloc[0]["Model"]
        )

        st.session_state.best_model_name = (
            best_model
        )

        st.session_state.best_model_pipeline = (
            trained_models[best_model]
        )

        # ---------------------------------------------
        # RESULTS
        # ---------------------------------------------

        st.subheader(
            "📊 Model Comparison"
        )

        st.dataframe(
            results_df,
            use_container_width=True
        )

        st.success(
            f"🏆 Best Model: **{best_model}**"
        )

        # ---------------------------------------------
        # METRICS
        # ---------------------------------------------

        if problem_type == "Classification":

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

        else:

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

    else:

        if (
            st.session_state.ml_results
            is not None
        ):

            st.info(
                "Previous ML results are available. "
                "Click Train again to run a new analysis."
            )

            st.dataframe(
                st.session_state.ml_results,
                use_container_width=True
            )


# =========================================================
# EXPLAINABLE AI / SHAP
# =========================================================

elif page == "🔍 Explainable AI":

    st.header("🔍 Explainable AI")

    st.write(
        "Understand which features have the greatest "
        "influence on the machine learning model."
    )

    st.divider()

    if (
        st.session_state.best_model_pipeline
        is None
    ):

        st.info(
            "Go to Machine Learning and train a model first."
        )

    else:

        pipeline = (
            st.session_state.best_model_pipeline
        )

        X_test = (
            st.session_state.X_test
        )

        model_name = (
            st.session_state.best_model_name
        )

        model_class = (
            pipeline
            .named_steps["model"]
            .__class__
            .__name__
        )

        st.info(
            f"Model being explained: **{model_name}**"
        )

        if (
            "RandomForest"
            in model_class
            or
            "XGB"
            in model_class
            or
            "DecisionTree"
            in model_class
        ):

            if st.button(
                "🔍 Generate SHAP Explanation",
                key="generate_shap"
            ):

                try:

                    (
                        shap_values,
                        X_transformed,
                        feature_names
                    ) = calculate_shap_values(
                        pipeline,
                        X_test
                    )

                    importance_df = (
                        get_feature_importance(
                            shap_values,
                            feature_names
                        )
                    )

                    st.session_state.shap_importance = (
                        importance_df
                    )

                    st.subheader(
                        "📊 Feature Importance"
                    )

                    st.dataframe(
                        importance_df.head(15),
                        use_container_width=True
                    )

                    st.subheader(
                        "📈 Top Predictive Features"
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
                        "✅ SHAP analysis completed."
                    )

                except Exception as e:

                    st.error(
                        f"SHAP explanation failed: {e}"
                    )

        else:

            st.warning(
                "The selected best model is not currently "
                "supported by this SHAP visualization. "
                "Tree-based models such as Random Forest "
                "or XGBoost are recommended."
            )

        # Show previous SHAP results if available

        if (
            st.session_state.shap_importance
            is not None
        ):

            st.subheader(
                "📌 Saved SHAP Results"
            )

            st.dataframe(
                st.session_state.shap_importance.head(15),
                use_container_width=True
            )


# =========================================================
# GENAI DATA SCIENCE ANALYST
# =========================================================

elif page == "🧠 AI Data Scientist":

    st.header("🧠 AI Data Science Analyst")

    st.write(
        "Ask questions about your actual EDA, statistical, "
        "machine learning and SHAP results."
    )

    st.divider()

    if (
        st.session_state.best_model_pipeline
        is None
    ):

        st.info(
            "Please train a machine learning model first."
        )

    else:

        question = st.text_area(
            "💬 Ask the AI Data Scientist",
            placeholder=(
                "Example: What are the most important "
                "factors affecting the prediction?"
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

                context_parts = []

                # -----------------------------------------
                # DATASET
                # -----------------------------------------

                context_parts.append(
                    f"""
DATASET INFORMATION

Dataset:
{st.session_state.file_name}

Rows:
{df.shape[0]}

Columns:
{df.shape[1]}

Column names:
{list(df.columns)}
"""
                )

                # -----------------------------------------
                # DATA QUALITY
                # -----------------------------------------

                missing_values = (
                    df.isnull().sum()
                )

                context_parts.append(
                    f"""
DATA QUALITY

Total missing values:
{int(missing_values.sum())}

Duplicate rows:
{int(df.duplicated().sum())}

Missing values by column:

{missing_values.to_string()}
"""
                )

                # -----------------------------------------
                # NUMERICAL STATISTICS
                # -----------------------------------------

                try:

                    num_summary = (
                        numerical_summary(df)
                    )

                    context_parts.append(
                        f"""
NUMERICAL STATISTICS

{num_summary.to_string()}
"""
                    )

                except Exception:
                    pass

                # -----------------------------------------
                # CATEGORICAL ANALYSIS
                # -----------------------------------------

                try:

                    cat_summary = (
                        categorical_summary(df)
                    )

                    if not cat_summary.empty:

                        context_parts.append(
                            f"""
CATEGORICAL ANALYSIS

{cat_summary.to_string(
    index=False
)}
"""
                        )

                except Exception:
                    pass

                # -----------------------------------------
                # CORRELATION
                # -----------------------------------------

                try:

                    correlation_results = (
                        correlation_analysis(df)
                    )

                    if not correlation_results.empty:

                        context_parts.append(
                            f"""
CORRELATION ANALYSIS

{correlation_results.to_string(
    index=False
)}
"""
                        )

                except Exception:
                    pass

                # -----------------------------------------
                # ML RESULTS
                # -----------------------------------------

                if (
                    st.session_state.ml_results
                    is not None
                ):

                    context_parts.append(
                        f"""
MACHINE LEARNING RESULTS

Problem Type:
{st.session_state.problem_type}

Target:
{st.session_state.target_column}

Best Model:
{st.session_state.best_model_name}

Model Comparison:

{st.session_state.ml_results.to_string(
    index=False
)}
"""
                    )

                # -----------------------------------------
                # SHAP
                # -----------------------------------------

                if (
                    st.session_state.shap_importance
                    is not None
                ):

                    context_parts.append(
                        f"""
SHAP FEATURE IMPORTANCE

Top Features:

{st.session_state.shap_importance
.head(15)
.to_string(index=False)}
"""
                    )

                # -----------------------------------------
                # COMBINE CONTEXT
                # -----------------------------------------

                analysis_context = (
                    "\n".join(context_parts)
                )

                # -----------------------------------------
                # GEMINI
                # -----------------------------------------

                with st.spinner(
                    "🧠 AI is analyzing your results..."
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


# =========================================================
# REPORT
# =========================================================

elif page == "📄 Report":

    st.header("📄 Data Science Report")

    st.write(
        "Generate a structured summary of the analysis."
    )

    st.divider()

    st.subheader(
        "📋 Current Analysis Summary"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Dataset Rows",
            f"{total_rows:,}"
        )

    with col2:

        st.metric(
            "Dataset Columns",
            total_columns
        )

    with col3:

        st.metric(
            "Missing Values",
            f"{total_missing:,}"
        )

    st.divider()

    st.subheader(
        "📊 Dataset Information"
    )

    st.write(
        f"**Dataset:** {st.session_state.file_name}"
    )

    st.write(
        f"**Rows:** {total_rows}"
    )

    st.write(
        f"**Columns:** {total_columns}"
    )

    st.write(
        f"**Numerical columns:** "
        f"{len(numerical_columns)}"
    )

    st.write(
        f"**Categorical columns:** "
        f"{len(categorical_columns)}"
    )

    if (
        st.session_state.problem_type
        is not None
    ):

        st.divider()

        st.subheader(
            "🤖 Machine Learning Summary"
        )

        st.write(
            f"**Problem Type:** "
            f"{st.session_state.problem_type}"
        )

        st.write(
            f"**Target:** "
            f"{st.session_state.target_column}"
        )

        st.write(
            f"**Best Model:** "
            f"{st.session_state.best_model_name}"
        )

        if (
            st.session_state.ml_results
            is not None
        ):

            st.dataframe(
                st.session_state.ml_results,
                use_container_width=True
            )

    if (
        st.session_state.shap_importance
        is not None
    ):

        st.divider()

        st.subheader(
            "🔍 Top SHAP Features"
        )

        st.dataframe(
            st.session_state.shap_importance.head(10),
            use_container_width=True
        )
        st.divider()

    st.subheader("📝 Generate AI Report")

    st.write(
        "Generate a complete Data Science report "
        "using the results from your analysis."
    )

    if st.button(
        "📄 Generate Report",
        key="generate_report"
    ):

        report_parts = []

        # ---------------------------------------------
        # TITLE
        # ---------------------------------------------

        report_parts.append(
            "# GenAI-Powered Data Science Analysis Report\n"
        )

        report_parts.append(
            f"**Dataset:** "
            f"{st.session_state.file_name}\n"
        )

        # ---------------------------------------------
        # DATASET OVERVIEW
        # ---------------------------------------------

        report_parts.append(
            "\n## 1. Dataset Overview\n"
        )

        report_parts.append(
            f"- Rows: {total_rows}\n"
            f"- Columns: {total_columns}\n"
            f"- Numerical columns: "
            f"{len(numerical_columns)}\n"
            f"- Categorical columns: "
            f"{len(categorical_columns)}\n"
        )

        # ---------------------------------------------
        # DATA QUALITY
        # ---------------------------------------------

        report_parts.append(
            "\n## 2. Data Quality\n"
        )

        report_parts.append(
            f"- Missing values: "
            f"{total_missing}\n"
            f"- Duplicate rows: "
            f"{total_duplicates}\n"
        )

        # ---------------------------------------------
        # ML RESULTS
        # ---------------------------------------------

        if (
            st.session_state.ml_results
            is not None
        ):

            report_parts.append(
                "\n## 3. Machine Learning Results\n"
            )

            report_parts.append(
                f"- Problem type: "
                f"{st.session_state.problem_type}\n"
            )

            report_parts.append(
                f"- Target column: "
                f"{st.session_state.target_column}\n"
            )

            report_parts.append(
                f"- Best model: "
                f"{st.session_state.best_model_name}\n"
            )

            report_parts.append(
                "\n### Model Comparison\n\n"
            )

            report_parts.append(
                st.session_state.ml_results
                .to_markdown(
                    index=False
                )
            )

        # ---------------------------------------------
        # SHAP
        # ---------------------------------------------

        if (
            st.session_state.shap_importance
            is not None
        ):

            report_parts.append(
                "\n\n## 4. Explainable AI\n"
            )

            report_parts.append(
                "\n### Top Predictive Features\n\n"
            )

            report_parts.append(
                st.session_state.shap_importance
                .head(10)
                .to_markdown(
                    index=False
                )
            )

        # ---------------------------------------------
        # GEMINI SUMMARY
        # ---------------------------------------------

        if (
            st.session_state.ml_results
            is not None
        ):

            report_context = "\n".join(
                report_parts
            )

            report_question = """
Provide an executive summary of this Data Science
analysis.

Include:

1. Most important findings
2. Important statistical observations
3. Machine learning performance
4. Important predictive features
5. Practical business recommendations
6. Limitations of the analysis

Use only the provided information.
Do not invent numbers.
"""

            with st.spinner(
                "🧠 Gemini is preparing the report..."
            ):

                try:

                    ai_report = generate_analysis(
                        report_question,
                        report_context
                    )

                    report_parts.append(
                        "\n\n## 5. GenAI Executive Summary\n"
                    )

                    report_parts.append(
                        ai_report
                    )

                except Exception as e:

                    st.warning(
                        f"AI summary could not be generated: {e}"
                    )

        # ---------------------------------------------
        # FINAL REPORT
        # ---------------------------------------------

        final_report = "\n".join(
            report_parts
        )

        st.success(
            "✅ Report generated successfully!"
        )

        st.markdown(
            final_report
        )

        # ---------------------------------------------
        # DOWNLOAD
        # ---------------------------------------------

        st.download_button(
            label="⬇️ Download Report",
            data=final_report,
            file_name="data_science_report.md",
            mime="text/markdown"
        )
    
    st.divider()

    # -----------------------------------------------------
    # SHAP SUMMARY
    # -----------------------------------------------------

    st.subheader(
        "🔍 Explainable AI Summary"
    )

    if (
        st.session_state.shap_importance
        is not None
    ):

        st.write(
            "Top features influencing the model:"
        )

        top_features = (
            st.session_state.shap_importance
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

    else:

        st.info(
            "Generate SHAP explanations from the "
            "Explainable AI section to see feature "
            "importance here."
        )

# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "GenAI-Powered Data Science Analyst"
)

st.sidebar.caption(
    "Built with Python, Streamlit, Scikit-learn, SHAP & Gemini"
)