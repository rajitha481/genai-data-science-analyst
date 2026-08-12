# 🤖 GenAI-Powered Data Science Analyst

An end-to-end AI-powered Data Science platform that automatically analyzes datasets using Exploratory Data Analysis, statistical testing, machine learning, SHAP explainability, and Generative AI.

## 🌐 Live Demo

(https://genai-data-science-analyst-e8gqxkpqyig3cswa5isrvx.streamlit.app)

## 📌 Overview

Traditional data analysis requires analysts to manually perform data profiling, exploratory analysis, statistical testing, machine learning, model interpretation, and report generation.

This project combines these steps into a single interactive Streamlit application.

Users can upload a CSV or Excel dataset and receive:

- Dataset profiling
- Data quality analysis
- Exploratory Data Analysis
- Statistical analysis
- Machine learning model comparison
- Explainable AI using SHAP
- GenAI-powered interpretation
- Business recommendations
- Automated Data Science report

---

## 🎯 Problem Statement

Data scientists often spend significant time performing repetitive analysis tasks before reaching actionable insights.

The goal of this project is to build an intelligent analytics assistant that automates the complete workflow while keeping the numerical analysis grounded in actual computed results.

---

## 💡 Solution

The application follows an end-to-end Data Science workflow:

```text
Dataset Upload
      ↓
Data Profiling
      ↓
Exploratory Data Analysis
      ↓
Statistical Analysis
      ↓
Machine Learning
      ↓
Model Evaluation
      ↓
SHAP Explainability
      ↓
Structured Analysis Context
      ↓
Gemini GenAI
      ↓
Insights & Recommendations
      ↓
Automated Report


```markdown
## 🚀 Key Features

### 📊 1. Dataset Profiling

Automatically identifies:

- Number of rows
- Number of columns
- Data types
- Missing values
- Duplicate records
- Numerical variables
- Categorical variables
- Unique values

### 📈 2. Exploratory Data Analysis

Provides:

- Descriptive statistics
- Numerical summaries
- Categorical summaries
- Distribution analysis
- Boxplots
- Outlier detection
- Correlation matrix

### 📐 3. Statistical Analysis

Includes:

- Pearson correlation
- Spearman correlation
- Chi-Square test
- Independent T-Test
- ANOVA

Statistical results help identify meaningful relationships and differences in the dataset.

### 🤖 4. Machine Learning

The application automatically detects whether the problem is:

- Classification
- Regression

Models are trained and compared using appropriate evaluation metrics.

**Classification metrics:**

- Accuracy
- Precision
- Recall
- F1 Score

**Regression metrics:**

- MAE
- RMSE
- R² Score

### 🔍 5. Explainable AI

SHAP is used to identify the most influential features in the machine learning model.

This helps answer:

> Why did the model make this prediction?

### 🧠 6. GenAI Data Scientist

Gemini receives the actual computed:

- Statistical results
- ML metrics
- Model comparison results
- SHAP feature importance
- Dataset information

The AI converts these results into understandable insights and recommendations.

### 📄 7. Automated Report Generation

The application generates a structured Data Science report containing:

- Dataset overview
- Data quality
- Machine learning results
- SHAP insights
- GenAI executive summary
- Recommendations

The report can be downloaded directly from the application.
---

##  System Architecture

                    USER
                      │
                      ↓
               Dataset Upload
                      │
                      ↓
               Data Validation
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
         EDA      Statistics      ML
          │           │           │
          │           │           ↓
          │           │       Model Metrics
          │           │           │
          └───────────┼───────────┘
                      ↓
                    SHAP
                      │
                      ↓
            Structured Context
                      │
                      ↓
                   Gemini
                      │
             ┌────────┴────────┐
             ↓                 ↓
          Insights       Recommendations
             │                 │
             └────────┬────────┘
                      ↓
              Automated Report


# 3. Add Technology Stack


## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Frontend | Streamlit |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Statistics | SciPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainable AI | SHAP |
| Generative AI | Google Gemini API |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |



## 📁 Project Structure


genai-data-science-analyst/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── titanic.csv
│
└── src/
    ├── eda.py
    ├── statistics.py
    ├── ml_pipeline.py
    ├── explainability.py
    └── llm_analyst.py


# 5. Add Installation

Then:

## ⚙️ Installation

### 1. Clone the repository


git clone YOUR_GITHUB_REPOSITORY_URL
cd genai-data-science-analyst
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 6. Add API Configuration

This is important because you're using Gemini.

## 🔐 API Configuration

Create a `.env` file in the project root:

GEMINI_API_KEY="AQ.Ab8RN6KE_Nz92H4i1WN_OO5gA47RJBnfy8cd1BOBn_RMXmSX8g"

# 7. Add Running Instructions

## ▶️ Run the Application

Activate your virtual environment and run:

streamlit run app.py

# 8. Add Example Dataset



##  Example Dataset

The Titanic dataset can be used to demonstrate the complete workflow.

**Target column:**
Survived

# 9. Add Live Demo

## Live Demo

(https://genai-data-science-analyst-e8gqxkpqyig3cswa5isrvx.streamlit.app/)
---

## Future Improvements

- Automated feature engineering
- Hyperparameter optimization
- Advanced model selection
- Time-series forecasting
- Natural-language dataset querying
- PDF report generation
- Data drift detection
- Model monitoring
- Multi-dataset comparison
- Cloud database integration

## 👩‍💻 Author

**M. Rajitha**

Computer Science & Engineering — Data Science

### Project Focus

Data Science • Machine Learning • Statistics • Explainable AI • Generative AI
