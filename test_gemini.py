from src.llm_analyst import generate_analysis


context = """
Dataset: Titanic

Rows: 891
Columns: 12

The machine learning pipeline detected
a binary classification problem.

The best model was selected based on F1 score.
"""


question = "Explain the dataset analysis in simple language."


answer = generate_analysis(
    question,
    context
)


print(answer)