import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not configured. "
        "Please add it to your .env file."
    )

client = genai.Client(
    api_key=api_key
)


def generate_analysis(question, analysis_context):
    """
    Generate a natural-language answer using
    the actual Data Science analysis results.
    """

    prompt = f"""
You are an expert Data Science Analyst.

Your task is to explain data analysis results
clearly and accurately.

IMPORTANT RULES:

1. Use ONLY the analytical results provided below.
2. Never invent statistics, p-values, metrics,
   correlations, or model results.
3. Clearly distinguish correlation from causation.
4. Explain technical results in simple language.
5. Give practical recommendations when appropriate.
6. If the provided results are insufficient,
   clearly say so instead of guessing.

ACTUAL DATA SCIENCE ANALYSIS RESULTS:
--------------------------------------

{analysis_context}

--------------------------------------

USER QUESTION:
{question}

Answer using only the information provided above.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    return interaction.output_text