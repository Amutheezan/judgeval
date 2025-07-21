import os

import secret_key
from judgeval.tracer import Tracer
from google import genai

os.environ["JUDGMENT_API_KEY"] = secret_key.JUDGMENT_API_KEY
os.environ["JUDGMENT_ORG_ID"] = secret_key.JUDGMENT_ORG_ID

gen_ai_client = genai.client.Client(api_key=secret_key.GEMINI_API_KEY)

judgment = Tracer(project_name="my_project")

@judgment.observe(span_type="tool")
def format_question(question: str) -> str:
    # dummy tool
    return f"Question : {question}"


@judgment.observe(span_type="function")
def run_agent(prompt: str) -> str:
    task = format_question(prompt)
    response = gen_ai_client.models.generate_content(model="gemini-2.5-pro", contents=task)
    return response.text


print(run_agent("What is the capital of the United States?"))

from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer
client = JudgmentClient(api_key=secret_key.JUDGMENT_API_KEY, organization_id=secret_key.JUDGMENT_ORG_ID)
task = "What is the capital of the United States?"
example = Example(
    input=task,
    actual_output=run_agent(task),  # e.g. "The capital of the U.S. is Washington, D.C."
    retrieval_context=["Washington D.C. was founded in 1790 and became the capital of the U.S."],
)
scorer = FaithfulnessScorer(threshold=0.5)
client.assert_test(
    examples=[example],
    scorers=[scorer],
    model="gemini-2.5-pro",
)