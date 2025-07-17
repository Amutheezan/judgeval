import os

import secret_key
from judgeval.tracer import Tracer
from google import genai

os.environ["JUDGMENT_API_KEY"] = secret_key.JUDGMENT_API_KEY
os.environ["JUDGMENT_ORG_ID"] = secret_key.JUDGMENT_ORG_ID

client = genai.client.Client(api_key=secret_key.GEMINI_API_KEY)

judgment = Tracer(project_name="my_project")
print(client.models.list())

@judgment.observe(span_type="tool")
def format_question(question: str) -> str:
    # dummy tool
    return f"Question : {question}"


@judgment.observe(span_type="function")
def run_agent(prompt: str) -> str:
    task = format_question(prompt)
    response = client.models.generate_content(model="gemini-2.5-pro", contents=task)
    return response.text


print(run_agent("What is the capital of the United States?"))
