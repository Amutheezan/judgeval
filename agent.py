import os

import secret_key
from judgeval.tracer import Tracer, wrap
from openai import OpenAI

client = wrap(OpenAI())  # tracks all LLM calls
judgment = Tracer(project_name="my_project")

os.environ["JUDGMENT_API_KEY"] = secret_key.JUDGMENT_API_KEY
os.environ["JUDGMENT_ORG_ID"] = secret_key.JUDGMENT_ORG_ID

@judgment.observe(span_type="tool")
def format_question(question: str) -> str:
    # dummy tool
    return f"Question : {question}"


@judgment.observe(span_type="function")
def run_agent(prompt: str) -> str:
    task = format_question(prompt)
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": task}]
    )
    return response.choices[0].message.content


run_agent("What is the capital of the United States?")
