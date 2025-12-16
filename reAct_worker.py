import json
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.genai import types
import asyncio

from google.adk.runners import InMemoryRunner
from google.adk.apps.app import App
from google.adk.tools.function_tool import FunctionTool

from evaluation_utils import evaluate
from search_fhir_terminology import terminology_search as search_fhir_terminology



load_dotenv()

NOTE_DB = {}   # Will hold {note_id: note_text}
TRUTH_DB = {}  # Will hold {note_id: ground_truth_dict}

try:
    # Get the API key from the environment
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")
    print(" Gemini API key setup complete.")
except Exception as e:
    print(f" Authentication Error: {e}")

# retry option
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

worker_agent = Agent(
    name="WorkerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
        api_key = GOOGLE_API_KEY
    ),
    instruction="""
You are the Worker Agent.

You MUST use the ReAct pattern:
1. Thought: describe what you need to do
2. Action: call one of the provided tools
3. Observation: you will receive tool output
4. Answer: produce the final structured JSON output

Your job is to process one or multiple clinical notes and extract structured clinical indicators.

BUT IMPORTANT:
When identifying diagnoses, conditions, symptoms, medications, or lab analytes,
you MUST normalize them using the Cloud Healthcare API tool `terminology_search`.
Use 'http://snomed.info/sct' for diagnoses, conditions, and symptoms.
Use 'http://www.nlm.nih.gov/research/umls/rxnorm' for medications.
Use 'http://loinc.org' for lab analytes.
Never guess medical codes. Always perform a tool search.

=========================
Tasks:
For each clinical note:
1. Extract key indicators:
   - Vital signs (BP, HR, RR, SpO2, temperature)
   - Labs (Hb, WBC, platelets, glucose, Cr, Na/K, etc.)
   - Diagnoses and symptoms (normalized via SNOMED using ReAct + tool calls)
   - Treatments / medications (normalize via RxNorm)
2. Always perform reasoning step-by-step using ReAct.
3. Then output the final structured data.

=========================
Output Schema:
Return a JSON list. Example format:

[
  {
    "note_id": "",
    "visit_date": "",
    "indicators": {
        "vitals": {...},
        "labs": {...}
    },
    "diagnosis": [
       {
         "text": "",
         "normalized": {
            "code": "",
            "system": "",
            "display": ""
         }
       }
    ],
    "treatment": [
       {
         "text": "",
         "normalized": {
            "code": "",
            "system": "",
            "display": ""
         }
       }
    ],
    "summary": ""
  }
]
=========================

You MUST use tools with explicit ReAct steps.
Do NOT output thoughts in the final answer.
""",
    output_key="note_extractions",
    tools = [FunctionTool(search_fhir_terminology)]


)

# run worker agent
app = App(name="WorkerAgentApp", root_agent=worker_agent)

# Runner
test_runner = InMemoryRunner(agent=app.root_agent)


# Async execution
async def main():
    response = await test_runner.run_debug(
        "test notes..., please extract structured clinical indicators as per the instructions."
    )
    final_answers = []
    for event in response:
        if event.type == "Answer":
            final_answers.append(event.content)

    # Print nicely
    print(json.dumps(final_answers, indent=2))

asyncio.run(main())
