import json
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

import uuid
import time
import asyncio

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool
from requests import session

from evaluation_utils import evaluate

from google.adk.runners import InMemoryRunner
from search_fhir_terminology import terminology_search as search_fhir_terminology


# Load environment variables from the .env file
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


def load_test_data(csv_path, json_path):
    """Loads the CSV and JSON into global dictionaries."""
    global NOTE_DB, TRUTH_DB
    
    print(f"📂 Loading data from {csv_path} and {json_path}...")
    
    # 1. Load CSV (The Clinical Notes)
    try:
        df = pd.read_csv(csv_path)
        # Create a unique Note ID combining Subject and Row if specific ID absent
        # Assuming CSV has 'SUBJECT_ID' and 'TEXT' columns based on your previous prompt
        for index, row in df.iterrows():
            # Construct a consistent Note ID. 
            # Strategy: Use SubjectID_RowID to ensure uniqueness
            note_id = str(row["SUBJECT_ID"])
            NOTE_DB[note_id] = row["TEXT"]
            
            # Map Subject ID to list of notes for the Orchestrator
            if row['SUBJECT_ID'] not in PATIENT_INDEX:
                PATIENT_INDEX[row['SUBJECT_ID']] = []
            PATIENT_INDEX[row['SUBJECT_ID']].append(note_id)
            
        print(f"   -> Loaded {len(NOTE_DB)} notes.")
    except Exception as e:
        print(f"   -> Error loading CSV: {e}")
        return False

    # 2. Load JSON (The Ground Truth)
    try:
        with open(json_path, 'r') as f:
            TRUTH_DB = json.load(f)
        print(f"   -> Loaded ground truth for {len(TRUTH_DB)} notes.")
    except Exception as e:
        print(f"   -> Error loading JSON: {e}")
        return False
        
    return True

PATIENT_INDEX = {}

def get_note_text(note_id):
    return NOTE_DB.get(note_id, "")

def get_ground_truth(note_id):
    return TRUTH_DB.get(note_id, {})

# retry option
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# ----------------------------------------------------
#  AGENT SETUP (Orchestrator, Worker, Synthesizer)
# ----------------------------------------------------

orchestrator_agent = Agent(
    name="OrchestratorAgent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""You are the Orchestrator Agent managing multiple clinical note analyses.
Steps:
1. Identify all note_ids for the patient.
2. For each note_id, call WorkerAgent to process it.
3. Collect WorkerAgent outputs that are approved.
4. Send all approved outputs to SynthesizerAgent for trend analysis.

Return JSON:
{
  "patient_id": <patient_id>,
  "note_ids": [],
  "approved_outputs": [],
  "next_agent": "SynthesizerAgent"
}""",
    tools=[],
    output_key="orchestration_plan",
)
print(" OrchestratorAgent ready.")

# ----------------------------------------------------
#  WORKER SETUP
# ----------------------------------------------------

worker_agent = Agent(
    name="WorkerAgent",
    model=Gemini(
        model="gemini-2.5-pro",
        retry_options=retry_config
    ),
    instruction="""
    You are the Worker Agent.
    Your task:
    1. Read the clinical note provided in the input field "text". It contains the full raw clinical note.
    2. Extract structured key clinical information (symptoms, labs, treatments, events).
    3. If `feedback` is provided, refine your extraction accordingly.
    Return JSON only:
    {
      "note_id": "",
      "extracted_entities": {},
      "timestamp": "",
      "raw_text_used": true
    }
    """,
    tools=[],
    output_key="note_extractions",
)
print(" WorkerAgent created.")

# worker_agent = Agent(
#     name="WorkerAgent",
#     model=Gemini(
#         model="gemini-2.5-flash-lite",
#         retry_options=retry_config,
#         api_key = GOOGLE_API_KEY
#     ),
#     instruction="""
# You are the Worker Agent.

# You MUST use the ReAct pattern:
# 1. Thought: describe what you need to do
# 2. Action: call one of the provided tools
# 3. Observation: you will receive tool output
# 4. Answer: produce the final structured JSON output

# Your job is to process one or multiple clinical notes and extract structured clinical indicators.

# BUT IMPORTANT:
# When identifying diagnoses, conditions, symptoms, medications, or lab analytes,
# you MUST normalize them using the Cloud Healthcare API tool `terminology_search`.
# Use 'http://snomed.info/sct' for diagnoses, conditions, and symptoms.
# Use 'http://www.nlm.nih.gov/research/umls/rxnorm' for medications.
# Use 'http://loinc.org' for lab analytes.
# Never guess medical codes. Always perform a tool search.

# =========================
# Tasks:
# For each clinical note:
# 1. Extract key indicators:
#    - Vital signs (BP, HR, RR, SpO2, temperature)
#    - Labs (Hb, WBC, platelets, glucose, Cr, Na/K, etc.)
#    - Diagnoses and symptoms (normalized via SNOMED using ReAct + tool calls)
#    - Treatments / medications (normalize via RxNorm)
# 2. Always perform reasoning step-by-step using ReAct.
# 3. Then output the final structured data.

# =========================
# Output Schema:
# Return a JSON list. Example format:

# [
#   {
#     "note_id": "",
#     "visit_date": "",
#     "indicators": {
#         "vitals": {...},
#         "labs": {...}
#     },
#     "diagnosis": [
#        {
#          "text": "",
#          "normalized": {
#             "code": "",
#             "system": "",
#             "display": ""
#          }
#        }
#     ],
#     "treatment": [
#        {
#          "text": "",
#          "normalized": {
#             "code": "",
#             "system": "",
#             "display": ""
#          }
#        }
#     ],
#     "summary": ""
#   }
# ]
# =========================

# You MUST use tools with explicit ReAct steps.
# Do NOT output thoughts in the final answer.
# """,
#     output_key="note_extractions",
#     tools = [FunctionTool(search_fhir_terminology)]

# )
# # run worker agent
# app = App(name="WorkerAgentApp", root_agent=worker_agent)

# # Runner
# test_runner = InMemoryRunner(agent=app.root_agent)


# # Async execution
# async def main():
#     response = await test_runner.run_debug(
#         "test notes..., please extract structured clinical indicators as per the instructions."
#     )
#     final_answers = []
#     for event in response:
#         if event.type == "Answer":
#             final_answers.append(event.content)

#     # Print nicely
#     print(json.dumps(final_answers, indent=2))

# asyncio.run(main())

# ----------------------------------------------------
#  EXTRACTION LOGIC (ITERATIVE)
# ----------------------------------------------------
# 1. Global Session Service (Required by Runner)
session_service = InMemorySessionService()
async def invoke_agent(agent, input_data):
    """
    Wraps the Agent in a Runner to execute it.
    """
    # Create a runner for this specific agent
    APP_NAME = "agents"
    USER_ID = "user_1"
    runner = Runner(agent=agent, session_service=session_service, app_name=APP_NAME)

    # Convert Input to types.Content
    # The runner demands a strict Content object, not a string.
    text_payload = json.dumps(input_data) if isinstance(input_data, dict) else str(input_data)
    
    # create session
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID
    )
    session_id = session.id
    message = types.Content(
        role="user",
        parts=[types.Part(text=text_payload)]
    )

    # Execute (Consume the Generator)
    # We pass 'new_message' as required by the signature
    event_stream = runner.run(
        user_id="user_1",
        session_id=session_id,
        new_message=message
    )

    # extract final output from event stream
    final_text = ""
    
    print(f"   ... streaming response ...")
    
    for event in event_stream:
        # Logic to extract text from the specific Event object
        # Depending on the exact ADK version, text might be in different spots.
        # We check common locations:
        try:
            # Check 1: Direct text attribute
            if hasattr(event, 'text') and event.text:
                final_text = event.text
                
            # Check 2: Nested content/parts (Standard GenAI structure)
            elif hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts'):
                    final_text = event.content.parts[0].text
                    
            # Check 3: If it's a dictionary (unlikely but possible in some versions)
            elif isinstance(event, dict):
                final_text = event.get('text', final_text)
                
        except Exception:
            continue

    return final_text


# 2. Extraction with Feedback Loop
async def execute_worker_with_feedback_loop(note_id: str):
    """
    Manages the Worker Agent -> Evaluator -> Feedback Loop.
    """
    ground_truth = get_ground_truth(note_id)
    raw_text = get_note_text(note_id)
    
    if not raw_text:
        print(f"   ⚠️ Warning: No text found for {note_id}")
        return {"status": "failed", "data": None}
    
    current_feedback = None
    max_retries = 3
    threshold = 0.6
    
    print(f"\n--- [Worker] Processing Note: {note_id} ---")

    for attempt in range(1, max_retries + 1):
        print(f"   > Attempt {attempt}/{max_retries} (Feedback: {current_feedback if current_feedback else 'None'})")
        
        # 1. Call Worker Agent
        # We pass feedback into the prompt context via the dictionary
        agent_input = {
            "note_id": note_id, 
            "text": raw_text,
            "feedback": current_feedback
        }
        
        # The agent returns a string or JSON. We parse it.
        try:
            response_obj = await invoke_agent(worker_agent, agent_input)
            
            # ADK often returns a result object; ensure we get the text/json content
            # Assuming response_obj is the JSON dictionary from the agent
            extracted_data = response_obj 
            
        except Exception as e:
            print(f"   > Error running worker: {e}")
            continue

        # 2. Evaluate (Embedding Similarity Check)
        # Replace this call with your actual `evaluation_tool` or API call
        extracted_str = json.dumps(extracted_data, sort_keys=True)
        ground_truth_str = json.dumps(ground_truth, sort_keys=True)
        evaluation_result = evaluate(ground_truth_str, extracted_str, use_embedding=True)

        print(f"   > Evaluation Score: {evaluation_result['embedding_similarity']:.2f}")

        # 3. Check Threshold
        if evaluation_result['embedding_similarity'] >= threshold:
            print(f"   > SUCCESS: High similarity achieved.")
            return {
                "status": "approved",
                "data": extracted_data,
                "score": evaluation_result['embedding_similarity']
            }
        
        # 4. Generate Feedback for Next Loop
        current_feedback = (
            f"Previous extraction accuracy was {evaluation_result['embedding_similarity']:.2f} (Threshold: {threshold}). "
            f"Issues detected: {', '.join(evaluation_result['feedback']['missing_points'])}. "
            "Please strictly look for missing clinical entities."
        )

    print(f"   > FAILURE: Max retries reached for note {note_id}.")
    return {"status": "failed", "data": None}

# ----------------------------------------------------
#  SYNTHESIZER AGENT
# ----------------------------------------------------

synthesizer_agent = Agent(
    name="SynthesizerAgent",
    model=Gemini(
        model="gemini-3-pro-preview",
        retry_options=retry_config
    ),
    instruction="""You receive multiple structured JSON outputs from WorkerAgents.
You must:
1. Sort notes chronologically.
2. Identify clinical trends.
3. Produce a 150-word summary.

Return:
{
  "patient_id": "",
  "indicator_trends": {},
  "overall_summary": ""
}""",
    tools=[],
    output_key="patient_summary",
)
print(" SynthesizerAgent created.")


# debug invoke runner
# import inspect
# from google.adk.runners import Runner

# print("\n🔍 INSPECTING RUNNER.RUN SIGNATURE:")
# try:
#     # Inspect the 'run' method of the Runner class
#     sig = inspect.signature(Runner.run)
#     print(f"   def run{sig}")
#     print("   Valid parameters:", list(sig.parameters.keys()))
# except Exception as e:
#     print(f"   Could not inspect: {e}")
# print("----------------------------------------\n")

# ----------------------------------------------------
#  OVERALL WORKFLOW EXECUTION
# ----------------------------------------------------

async def run_clinical_pipeline(patient_id: str):

    available_notes = PATIENT_INDEX.get(int(patient_id), [])
    if not available_notes:
        # Try string lookup if int failed
        available_notes = PATIENT_INDEX.get(str(patient_id), [])
    
    if not available_notes:
        print(f"❌ No notes found in CSV for Patient {patient_id}")
        return

    # ==========================================
    # PHASE 1: ORCHESTRATOR (Planning)
    # ==========================================
    print(f"🔵 [Orchestrator] Planning for Patient: {patient_id}")
    
    # Orchestrator decides which notes to pull
    orchestrator_input = {
        "patient_id": patient_id,
        "available_note_ids": available_notes
    }
    
    # Run Orchestrator
    plan_result = await invoke_agent(orchestrator_agent, orchestrator_input)
    
    # Fallback parsing if simple string returned
    target_notes = available_notes
    if isinstance(plan_result, dict) and "note_ids" in plan_result:
        target_notes = plan_result["note_ids"]

    # ==========================================
    # PHASE 2: WORKER LOOPS (Execution)
    # ==========================================
    collected_extractions = []

    for note_id in target_notes:
        result = await execute_worker_with_feedback_loop(note_id)
        
        if result["status"] == "approved":
            collected_extractions.append(result["data"])
        else:
            print(f"   > Skipping note {note_id} due to low quality extraction.")

    if not collected_extractions:
        print("🔴 Pipeline stopped: No approved extractions available.")
        return

    # Save the collected approved extractions to a file
    try:
        output_filename = f"/Users/yixinshen/Agent/approved_extractions_{patient_id}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(collected_extractions, f, indent=4, ensure_ascii=False)
        print(f"💾 Approved worker outputs saved to: {output_filename}")
    except Exception as e:
        print(f"   -> Error saving extractions: {e}")

    # ==========================================
    # PHASE 3: SYNTHESIZER (Summary)
    # ==========================================
    print(f"\n🟢 [Synthesizer] Aggregating {len(collected_extractions)} approved notes...")
    
    summary = await invoke_agent(synthesizer_agent, {
        "patient_id": patient_id,
        "approved_outputs": collected_extractions
    })
    
    print(json.dumps(summary, indent=2))
    
    print("\n================ FINAL REPORT ================")
    print(summary)
    print("==============================================")

    return summary

# Run!
async def main():
    await run_clinical_pipeline(test_patient_id)


if __name__ == "__main__":
    files_loaded = load_test_data(
        "/Users/yixinshen/Agent/single_record.csv",
        "/Users/yixinshen/Agent/single_truth.json"
    )

    if files_loaded:
        test_patient_id = list(PATIENT_INDEX.keys())[0]
        print(f"\n🚀 Starting Pipeline for Test Patient: {test_patient_id}")

        # run the async main
        asyncio.run(main())
