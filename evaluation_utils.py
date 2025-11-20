import numpy as np
from rapidfuzz import fuzz
import google.generativeai as genai
import os
import typing
import json
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY or API_KEY == "YOUR_ACTUAL_API_KEY_HERE":
    raise ValueError("GOOGLE_API_KEY is not set or is a placeholder. Please create a .env file and add your key.")

genai.configure(api_key=API_KEY)

# print("Available Models:")
# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(f"- {m.name}")

# ---------------------------------------------
# Helper: cosine similarity
# ---------------------------------------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Prevent Division by Zero
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return np.dot(a, b) / (norm_a * norm_b)

# ---------------------------------------------
# Helper: get text embedding (Gemini model)
# ---------------------------------------------
def get_embedding(text: str):
    # 1. Handle Dictionary Input: Convert to JSON string
    if isinstance(text, dict) or isinstance(text, list):
        text = json.dumps(text)
    
    # 2. Handle Empty Input
    if not text:
        return []

    try:
        # existing code...
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="semantic_similarity"
        )
        return result['embedding']
    except Exception as e:
        print(f"Embedding error: {e}")
        return []


# ---------------------------------------------
# Main evaluation function
# ---------------------------------------------
def evaluate(ground_truth: str, output: str, use_embedding=True):
    results = {}

    # 1. Exact Match
    results["exact_match"] = int(ground_truth.strip() == output.strip())

    # 2. Token Set Ratio (fuzzy match)
    results["token_set_ratio"] = fuzz.token_set_ratio(ground_truth, output) / 100.0

    # 3. Embedding similarity
    if use_embedding:
        emb_gt = get_embedding(ground_truth)
        emb_out = get_embedding(output)
        results["embedding_similarity"] = cosine_similarity(emb_gt, emb_out)

        if results["embedding_similarity"] < 0.6:
            feedback = generate_improvement_feedback(ground_truth, output)
            results["feedback"] = feedback
        else:
            results["feedback"] = None
    else:
        results["embedding_similarity"] = None

    return results

# ---------------------------------------------
# Define Output Schema for Feedback
# ---------------------------------------------
class EvaluationFeedback(typing.TypedDict):
    missing_points: list[str]
    incorrect_points: list[str]
    improvement_advice: list[str]


# ---------------------------------------------
# Helper: Generate Qualitative Feedback (New)
# ---------------------------------------------
def generate_improvement_feedback(ground_truth: str, actual_output: str, model_name="gemini-2.5-pro"):
    """
    Uses an LLM to compare ground truth vs actual output and generate structured feedback.
    Only runs when similarity is low to save costs/latency.
    """
    
    prompt = f"""
    You are an expert evaluator. Compare the Ground Truth and the Actual Output.
    Identify what is missing in the output, what is factually incorrect, and provide specific advice to fix it.
    
    Ground Truth:
    {ground_truth}
    
    Actual Output:
    {actual_output}
    """

    model = genai.GenerativeModel(model_name)
    
    # We force the model to return JSON matching our schema
    result = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=EvaluationFeedback
        )
    )
    
    try:
        return json.loads(result.text)
    except json.JSONDecodeError:
        return {
            "missing_points": ["Error parsing model response"], 
            "incorrect_points": [], 
            "improvement_advice": []
        }