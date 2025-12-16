import numpy as np
from rapidfuzz import fuzz
import google.generativeai as genai
import os
import typing
import json
from dotenv import load_dotenv
import re
import math

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


class ClinicalReadabilityEvaluator:
    """
    A tool to evaluate the readability of clinical texts (discharge summaries, 
    patient education materials, etc.).
    
    It implements:
    1. SMOG Index (Gold standard for healthcare materials).
    2. Flesch-Kincaid Grade Level.
    3. Medical Jargon Density (Custom heuristic for terminology).
    """

    def __init__(self):
        # Common medical suffixes/prefixes to help identify jargon
        # without requiring a massive external dictionary dependency.
        self.medical_morphemes = [
            'itis', 'osis', 'ectomy', 'otomy', 'scopy', 'plasty', 'ology', 
            'pathy', 'algia', 'rrhea', 'dys', 'hyper', 'hypo', 'tachy', 
            'brady', 'intra', 'extra', 'cardio', 'neuro', 'hema', 'gastro'
        ]

    def _count_syllables(self, word):
        """
        Heuristic syllable counter. 
        Counts vowel groups, handling silent 'e' and common endings.
        """
        word = word.lower()
        if len(word) <= 3:
            return 1
        
        # Remove common silent endings
        word = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', word)
        
        # Count vowel groups
        syllables = len(re.findall(r'[aeiouy]+', word))
        
        return max(1, syllables)

    def _get_text_stats(self, text):
        """
        Parses text into sentences and words, calculating raw counts.
        """
        # clean text
        text = text.strip()
        
        # Split into sentences (handles basic punctuation)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return None

        words = re.findall(r'\b\w+\b', text)
        num_words = len(words)
        
        complex_words = 0 # Words with 3+ syllables (for SMOG)
        total_syllables = 0
        jargon_count = 0

        for w in words:
            syllable_count = self._count_syllables(w)
            total_syllables += syllable_count
            
            if syllable_count >= 3:
                complex_words += 1
            
            # Check for medical jargon heuristics
            if any(w.lower().endswith(s) or w.lower().startswith(s) for s in self.medical_morphemes):
                # Only count as jargon if it's also reasonably long (not just "dys")
                if len(w) > 5:
                    jargon_count += 1

        return {
            "num_sentences": num_sentences,
            "num_words": num_words,
            "num_complex_words": complex_words,
            "total_syllables": total_syllables,
            "num_jargon": jargon_count
        }

    def calculate_smog_index(self, stats):
        """
        Calculates SMOG (Simple Measure of Gobbledygook).
        Preferred for healthcare consumer materials.
        Formula: 1.0430 * sqrt(complex_words * (30 / sentences)) + 3.1291
        """
        if stats['num_sentences'] == 0: return 0
        
        # SMOG is technically designed for 30 sentences, but we scale it mathematically
        # for shorter texts to get an approximation.
        try:
            slope = stats['num_complex_words'] * (30 / stats['num_sentences'])
            score = 1.0430 * math.sqrt(slope) + 3.1291
            return round(score, 2)
        except ValueError:
            return 0

    def calculate_flesch_kincaid(self, stats):
        """
        Calculates Flesch-Kincaid Grade Level.
        Formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
        """
        if stats['num_words'] == 0 or stats['num_sentences'] == 0: return 0
        
        avg_sentence_len = stats['num_words'] / stats['num_sentences']
        avg_syllables_word = stats['total_syllables'] / stats['num_words']
        
        score = (0.39 * avg_sentence_len) + (11.8 * avg_syllables_word) - 15.59
        return round(score, 2)

    def calculate_jargon_density(self, stats):
        """
        Calculates percentage of words identified as medical jargon.
        """
        if stats['num_words'] == 0: return 0
        return round((stats['num_jargon'] / stats['num_words']) * 100, 2)

    def evaluate(self, text):
        """
        Main entry point. Returns a dictionary containing all metrics.
        """
        stats = self._get_text_stats(text)
        
        if not stats:
            return {"error": "Text is empty or contains no valid sentences."}

        smog = self.calculate_smog_index(stats)
        fk_grade = self.calculate_flesch_kincaid(stats)
        jargon_pct = self.calculate_jargon_density(stats)

        # Interpretation logic
        assessment = "Appropriate for General Public"
        if smog > 14 or jargon_pct > 15:
            assessment = "Professional/Academic Level (Too complex for patients)"
        elif smog > 9:
            assessment = "Difficult (Requires High School education)"
        elif smog < 6:
            assessment = "Very Easy (Appropriate for low literacy)"

        return {
            "metrics": {
                "SMOG_Index": smog,
                "Flesch_Kincaid_Grade": fk_grade,
                "Jargon_Density_Percent": jargon_pct
            },
            "statistics": stats,
            "assessment": assessment
        }
    

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