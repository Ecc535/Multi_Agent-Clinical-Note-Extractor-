import json
import os

def create_icd10_bundle(input_file, output_file):
    print(f"Reading from: {input_file}")
    
    # --- 1. Generate the CodeSystem (The Dictionary) ---
    concepts = []
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                # Split on the first whitespace only (Code vs. Description)
                parts = line.strip().split(None, 1)
                if len(parts) != 2:
                    continue

                code, display = parts
                concepts.append({
                    "code": code,
                    "display": display
                })
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please check the path.")
        return

    code_system = {
        "resourceType": "CodeSystem",
        "id": "icd10cm-2025",
        "url": "http://hl7.org/fhir/sid/icd-10-cm",
        "version": "2025",
        "name": "ICD10CM",
        "title": "ICD-10-CM 2025",
        "status": "active",
        "content": "complete",
        "count": len(concepts), # Useful metadata for FHIR
        "concept": concepts
    }

    # --- 2. Generate the ValueSet (The Filter) ---
    # This is the "All Codes" ValueSet required for search validation
    value_set = {
        "resourceType": "ValueSet",
        "id": "icd-10-cm-vs",
        "url": "http://hl7.org/fhir/sid/icd-10-cm?fhir_vs", # Matches the search URL
        "version": "2025",
        "name": "ICD10CM_All_Codes",
        "title": "All ICD-10-CM Codes",
        "status": "active",
        "compose": {
            "include": [
                {
                    "system": "http://hl7.org/fhir/sid/icd-10-cm" # Must match CodeSystem URL
                }
            ]
        }
    }

    # --- 3. Wrap in a Single Bundle ---
    bundle = {
        "resourceType": "Bundle",
        "type": "collection", # 'collection' is standard for GCS imports
        "entry": [
            {
                "resource": code_system
            },
            {
                "resource": value_set
            }
        ]
    }

    # --- 4. Write to File ---
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    print(f"Success! Generated '{output_file}' with {len(concepts)} concepts.")

if __name__ == "__main__":
    # Update this path to where your .txt file is located in Cloud Shell
    # If using Cloud Shell, you might need to upload the .txt file first.
    INPUT_PATH = "/Users/yixinshen/Agent/icd10OrderFiles2025_0/icd10cm_codes_2025.txt" 
    OUTPUT_PATH = "icd10_bundle.json"
    
    create_icd10_bundle(INPUT_PATH, OUTPUT_PATH)