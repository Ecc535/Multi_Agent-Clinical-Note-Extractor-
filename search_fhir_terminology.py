import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import os

PROJECT = "focal-elf-478721-n1"
LOCATION = "us-central1"
DATASET = "my_dataset"   # Updated: uses underscore
FHIR_STORE = "icd10"

TERMINOLOGY_BASE = (
    f"https://healthcare.googleapis.com/v1/projects/{PROJECT}/locations/{LOCATION}"
    f"/datasets/{DATASET}/fhirStores/{FHIR_STORE}/fhir"
)

def get_token():
    credentials = service_account.Credentials.from_service_account_file(
        "service-account.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    return credentials.token


def terminology_search(query, system):
    """
    CORRECTED: Uses ValueSet/$expand for text search.
    """
    token = get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/fhir+json"
    }

    # CRITICAL CHANGE: 
    # Use 'ValueSet/$expand' for searching text (e.g. "heart failure").
    # Do NOT use 'CodeSystem/$lookup' (that is for finding details of a specific code like "I50.9").
    url = f"{TERMINOLOGY_BASE}/ValueSet/$expand"
    
    # We target the 'implicit value set' of the system (system + "?fhir_vs")
    # This tells FHIR to search ALL codes in that system.
    value_set_url = f"{system}?fhir_vs" 

    params = {
        "url": value_set_url, 
        "filter": query,      # The text you are searching for
        "count": 5            
    }

    print(f"DEBUG: Sending GET to {url} with filter='{query}'...")
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        # Print the URL that actually failed to help debugging
        print(f"Failed URL: {url}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# simple test call
if __name__ == "__main__":
    # Correct System URL for ICD-10-CM (The US Clinical Modification standard)
    # If this returns 0 results, try the generic one: "http://hl7.org/fhir/sid/icd-10"
    ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10-cm"
    
    print("--- Testing ICD-10 search: 'heart failure' ---")
    
    result = terminology_search(
        query="heart failure",
        system=ICD10_SYSTEM
    )
    
    if result:
        print("\nParsed JSON (Matches found):")
        expansion = result.get('expansion', {}).get('contains', [])
        
        for item in expansion: 
            print(f"- Code: {item.get('code')} | Display: {item.get('display')}")
            
        if not expansion:
            print("❌ Request succeeded (200 OK), but the list is empty.")
            print("Tip: Ensure your ICD-10 CodeSystem is successfully imported into the store.")
    else:
        print("❌ Search failed.")
