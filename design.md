# Design Document

## System Architecture

This project implements a **LangGraph-based clinical note extraction agent system** that transforms unstructured clinical text into **standards-compliant, structured clinical entities with values**, while ensuring privacy, correctness, and iterative quality improvement.

The system is designed as a **multi-node, stateful agent workflow** with shared orchestration, caching, terminology normalization, validation, and human-in-the-loop feedback. Each node is responsible for a clearly scoped task and communicates via a shared agent state managed by LangGraph.

## Design Diagram Overview
<p align="center">
  <img src="Design_Diagram.png" width="1600">
</p>

## LangGraph Orchestration Layer

The system is orchestrated using **LangGraph**, which manages:
- Node execution order
- Shared agent state across nodes
- Retry logic and conditional branching
- Integration with evaluation and improvement loops

### Shared Agent State
- Extracted entities and values
- Normalized terminology candidates
- Validation scores and metadata
- Retry counters and performance metrics

This shared state enables coordination across nodes without tight coupling.

---

## Input Processing

### Format & PHI Masking

Before any LLM inference:
- Raw clinical text is normalized into a consistent format
- Protected Health Information (PHI) is masked or removed
- Ensures compliance with healthcare privacy requirements (e.g., HIPAA)

## Extract Node (LLM_1)
### Purpose
Extract clinically relevant entities and their associated values from unstructured clinical notes.

### Implementation
- Implemented as a LangGraph node
- Uses LLM inference (Vertex AI)
- Supports asynchronous execution for batch processing

---

## Cache Layer (BigQuery)

### Purpose
Reduce redundant computation and LLM calls.

### Mechanism
- A hash of the formatted input is computed
- If a matching result exists in **BigQuery Cache**:
  - Cached structured output is returned directly
- Otherwise:
  - Workflow proceeds to the Search Node

---

## Search Node (LLM_2)

### Purpose
Lookup and expand standardized medical descriptions for extracted entities for further validation.

#### FHIR Terminology Normalization Tool
- Maps extracted entities to standard codes
- Returns normalized identifiers and preferred terms

#### Supported Terminologies
- SNOMED CT  
- RxNorm  
- ICD-9  
- Custom ValueSets  

#### Custom ValueSets
- Defined for specific use cases
- Constrain search space during lookup
- Improve precision and reduce ambiguity

### Storage & Memory
- **PostgreSQL**: backs the FHIR terminology server
- **Vector Store**:
  - Stores embeddings of normalized entities
  - Supports semantic memory and reuse across requests

---

## FastAPI Abstraction Layer

A FastAPI service wraps the FHIR terminology server to:

- Handle massive terminology lookup requests
- Normalize request and response schemas
- Decouple agent logic from FHIR server internals

---

## Validate Node (LLM_3)

### Purpose
Ensure extracted and normalized entities meet quality and performance thresholds.

### Validation Metrics

1. **Exact Match**
   - Direct string-level comparison with official terminology names

2. **Token Set Ratio (Fuzzy Matching)**
   - Robust to word reordering and partial overlap
   - Effective for abbreviated or noisy clinical expressions

3. **Semantic Similarity**
   - Embedding-based similarity scoring
   - Captures conceptual equivalence beyond surface text

### Decision Logic
- Entities meeting or exceeding the performance threshold are accepted
- Failed entities trigger retry or escalation logic

---

## Retry & Improvement Logic

### Automatic Retries
- If validation fails and retry count ≤ 5:
  - System logs failure details
  - Re-enters the workflow with updated context

### Improvement Logs
- Validation errors and performance gaps are recorded
- Logs are fed back into orchestration and prompt refinement

---

## Human-in-the-Loop Review

### Trigger Conditions
- Validation failures with retry count > 5
- Low-confidence or ambiguous clinical mappings

### Framework
Aasynchronous Pipeline

### Outcomes
- Human feedback informs:
  - Prompt updates
  - Validation thresholds
  - Terminology constraints

---

## Final Output

### Output Format
- Structured clinical entities with:
  - Standardized codes
  - Normalized names
  - Associated values and units
  - Validation metadata

---

## Summary

This architecture combines **LLMs, FHIR standards, vector memory, caching, and human oversight** into a robust clinical information extraction pipeline. The design emphasizes scalability, correctness, and continuous learning, making it suitable for real-world clinical NLP deployments.


