# LLM Evaluation - Master Thesis

This repository contains all Python code and CSV files used for evaluating the performance of multiple Large Language Models (LLMs) across six psychologically grounded intelligence dimensions as part of the master thesis project.

## Description

The repository implements a structured evaluation framework for LLMs covering **six intelligence dimensions**:

1. Mathematical-Logical Reasoning  
2. Linguistic Intelligence  
3. Factual World Knowledge  
4. Social & Emotional Intelligence  
5. Ethical Robustness & Safety  
6. Context Memory & Consistency  

Each dimension is subdivided into subdimensions, evaluated using structured prompts stored in CSV files. Every CSV corresponds to a single dimension and LLM, containing:

- `prompt`: The evaluation prompt text.  
- `model_response`: LLM-generated response.  
- scoring columns: used for human ratings & automated scoring.

---

### **Description of Scripts**
- **run_prompt_sets_.py**: Sends prompts of 1 specific dimension from CSV files to specified LLM APIs and stores responses.  
- **run_scoring_sets_*.py**: Evaluate LLM responses for a given set of prompts. Works with both **Gemini** and **GPT** models.   
- **icc_*.py**: Calculates inter-rater reliability and other evaluation metrics (ICC, Kappa, etc.) for the LLM evaluations.  


