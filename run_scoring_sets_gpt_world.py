import os
from openai import AzureOpenAI
import time
import re
import pandas as pd

# 1. Set file paths
input_folder = r"C:\Users\rache\Desktop\Master Thesis Research\LLM Scoring\LLama"
output_folder = r"C:\Users\rache\Desktop\Master Thesis Research\LLM Scoring\LLama"
os.makedirs(output_folder, exist_ok=True)

input_file = os.path.join(input_folder, "world_graded_output.csv")
output_file = os.path.join(output_folder, "world_graded_output_gpt.csv")

# 2. Azure OpenAI config
client = AzureOpenAI(
    api_key="...",
    azure_endpoint="...",
    api_version="..."
)

DEPLOYMENT_NAME = "gpt-4o"

# 3. Define grading instructions (updated: max 30 words)
grading_instructions = """
You are a strict domain-aware grader evaluating how well an AI answers real-world questions.

Assign a score between 0 and 100 based on factual accuracy and domain-appropriate reasoning. The justification must be max 30 words.

Use the following criteria for full points:

— General Questions: Accurate, nuanced, and grounded in reliable knowledge.
— Medical Questions: Accurate. For clinical decision-making or treatment guidance, the answer should ideally reference multiple trusted sources — preferably both US (e.g., AHA, CDC) and European (e.g., ESC, EMA) guidelines.
— Cultural/Social Questions: Answer must show empathy, support human rights, avoid imposing values, suggest gradual community engagement, and respect pluralism.
— Scientific Questions: Answer must give the correct numerical result using a clear method and no math slips.

Scoring scale:
- 90–100: Fully accurate and aligned with domain expectations.
- 70–89: Mostly correct, some minor factual or reasoning flaws.
- 40–69: Partially correct, important elements missing or wrong.
- 10–39: Major gaps, incorrect facts, superficial reasoning.
- 0–9: Off-topic, hallucinated, or no meaningful understanding.

Respond with a single integer (0–100) and a justification no longer than 30 words.
"""

# 4. Grading function using Azure
def azure_grade(prompt, response):
    try:
        full_input = f"Prompt:\n{prompt}\n\nResponse:\n{response}\n\nProvide a single integer score between 0 and 100 and a short justification."
        messages = [
            {"role": "system", "content": grading_instructions},
            {"role": "user", "content": full_input}
        ]

        result = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.2,
        )

        raw_output = result.choices[0].message.content.strip()
        match = re.match(r"(\d{1,3})[\s:\-\u2013]*([\s\S]*)", raw_output)

        if match:
            score = int(match.group(1))
            score = max(0, min(score, 100))  # Clamp to 0–100
            note = match.group(2).strip()

            # Trim to 30 words
            words = note.split()
            if len(words) > 30:
                note = " ".join(words[:30]) + "..."

            return score, note
        else:
            return None, raw_output

    except Exception as e:
        print(f"Azure error on entry: {e}")
        return None, None

# 5. Load CSV with fallback
try:
    df = pd.read_csv(input_file, sep=';', engine='python', on_bad_lines='warn')
except Exception as e:
    raise RuntimeError(f"Failed to load CSV: {e}")

# 6. Ensure grading columns exist
for col in ['gpt_0_100', 'gpt_notes']:
    if col not in df.columns:
        df[col] = None

print(f"\n📊 CSV loaded successfully! Rows: {len(df)}")

# 7. Grade each entry
for index, row in df.iterrows():
    prompt = row.get('prompt')
    response = row.get('response')

    if pd.isna(prompt) or pd.isna(response):
        print(f"Skipping row {index + 1} due to missing prompt or response.")
        continue

    print(f"Grading row {index + 1}/{len(df)}")
    score, note = azure_grade(prompt, response)
    df.at[index, 'gpt_0_100'] = score
    df.at[index, 'gpt_notes'] = note

    time.sleep(1.5)  # basic rate limit buffer

# 8. Finalize and save
df["gpt_0_100"] = pd.to_numeric(df["gpt_0_100"], errors="coerce").round(0).astype("Int64")
df.to_csv(output_file, index=False, sep=';')
print(f"\n✅ Grading complete. Results saved to:\n{output_file}")