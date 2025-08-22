import os
from openai import AzureOpenAI
import time
import re
import pandas as pd

# 1. Set file paths
input_folder = r"C:\Users\rache\Desktop\Master Thesis Research\LLM Scoring\LLama"
output_folder = r"C:\Users\rache\Desktop\Master Thesis Research\LLM Scoring\LLama"
os.makedirs(output_folder, exist_ok=True)

input_file = os.path.join(input_folder, "math_graded_output.csv")
output_file = os.path.join(output_folder, "math_graded_output_gpt.csv")

# 2. Azure OpenAI config
client = AzureOpenAI(
    api_key="...",
    azure_endpoint="...",
    api_version="..."
)

DEPLOYMENT_NAME = "gpt-4o"

# 3. Grading instructions
grading_instructions = """
You are a strict mathematical grader. Based only on the correctness and mathematical reasoning of the response, assign a score from 0 to 100 using this scale:

90–100: Fully correct
70–89: Mostly correct
40–69: Partially correct
10–39: Limited understanding
0–9: No understanding

Respond with a score and brief note (max 30 words).
"""

# 4. Grading function
def gpt_grade(prompt, response):
    try:
        full_input = f"Prompt:\n{prompt}\n\nResponse:\n{response}\n\nReturn a score (0–100) and short note (max 30 words)."
        res = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": grading_instructions},
                {"role": "user", "content": full_input}
            ],
            temperature=0.2,
            max_tokens=150,
        )
        text = res.choices[0].message.content.strip()
        match = re.match(r"(\d{1,3})[\s:\-\u2013]*([\s\S]*)", text)
        if match:
            score = int(match.group(1))
            score = max(0, min(score, 100))
            note = match.group(2).strip()
            if len(note.split()) > 30:
                note = ' '.join(note.split()[:30])
            return score, note
        return None, text
    except Exception as e:
        print(f"Azure error on row: {e}")
        return None, None

# 5. Load CSV (same logic as before)
methods = [
    {"sep": ";", "quoting": 3, "engine": "python"},
    {"sep": ";", "engine": "python", "on_bad_lines": "skip"},
    {"sep": ";", "engine": "python", "on_bad_lines": "warn"},
    {"sep": ";", "quoting": 1, "engine": "python"},
    {"sep": ";", "quoting": 0, "engine": "python"},
]
df = None
for i, method in enumerate(methods):
    try:
        df = pd.read_csv(input_file, **method)
        print(f"✅ Loaded CSV with method {i+1}")
        break
    except Exception as e:
        print(f"❌ Method {i+1} failed: {str(e)[:100]}...")
if df is None:
    raise Exception("All CSV reading methods failed")

# 6. Add output columns if missing
for col in ['gpt_score_0_100', 'gpt_notes']:
    if col not in df.columns:
        df[col] = None

# 7. Grade rows
for i, row in df.iterrows():
    prompt = row.get("prompt")
    response = row.get("response")
    if pd.isna(prompt) or pd.isna(response):
        continue
    print(f"Grading {i+1}/{len(df)}")
    score, note = gpt_grade(prompt, response)
    df.at[i, 'gpt_score_0_100'] = score
    df.at[i, 'gpt_notes'] = note
    time.sleep(1.5)

df["gpt_score_0_100"] = pd.to_numeric(df["gpt_score_0_100"], errors="coerce").round(0).astype("Int64")
df.to_csv(output_file, index=False, sep=";")
print(f"\n✅ Done. Output saved to: {output_file}")