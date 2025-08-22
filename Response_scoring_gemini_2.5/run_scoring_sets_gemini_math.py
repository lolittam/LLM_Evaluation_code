import os
import google.generativeai as genai
import pandas as pd
import time
import re

# 1. Set file paths
input_folder = r"C:\Users\rache\Desktop\Master Thesis Research\Responses\Gemini Cleaned csv"
output_folder = r"C:\Users\rache\Desktop\Master Thesis Research\LLM Scoring\Gemini"
os.makedirs(output_folder, exist_ok=True)

input_file = os.path.join(input_folder, "responses_math_rated.csv")
output_file = os.path.join(output_folder, "math_graded_output.csv")

# 2. Load Gemini API key from environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 3. Initialize Gemini model
model = genai.GenerativeModel("models/gemini-2.5-flash")

# 4. Define grading instructions (0–100 based on Smarter Balanced rubric)
grading_instructions = """
You are a strict mathematical grader. Based only on the correctness and mathematical reasoning of the response, assign a score from 0 to 100 using the following scale:

- 90–100: Full and complete understanding; mathematically sound approach with strong problem solving, reasoning, or modeling. Only minor flaws that don't detract from correctness.
- 70–89: Reasonable understanding; mostly correct, but incomplete. Some errors due to misinterpretation, carelessness, or insufficient explanation. Evidence of competence, but not full understanding.
- 40–69: Partial understanding; some correct elements but key errors in method or reasoning. Fundamental gaps in knowledge or problem-solving ability.
- 10–39: Limited understanding; incomplete response with many flaws. Some attempt made, but reasoning or method is faulty.
- 0–9: No meaningful understanding demonstrated. Response is blank, off-topic, or shows only superficial acquaintance with the task.

Only consider mathematical accuracy and reasoning. Do not reward verbosity or style.
Respond with a single integer score (0–100) and a very short justification.
"""

# 5. Define Gemini grading function (returns only 0-100 score and notes)
def gemini_grade(prompt, response):
    try:
        full_input = f"Prompt:\n{prompt}\n\nResponse:\n{response}\n\nProvide a single integer score between 0 and 100 and a short justification."
        result = model.generate_content([grading_instructions, full_input])
        raw_output = result.text.strip()

        match = re.match(r"(\d{1,3})[\s:\-\u2013]*([\s\S]*)", raw_output)
        if match:
            score = int(match.group(1))
            score = max(0, min(score, 100))  # Clamp to 0–100 range
            note = match.group(2).strip()
            return score, note
        else:
            return None, raw_output
    except Exception as e:
        print(f"Gemini error on entry: {e}")
        return None, None

# 6. Load CSV with error handling for malformed lines
try:
    # Try different methods to read the CSV
    methods = [
        {"sep": ";", "quoting": 3, "engine": "python"},
        {"sep": ";", "engine": "python", "on_bad_lines": "skip"},
        {"sep": ";", "engine": "python", "on_bad_lines": "warn"},
        {"sep": ";", "quoting": 1, "engine": "python"},  # QUOTE_ALL
        {"sep": ";", "quoting": 0, "engine": "python"},  # QUOTE_MINIMAL
    ]
    
    df = None
    for i, method in enumerate(methods):
        try:
            print(f"Trying to read CSV with method {i+1}...")
            df = pd.read_csv(input_file, **method)
            print(f"✅ Successfully loaded CSV with method {i+1}")
            print(f"   Shape: {df.shape}")
            break
        except Exception as e:
            print(f"❌ Method {i+1} failed: {str(e)[:100]}...")
            continue
    
    if df is None:
        raise Exception("All CSV reading methods failed")
        
except FileNotFoundError:
    print(f"Error: Could not find file at {input_file}")
    raise
except Exception as e:
    print(f"Error loading CSV: {e}")
    print("\n🔧 Attempting to clean the CSV file first...")
    
    # Emergency CSV cleaning
    import csv
    temp_file = input_file.replace('.csv', '_cleaned_temp.csv')
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(temp_file, 'w', newline='', encoding='utf-8') as outfile:
        
        csv_writer = csv.writer(outfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            fields = line.split(';')
            
            # If we have more than 11 fields, merge excess into the response field (index 6)
            if len(fields) > 11:
                print(f"Fixing line {line_num}: {len(fields)} fields -> 11 fields")
                fixed_fields = fields[:6]  # First 6 fields
                merged_response = ';'.join(fields[6:len(fields)-4])  # Merge middle fields
                fixed_fields.append(merged_response)
                fixed_fields.extend(fields[-4:])  # Last 4 fields
                
                # Ensure exactly 11 fields
                while len(fixed_fields) < 11:
                    fixed_fields.append('')
                if len(fixed_fields) > 11:
                    fixed_fields = fixed_fields[:11]
                    
                csv_writer.writerow(fixed_fields)
            elif len(fields) < 11:
                # Pad with empty fields
                while len(fields) < 11:
                    fields.append('')
                csv_writer.writerow(fields)
            else:
                csv_writer.writerow(fields)
    
    # Try to load the cleaned file
    print(f"✅ Cleaned CSV saved to: {temp_file}")
    df = pd.read_csv(temp_file, sep=';', engine='python')
    print(f"✅ Successfully loaded cleaned CSV. Shape: {df.shape}")

# 7. Ensure output columns exist
for col in ['gemini_score_0_100', 'gemini_notes']:
    if col not in df.columns:
        df[col] = None

print(f"\n📊 CSV loaded successfully!")
print(f"   Rows: {len(df)}")
print(f"   Columns: {list(df.columns)}")

# 8. Grade each entry
for index, row in df.iterrows():
    prompt = row.get('prompt')
    response = row.get('response')

    if pd.isna(prompt) or pd.isna(response):
        print(f"Skipping row {index + 1} due to missing prompt or response.")
        continue

    print(f"Grading row {index + 1}/{len(df)}")

    score, note = gemini_grade(prompt, response)
    df.at[index, 'gemini_score_0_100'] = score
    df.at[index, 'gemini_notes'] = note

    time.sleep(1.5)  # rate limiting buffer

# 9. Round scores and save output as integers
df["gemini_score_0_100"] = pd.to_numeric(df["gemini_score_0_100"], errors="coerce").round(0).astype("Int64")

df.to_csv(output_file, index=False, sep=';')
print(f"\n✅ Grading complete. Results saved to:\n{output_file}")

# Clean up temporary file if it was created
temp_file = input_file.replace('.csv', '_cleaned_temp.csv')
if os.path.exists(temp_file):
    os.remove(temp_file)
    print(f"🗑️  Cleaned up temporary file: {temp_file}")