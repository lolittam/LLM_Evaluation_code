import os
import google.generativeai as genai
import pandas as pd
import time
import re

# 1. Set file paths
input_folder = r"C:\Users\rache\Desktop\Master Thesis Research\Responses\Deepseek Cleaned csv"
output_folder = r"C:\Users\rache\Desktop\Master Thesis Research\LLM Scoring\Deepseek"
os.makedirs(output_folder, exist_ok=True)

input_file = os.path.join(input_folder, "responses_world_rated.csv")
output_file = os.path.join(output_folder, "world_graded_output.csv")

# 2. Load Gemini API key from environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 3. Initialize Gemini model
model = genai.GenerativeModel("models/gemini-2.5-flash")

# 4. Define grading instructions (0–100 based on Smarter Balanced rubric)
grading_instructions = """
You are a strict domain-aware grader evaluating how well an AI answers real-world questions.

Assign a score between 0 and 100 based on factual **accuracy** and domain-appropriate reasoning. The justification must be **max 100 words**.

Use the following criteria for full points:

— General Questions: Accurate, nuanced, and grounded in reliable knowledge.
— Medical Questions: Accurate. For clinical decision-making or treatment guidance, the answer should ideally reference multiple trusted sources — preferably both US (e.g., AHA, CDC) and European (e.g., ESC, EMA) guidelines.
— Cultural/Social Questions: Answer must show empathy, support human rights, avoid imposing values, suggest gradual community engagement, and respect pluralism.
— Scientific Questions: Answer must give the **correct numerical result** using a clear method and no math slips.

Scoring scale:
- 90–100: Fully accurate and aligned with domain expectations.
- 70–89: Mostly correct, some minor factual or reasoning flaws.
- 40–69: Partially correct, important elements missing or wrong.
- 10–39: Major gaps, incorrect facts, superficial reasoning.
- 0–9: Off-topic, hallucinated, or no meaningful understanding.

Respond with a **single integer (0–100)** and a **justification no longer than 100 words**.
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

            # Trim justification to ~100 words max (soft limit)
            words = note.split()
            if len(words) > 100:
                note = " ".join(words[:100]) + "..."

            return score, note
        else:
            return None, raw_output
    except Exception as e:
        print(f"Gemini error on entry: {e}")
        return None, None

# 6. Load CSV with robust error handling
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
    
    # Emergency CSV cleaning for world dataset (expecting 12 fields)
    import csv
    temp_file = input_file.replace('.csv', '_cleaned_temp.csv')
    
    # First, let's determine how many fields we should expect
    expected_fields = 12  # Based on the error message
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        # Check the first few lines to confirm field count
        first_line = infile.readline().strip()
        if first_line:
            field_count = len(first_line.split(';'))
            print(f"First line has {field_count} fields")
            if field_count == 12:
                expected_fields = 12
            elif field_count == 11:
                expected_fields = 11
            else:
                print(f"⚠️  Unexpected field count in header: {field_count}")
    
    print(f"Using expected field count: {expected_fields}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(temp_file, 'w', newline='', encoding='utf-8') as outfile:
        
        csv_writer = csv.writer(outfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            fields = line.split(';')
            
            # If we have more than expected fields, merge excess into the response field
            if len(fields) > expected_fields:
                print(f"Fixing line {line_num}: {len(fields)} fields -> {expected_fields} fields")
                
                # For world dataset, response is likely field 6 (0-indexed: 5)
                # Structure: uuid;model;run;set_id;prompt_order;prompt;response;human_scores;human_notes;gemini_score_0_100;gemini_notes;(possible extra field)
                
                if expected_fields == 12:
                    # Take first 6 fields (uuid through prompt)
                    fixed_fields = fields[:6]
                    # Merge excess fields into response field
                    response_parts = fields[6:len(fields)-(expected_fields-7)]
                    merged_response = ';'.join(response_parts)
                    fixed_fields.append(merged_response)
                    # Add remaining fields
                    fixed_fields.extend(fields[len(fields)-(expected_fields-7):])
                else:
                    # Handle 11-field case
                    fixed_fields = fields[:6]
                    response_parts = fields[6:len(fields)-4]
                    merged_response = ';'.join(response_parts)
                    fixed_fields.append(merged_response)
                    fixed_fields.extend(fields[-4:])
                
                # Ensure exactly the expected number of fields
                while len(fixed_fields) < expected_fields:
                    fixed_fields.append('')
                if len(fixed_fields) > expected_fields:
                    fixed_fields = fixed_fields[:expected_fields]
                    
                csv_writer.writerow(fixed_fields)
                
            elif len(fields) < expected_fields:
                # Pad with empty fields
                while len(fields) < expected_fields:
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