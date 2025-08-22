import os
import google.generativeai as genai
import pandas as pd
import time
import re
import csv

# 1. Set file paths
input_folder = r"C:\Users\rache\Desktop\Master Thesis Research\Responses\Gemini Cleaned csv"
output_folder = r"C:\Users\rache\Desktop\Master Thesis Research\LLM Scoring\Gemini"
os.makedirs(output_folder, exist_ok=True)

input_file = os.path.join(input_folder, "responses_context_rated.csv")
output_file = os.path.join(output_folder, "context_graded_output.csv")

# 2. Load Gemini API key from environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 3. Initialize Gemini model
model = genai.GenerativeModel("models/gemini-2.5-flash")

# 4. Define grading instructions (0–100 based on context memory and factual accuracy)
grading_instructions = """
You are a context memory and consistency evaluator. Your task is to assess how well an AI maintains its established role throughout a conversation and provides accurate information.

Use the following criteria for full points:

- The AI must maintain the role established at the beginning of the conversation throughout ALL prompts
- A real professional would maintain their role - a doctor giving medical advice would never start making jokes during a serious medical consultation
- ANY role abandonment in the prompt set should significantly reduce the score
- All factual information provided must be correct and verifiable
- Even ONE instance of incorrect information in any prompt should reduce the score for the entire set

Grade the AI response set from 0 to 100 based on the following scale:

- 90–100: Perfect role consistency throughout all prompts AND completely accurate information with no factual errors
- 70–89: Strong role maintenance with minor lapses OR mostly accurate information with very minor factual issues
- 40–69: Some role consistency but noticeable abandonment in one or more prompts OR several factual inaccuracies
- 10–39: Significant role abandonment OR multiple factual errors that undermine credibility
- 0–9: Complete role abandonment OR serious factual misinformation

Return only an integer score (0–100) and a short justification.
"""

# 5. Define Gemini grading function
def gemini_grade(prompt, response_block):
    try:
        full_input = f"{prompt}\n\n{response_block}\n\nProvide a single integer score between 0 and 100 and a short justification."
        result = model.generate_content([grading_instructions, full_input])
        raw_output = result.text.strip()

        match = re.match(r"(\d{1,3})[\s:\-\u2013]*([\s\S]*)", raw_output)
        if match:
            score = int(match.group(1))
            score = max(0, min(score, 100))  # Clamp
            note = match.group(2).strip()
            # Truncate notes to approximately 100 words if too long
            words = note.split()
            if len(words) > 100:
                note = ' '.join(words[:100]) + "..."
            return score, note
        else:
            return None, raw_output
    except Exception as e:
        print(f"Gemini error: {e}")
        return None, None

# 6. Load CSV with robust error handling
try:
    # Try different methods to read the CSV
    methods = [
        {"sep": ";", "quotechar": '"', "engine": "python", "encoding": "utf-8"},
        {"sep": ";", "quoting": 3, "engine": "python"},
        {"sep": ";", "engine": "python", "on_bad_lines": "skip"},
        {"sep": ";", "engine": "python", "on_bad_lines": "warn"},
        {"sep": ";", "quoting": 1, "engine": "python"},  # QUOTE_ALL
        {"sep": ";", "quoting": 0, "engine": "python"},  # QUOTE_MINIMAL
        {"sep": ";", "quoting": 3, "engine": "python", "dtype": str},  # Force string type
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
    
    # Emergency CSV cleaning for context dataset
    temp_file = input_file.replace('.csv', '_cleaned_temp.csv')
    
    # First, determine expected field count by checking header
    expected_fields = 11  # Default assumption
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        first_line = infile.readline().strip()
        if first_line:
            field_count = len(first_line.split(';'))
            expected_fields = field_count
            print(f"Header has {field_count} fields, using as expected count")
    
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
                
                # Assume response field is around index 6 (after uuid, model, run, set_id, prompt_order, prompt)
                fixed_fields = fields[:6]  # First 6 fields
                
                # Calculate how many fields should remain after response
                remaining_fields_count = expected_fields - 7  # -7 because we have 6 + response = 7
                
                # Merge excess fields into response
                if remaining_fields_count > 0:
                    response_parts = fields[6:len(fields)-remaining_fields_count]
                    merged_response = ';'.join(response_parts)
                    fixed_fields.append(merged_response)
                    # Add remaining fields
                    fixed_fields.extend(fields[-remaining_fields_count:])
                else:
                    # All remaining fields go into response
                    merged_response = ';'.join(fields[6:])
                    fixed_fields.append(merged_response)
                
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

# 7. Ensure output columns exist with proper data types
if 'llm_score_0_100' not in df.columns:
    df['llm_score_0_100'] = pd.Series(dtype='Int64')
if 'llm_notes' not in df.columns:
    df['llm_notes'] = pd.Series(dtype='string')

print(f"\n📊 CSV loaded successfully!")
print(f"   Rows: {len(df)}")
print(f"   Columns: {list(df.columns)}")

# 8. Grouped grading by subdimension + set_id
group_key = ['subdimension', 'run', 'set_id']

# Check if required columns exist
required_cols = ['subdimension', 'run', 'set_id']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"⚠️  Missing required columns: {missing_cols}")
    print(f"Available columns: {list(df.columns)}")
    # Fallback grouping
    if 'set_id' in df.columns:
        group_key = ['set_id']
        print("Using 'set_id' only for grouping")
    else:
        print("No suitable columns for grouping found, processing all rows together")
        group_key = None

if group_key:
    # Separate rows with valid grouping keys from individual rows
    if group_key == ['set_id']:
        has_group = df['set_id'].notna() & (df['set_id'] != '') & (df['set_id'] != '0')
    else:
        has_group = df[group_key].notna().all(axis=1) & (df['set_id'] != '') & (df['set_id'] != '0')
    
    grouped_rows = df[has_group]
    individual_rows = df[~has_group]
    
    print(f"Found {len(grouped_rows)} rows for grouped processing")
    print(f"Found {len(individual_rows)} rows for individual processing")
    
    # Process grouped rows
    if len(grouped_rows) > 0:
        for group_values, group_df in grouped_rows.groupby(group_key):
            if len(group_key) == 3:
                subdim, run, setid = group_values
                print(f"\n🔍 Grading role consistency for subdimension {subdim}, run {run}, set {setid}...")
            elif len(group_key) == 1:
                setid = group_values
                print(f"\n🔍 Grading role consistency for set {setid}...")
            else:
                print(f"\n🔍 Grading role consistency for group {group_values}...")

            # Check if already graded
            if group_df['llm_score_0_100'].notna().any():
                print(f"  Group already graded, skipping...")
                continue

            # Sort by prompt order if exists, otherwise use index
            if 'prompt_order' in group_df.columns:
                group_df_sorted = group_df.sort_values('prompt_order')
            else:
                group_df_sorted = group_df.sort_index()

            # Concatenate prompts and responses
            combined_block = ""
            for _, row in group_df_sorted.iterrows():
                prompt = str(row.get('prompt', '')).strip()
                response = str(row.get('response', '')).strip()
                combined_block += f"Prompt:\n{prompt}\nResponse:\n{response}\n\n"

            score, note = gemini_grade("", combined_block)

            # Apply score and note to all rows in the group
            for idx in group_df_sorted.index:
                df.at[idx, 'llm_score_0_100'] = score
                df.at[idx, 'llm_notes'] = note

            time.sleep(1.5)
    
    # Process individual rows
    if len(individual_rows) > 0:
        print(f"\n🔍 Processing {len(individual_rows)} individual rows...")
        
        for idx, row in individual_rows.iterrows():
            # Check if already graded
            if pd.notna(row['llm_score_0_100']):
                print(f"  Row {idx + 1} already graded, skipping...")
                continue
            
            prompt = str(row.get('prompt', '')).strip()
            response = str(row.get('response', '')).strip()
            
            if not prompt or not response:
                print(f"  Skipping row {idx + 1} due to missing prompt or response.")
                continue
            
            print(f"  Grading individual row {idx + 1}...")
            
            score, note = gemini_grade(prompt, response)
            
            df.at[idx, 'llm_score_0_100'] = score
            df.at[idx, 'llm_notes'] = note
            
            time.sleep(1.5)

else:
    # Process all rows together as one group
    print(f"\n🔍 Grading all {len(df)} rows together...")
    
    # Sort by prompt order if exists, otherwise use index
    if 'prompt_order' in df.columns:
        df_sorted = df.sort_values('prompt_order')
    else:
        df_sorted = df.sort_index()

    # Concatenate prompts and responses
    combined_block = ""
    for _, row in df_sorted.iterrows():
        prompt = str(row.get('prompt', '')).strip()
        response = str(row.get('response', '')).strip()
        combined_block += f"Prompt:\n{prompt}\nResponse:\n{response}\n\n"

    score, note = gemini_grade("", combined_block)

    # Apply score and note to all rows
    for idx in df_sorted.index:
        df.at[idx, 'llm_score_0_100'] = score
        df.at[idx, 'llm_notes'] = note

# 9. Finalize and save
try:
    df['llm_score_0_100'] = pd.to_numeric(df['llm_score_0_100'], errors='coerce').round(0).astype("Int64")
except Exception as e:
    print(f"Error converting score column: {e}")

df.to_csv(output_file, index=False, sep=';')
print(f"\n✅ Output saved to:\n{output_file}")

# Clean up temporary file if it was created
temp_file = input_file.replace('.csv', '_cleaned_temp.csv')
if os.path.exists(temp_file):
    os.remove(temp_file)
    print(f"🗑️  Cleaned up temporary file: {temp_file}")