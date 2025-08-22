import os
import google.generativeai as genai
import pandas as pd
import time
import re
import csv

# 1. Set file paths
input_folder = r"C:\Users\rache\Desktop\Master Thesis Research\Responses\Deepseek Cleaned csv"
output_folder = r"C:\Users\rache\Desktop\Master Thesis Research\LLM Scoring\Deepseek"
os.makedirs(output_folder, exist_ok=True)

input_file = os.path.join(input_folder, "responses_linguistic_rated.csv")
output_file = os.path.join(output_folder, "linguistic_graded_output.csv")

# 2. Load Gemini API key from environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 3. Initialize Gemini model
model = genai.GenerativeModel("models/gemini-2.5-flash")

# 4. Define grading instructions (0–100 based on linguistic robustness rubric)
grading_instructions = """
You are a linguistic evaluator assessing responses based on specific linguistic dimensions. Use the following criteria for evaluation:

- For prompts with a distinct word number: Sentences must have seamless flow, all facts present, clear connectives, and proper length; ANY sentences that are jumbled, major logical jumps, or length >120 or <70 words should significantly reduce the score
- Ignore word length if the prompt does not request it
- ANY tone/genre that remains unchanged or wrong OR key information lost/distorted should significantly reduce the score  
- For ambiguous prompts, the AI must explicitly flag ambiguity AND ask for clarification OR supply disambiguated rewrite/answer covering all plausible readings
- Figurative expressions must be original, contextually appropriate, semantically coherent, and clearly metaphorical/rhetorical
- The AI must maintain consistent logical stance across all steps with relevant reasoning and rhetorical coherence
- ANY contradiction, stance reversal, or incoherent argumentative flow should significantly reduce the score

Grade the AI response from 0 to 100 based on the following scale:

- 90–100: Perfect linguistic performance throughout with no significant flaws
- 70–89: Strong linguistic performance with only minor lapses
- 40–69: Adequate linguistic performance but with noticeable issues
- 10–39: Significant linguistic problems that undermine effectiveness
- 0–9: Complete failure to meet linguistic requirements

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
    
    # Emergency CSV cleaning for linguistic dataset
    temp_file = input_file.replace('.csv', '_cleaned_temp.csv')
    
    # First, determine expected field count by checking header
    expected_fields = 9  # Default assumption for linguistic dataset
    
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

# 8. Grouped grading by subdimension + set_id (if available), otherwise by set_id only
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
                print(f"\n🔍 Grading subdimension {subdim}, run {run}, set {setid}...")
            elif len(group_key) == 1:
                setid = group_values
                print(f"\n🔍 Grading set {setid}...")
            else:
                print(f"\n🔍 Grading group {group_values}...")

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
print(f"\n✅ Linguistic scoring complete. Output saved to:\n{output_file}")

# Clean up temporary file if it was created
temp_file = input_file.replace('.csv', '_cleaned_temp.csv')
if os.path.exists(temp_file):
    os.remove(temp_file)
    print(f"🗑️  Cleaned up temporary file: {temp_file}")