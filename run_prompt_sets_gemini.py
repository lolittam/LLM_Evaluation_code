import csv
import os
import time
import uuid
from collections import defaultdict
import traceback
import google.generativeai as genai
from dotenv import load_dotenv

# === Config ===
load_dotenv()
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "models/gemini-2.0-flash"
NUM_RUNS = 3

if GENAI_API_KEY is None:
    raise ValueError("Gemini API key not set! Please set the GEMINI_API_KEY environment variable.")

genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# === Paths ===
input_folder = r"C:\Users\rache\Desktop\Master Thesis Research"
output_folder = os.path.join(input_folder, "Responses_Gemini")
os.makedirs(output_folder, exist_ok=True)

# === Loop over CSV files ===
for filename in os.listdir(input_folder):
    if not filename.startswith("dimension_") or not filename.endswith(".csv"):
        continue

    dimension_name = filename[len("dimension_"):-len(".csv")]
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"responses_{dimension_name}.csv")

    print(f"\n=== Processing {filename} ===\n")

    with open(input_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)

    has_set_structure = (
        "set_id" in reader.fieldnames and
        "prompt_order" in reader.fieldnames and
        any(row.get("set_id", "").strip() and row.get("prompt_order", "").strip().isdigit() for row in rows)
    )

    output_rows = []

    if has_set_structure:
        prompt_sets = defaultdict(list)
        for row in rows:
            try:
                set_id = row["set_id"]
                prompt_order = int(row["prompt_order"])
                prompt_sets[set_id].append((prompt_order, row))
            except ValueError:
                continue

        for run in range(1, NUM_RUNS + 1):
            print(f"\n=== Run {run} for dimension '{dimension_name}' (grouped sets) ===\n")
            for set_id, prompts in prompt_sets.items():
                prompts.sort(key=lambda x: x[0])
                history = []
                try:
                    chat = model.start_chat(history=[])

                    for prompt_order, row in prompts:
                        user_prompt = row["prompt"]
                        response = chat.send_message(user_prompt)
                        model_reply = response.text.strip()

                        output_rows.append({
                            "uuid": str(uuid.uuid4()),
                            "model": MODEL_NAME,
                            "run": run,
                            "set_id": set_id,
                            "prompt_order": prompt_order,
                            "prompt": user_prompt,
                            "response": model_reply
                        })

                        time.sleep(5)

                except Exception as e:
                    print(f"Error in set {set_id}:\n{traceback.format_exc()}")
                    output_rows.append({
                        "uuid": str(uuid.uuid4()),
                        "model": MODEL_NAME,
                        "run": run,
                        "set_id": set_id,
                        "prompt_order": "",
                        "prompt": "",
                        "response": f"ERROR: {str(e)}"
                    })

    else:
        for run in range(1, NUM_RUNS + 1):
            print(f"\n=== Run {run} for dimension '{dimension_name}' (single prompt) ===\n")
            for row in rows:
                user_prompt = row["prompt"]

                try:
                    response = model.generate_content(user_prompt)
                    model_reply = response.text.strip()

                    output_rows.append({
                        "uuid": str(uuid.uuid4()),
                        "model": MODEL_NAME,
                        "run": run,
                        "set_id": "",
                        "prompt_order": "",
                        "prompt": user_prompt,
                        "response": model_reply
                    })

                except Exception as e:
                    print(f"Error in prompt:\n{traceback.format_exc()}")
                    output_rows.append({
                        "uuid": str(uuid.uuid4()),
                        "model": MODEL_NAME,
                        "run": run,
                        "set_id": "",
                        "prompt_order": "",
                        "prompt": user_prompt,
                        "response": f"ERROR: {str(e)}"
                    })

                time.sleep(5)

    # === Save output ===
    fieldnames = ["uuid", "model", "run", "set_id", "prompt_order", "prompt", "response"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\n✅ Finished: {filename} — Results saved to: {output_path}")
