import csv
import os
import time
import uuid
import httpx
from collections import defaultdict
import traceback

# === Config ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "gemma2-9b-it"
NUM_RUNS = 3

if GROQ_API_KEY is None:
    raise ValueError("Groq API key not set! Please set the GROQ_API_KEY environment variable.")

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

# === Paths ===
input_folder = r"C:\Users\rache\Desktop\Master Thesis Research"
output_folder = os.path.join(input_folder, "Responses")

os.makedirs(output_folder, exist_ok=True)

# === Loop over CSV files ===
for filename in os.listdir(input_folder):
    if not filename.startswith("dimension_") or not filename.endswith(".csv"):
        continue

    dimension_name = filename[len("dimension_"):-len(".csv")]
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"responses_{dimension_name}.csv")

    print(f"\n=== Processing {filename} ===\n")

    # Load prompts
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
        # === Grouped sets: multi-prompt conversations ===
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
                conversation = [{"role": "system", "content": "You are a helpful assistant."}]
                for prompt_order, row in prompts:
                    user_prompt = row["prompt"]
                    conversation.append({"role": "user", "content": user_prompt})

                    try:
                        payload = {
                            "model": MODEL_NAME,
                            "messages": conversation,
                            "temperature": 0.7,
                            "max_tokens": 1000,
                        }

                        response = httpx.post(GROQ_API_URL, headers=headers, json=payload)
                        response.raise_for_status()
                        model_reply = response.json()["choices"][0]["message"]["content"].strip()

                        conversation.append({"role": "assistant", "content": model_reply})

                        output_rows.append({
                            "uuid": str(uuid.uuid4()),
                            "model": MODEL_NAME,
                            "run": run,
                            "set_id": set_id,
                            "prompt_order": prompt_order,
                            "prompt": user_prompt,
                            "response": model_reply
                        })

                    except Exception as e:
                        print(f"Error in set {set_id}, prompt {prompt_order}:\n{traceback.format_exc()}")
                        output_rows.append({
                            "uuid": str(uuid.uuid4()),
                            "model": MODEL_NAME,
                            "run": run,
                            "set_id": set_id,
                            "prompt_order": prompt_order,
                            "prompt": user_prompt,
                            "response": f"ERROR: {str(e)}"
                        })

                    time.sleep(20)

    else:
        # === Single prompt per conversation ===
        for run in range(1, NUM_RUNS + 1):
            print(f"\n=== Run {run} for dimension '{dimension_name}' (single prompt) ===\n")
            for row in rows:
                user_prompt = row["prompt"]

                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt}
                ]

                try:
                    payload = {
                        "model": MODEL_NAME,
                        "messages": conversation,
                        "temperature": 0.7,
                        "max_tokens": 1000,
                    }

                    response = httpx.post(GROQ_API_URL, headers=headers, json=payload)
                    response.raise_for_status()
                    model_reply = response.json()["choices"][0]["message"]["content"].strip()

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

                time.sleep(20)

    # === Save output ===
    fieldnames = ["uuid", "model", "run", "set_id", "prompt_order", "prompt", "response"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\n✅ Finished: {filename} — Results saved to: {output_path}")