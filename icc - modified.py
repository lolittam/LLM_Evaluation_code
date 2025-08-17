import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pingouin import intraclass_corr

# === Config ===
root_folder = r"C:\Users\rache\Desktop\Master Thesis Research\LLM Scoring"
output_path = os.path.join(root_folder, "irr_summary_table_modified.csv")
results = []

def aggregate_ai_scores(df, dimension):
    """
    Aggregates AI evaluation data across 3 runs depending on dimension logic.

    - linguistic:
        * For rows WITH set_id: take first prompt per set per run, then average across 3 runs
        * For rows WITHOUT set_id: average per individual prompt across 3 runs
        * Combine both aggregated datasets
    - ethics/context:
        * Average score from only the first prompt of each set across 3 runs (grouped by subdimension + set_id)
    - other:
        * Average score per exact prompt text across 3 runs

    Returns:
        Aggregated dataframe with mean GPT and Gemini scores.
    """
    print(f"  → Aggregating scores for dimension: {dimension}")

    df = df.copy()
    df['prompt'] = df['prompt'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

    if dimension == 'linguistic':
        # Check if set_id column exists
        if 'set_id' in df.columns:
            # Separate data with and without set_id
            df_with_set = df[df['set_id'].notna() & (df['set_id'] != '') & (df['set_id'] != 'nan')]
            df_without_set = df[df['set_id'].isna() | (df['set_id'] == '') | (df['set_id'] == 'nan')]
            
            print(f"    → Found {len(df_with_set)} rows with set_id, {len(df_without_set)} rows without set_id")
            
            aggregated_parts = []
            
            # Handle data WITH set_id (first prompt per set per run, then average across runs)
            if len(df_with_set) > 0:
                print("    → Aggregating data WITH set_id: first prompt per set per run")
                df_first = df_with_set.sort_values(['set_id', 'run', 'prompt']).groupby(['set_id', 'run'], as_index=False).first()
                df_set_agg = df_first.groupby('set_id').agg({
                    'gpt_score_0_100': 'mean',
                    'gemini_score_0_100': 'mean'
                }).reset_index()
                aggregated_parts.append(df_set_agg[['gpt_score_0_100', 'gemini_score_0_100']])
            
            # Handle data WITHOUT set_id (average per individual prompt across runs)
            if len(df_without_set) > 0:
                print("    → Aggregating data WITHOUT set_id: per individual prompt across runs")
                df_prompt_agg = df_without_set.groupby('prompt', dropna=False).agg({
                    'gpt_score_0_100': 'mean',
                    'gemini_score_0_100': 'mean'
                }).reset_index()
                aggregated_parts.append(df_prompt_agg[['gpt_score_0_100', 'gemini_score_0_100']])
            
            # Combine all aggregated data
            if aggregated_parts:
                df_agg = pd.concat(aggregated_parts, ignore_index=True)
            else:
                df_agg = pd.DataFrame(columns=['gpt_score_0_100', 'gemini_score_0_100'])
        
        else:
            print("    → No set_id column — averaging per individual prompt across 3 runs")
            df_agg = df.groupby('prompt', dropna=False).agg({
                'gpt_score_0_100': 'mean',
                'gemini_score_0_100': 'mean'
            }).reset_index()

    elif dimension in ['ethics', 'context']:
        if 'subdimension' not in df.columns or 'set_id' not in df.columns:
            raise ValueError(f"{dimension} requires 'subdimension' and 'set_id' columns for aggregation")

        print("    → Aggregating first prompt per (subdimension, set_id) across 3 runs")
        df_first = df.sort_values(['subdimension', 'set_id', 'run', 'prompt']).groupby(
            ['subdimension', 'set_id', 'run'], as_index=False).first()

        df_agg = df_first.groupby(['subdimension', 'set_id']).agg({
            'gpt_score_0_100': 'mean',
            'gemini_score_0_100': 'mean'
        }).reset_index()

    else:
        print("    → Default: averaging per prompt across 3 runs")
        df_agg = df.groupby('prompt', dropna=False).agg({
            'gpt_score_0_100': 'mean',
            'gemini_score_0_100': 'mean'
        }).reset_index()

    print(f"    → Aggregated to {len(df_agg)} rows from {len(df)} raw entries")
    return df_agg

# === Helper: Calculate comprehensive agreement metrics between AI evaluators ===
def calculate_ai_agreement_metrics(gpt_scores, gemini_scores):
    """Calculate comprehensive agreement metrics between GPT-4 and Gemini scores"""
    
    gpt_scores = np.array(gpt_scores)
    gemini_scores = np.array(gemini_scores)
    
    # 1. ICC(3, consistency)
    icc_df = pd.DataFrame({
        "targets": list(range(len(gpt_scores))) + list(range(len(gemini_scores))),
        "rater": ["GPT4"] * len(gpt_scores) + ["Gemini"] * len(gemini_scores),
        "score": list(gpt_scores) + list(gemini_scores)
    })
    icc = intraclass_corr(data=icc_df, targets='targets', raters='rater', ratings='score')
    icc_value = icc[icc["Type"] == "ICC3"]["ICC"].values[0]
    
    # 2. Pearson correlation
    pearson_corr = np.corrcoef(gpt_scores, gemini_scores)[0, 1]
    
    # 3. Spearman correlation (rank-order agreement)
    spearman_corr, _ = spearmanr(gpt_scores, gemini_scores)
    
    # 4. Mean Absolute Error (MAE)
    mae = np.mean(np.abs(gpt_scores - gemini_scores))
    
    # 5. Bland-Altman analysis
    diff = gpt_scores - gemini_scores
    mean_diff = np.mean(diff)  # Systematic bias
    std_diff = np.std(diff, ddof=1)
    
    # Limits of Agreement (LoA)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    
    # 6. Agreement range (95% of differences within ±X points)
    agreement_range = 1.96 * std_diff  # ±range for 95% agreement
    
    return {
        'icc_value': icc_value,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
        'mae': mae,
        'systematic_bias': mean_diff,
        'agreement_range': agreement_range,
        'upper_loa': upper_loa,
        'lower_loa': lower_loa
    }

# === Walk through all subfolders ===
for dirpath, _, filenames in os.walk(root_folder):
    for file in filenames:
        if file.endswith("_output_gpt.csv"):
            file_path = os.path.join(dirpath, file)
            try:
                print(f"\n🔄 Processing: {file_path}")
                df = pd.read_csv(file_path, delimiter=";", quotechar='"')
                
                # Extract info
                model_name = os.path.basename(dirpath)
                dimension = file.replace("_output_gpt.csv", "").replace("_graded", "")
                
                print(f"  Model: {model_name}, Dimension: {dimension}")
                print(f"  Original data: {len(df)} rows")
                
                # Convert scores to proper format (only AI scores needed)
                df["gpt_score_0_100"] = df["gpt_score_0_100"].astype(str).str.replace(",", ".").astype(float)
                df["gemini_score_0_100"] = df["gemini_score_0_100"].astype(str).str.replace(",", ".").astype(float)
                
                # Aggregate AI scores only
                df_aggregated = aggregate_ai_scores(df, dimension)
                
                # Extract aggregated scores for AI evaluators only
                gpt_scores = df_aggregated["gpt_score_0_100"].values
                gemini_scores = df_aggregated["gemini_score_0_100"].values
                
                # === Calculate comprehensive agreement metrics ===
                metrics = calculate_ai_agreement_metrics(gpt_scores, gemini_scores)
                
                results.append({
                    "Model": model_name,
                    "Dimension": dimension,
                    "Comparison": "GPT4_vs_Gemini",
                    "N_evaluations": len(gpt_scores),
                    "ICC3": round(metrics['icc_value'], 3),
                    "Pearson_corr": round(metrics['pearson_corr'], 3),
                    "Spearman_corr": round(metrics['spearman_corr'], 3),
                    "MAE": round(metrics['mae'], 2),
                    "Systematic_bias": round(metrics['systematic_bias'], 2),
                    "Agreement_range_95": round(metrics['agreement_range'], 2),
                    "Upper_LoA": round(metrics['upper_loa'], 2),
                    "Lower_LoA": round(metrics['lower_loa'], 2)
                })
                
                print(f"  → AI Agreement metrics calculated for {len(gpt_scores)} evaluations")
                print(f"    ICC3: {round(metrics['icc_value'], 3)}, Pearson: {round(metrics['pearson_corr'], 3)}, MAE: {round(metrics['mae'], 2)}")
                print(f"    Systematic bias: {round(metrics['systematic_bias'], 2)} (GPT-Gemini), Agreement range: ±{round(metrics['agreement_range'], 2)}")
                
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")

# === Save summary table ===
summary = pd.DataFrame(results)
summary.to_csv(output_path, index=False)
print(f"\n✅ AI Evaluators Agreement Summary saved to: {output_path}")

# === Print summary statistics ===
print("\n📊 Inter-Rater Agreement Between AI Evaluators (GPT-4 vs Gemini):")
print("Overall Statistics:")
core_metrics = ["ICC3", "Pearson_corr", "Spearman_corr", "MAE", "Systematic_bias", "Agreement_range_95"]
print(summary[core_metrics].describe().round(3))

print("\nMean agreement metrics across all dimensions and models:")
print(summary[core_metrics].mean().round(3))

print("\n📈 Agreement by Dimension:")
dimension_summary = summary.groupby("Dimension")[["ICC3", "Pearson_corr", "MAE", "Agreement_range_95"]].mean().round(3)
print(dimension_summary)

print("\n📈 Agreement by Model:")
model_summary = summary.groupby("Model")[["ICC3", "Pearson_corr", "MAE", "Agreement_range_95"]].mean().round(3)
print(model_summary)

print("\n📊 Number of evaluation units by dimension:")
eval_counts = summary.groupby(["Dimension", "Model"])["N_evaluations"].first().reset_index()
print(eval_counts.groupby("Dimension")["N_evaluations"].describe())

print("\n🎯 Key Insights:")
print("- MAE: Average absolute difference in points")
print("- Systematic_bias: Positive = GPT scores higher, Negative = Gemini scores higher")
print("- Agreement_range_95: 95% of differences within ±this range")