import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from pingouin import intraclass_corr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# === Config ===
root_folder = r"C:\Users\rache\Desktop\Master Thesis Research\LLM Scoring"
output_path = os.path.join(root_folder, "human_ai_agreement_summary.csv")
results = []

# === Helper: Binning into 0, 1, 2 classes ===
def bin_scores_ai(scores):
    bins = []
    for s in scores:
        if s <= 39:
            bins.append(0)
        elif s <= 80:
            bins.append(1)
        else:
            bins.append(2)
    return bins

def bin_scores_human(scores):
    bins = []
    for s in scores:
        if s == 0:
            bins.append(0)
        elif s == 0.5:
            bins.append(1)
        elif s == 1:
            bins.append(2)
        else:
            bins.append(1)
    return bins

def aggregate_scores(df, dimension):
    print(f"  → Aggregating scores for dimension: {dimension}")

    df = df.copy()
    df['prompt'] = df['prompt'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    score_columns = ['gpt_score_0_100', 'gemini_score_0_100', 'human_score']

    if dimension == 'linguistic':
        if 'set_id' in df.columns:
            df_with_set = df[df['set_id'].notna() & (df['set_id'] != '') & (df['set_id'] != 'nan')]
            df_without_set = df[df['set_id'].isna() | (df['set_id'] == '') | (df['set_id'] == 'nan')]
            print(f"    → Found {len(df_with_set)} rows with set_id, {len(df_without_set)} rows without set_id")
            aggregated_parts = []

            if len(df_with_set) > 0:
                print("    → Aggregating data WITH set_id: first prompt per set per run")
                df_first = df_with_set.sort_values(['set_id', 'run', 'prompt']).groupby(['set_id', 'run'], as_index=False).first()
                df_set_agg = df_first.groupby('set_id').agg({col: 'mean' for col in score_columns}).reset_index()
                aggregated_parts.append(df_set_agg)

            if len(df_without_set) > 0:
                print("    → Aggregating data WITHOUT set_id: per individual prompt across runs")
                df_prompt_agg = df_without_set.groupby('prompt', dropna=False).agg({col: 'mean' for col in score_columns}).reset_index()
                aggregated_parts.append(df_prompt_agg)

            df_agg = pd.concat(aggregated_parts, ignore_index=True) if aggregated_parts else pd.DataFrame(columns=score_columns)

        else:
            print("    → No set_id column — averaging per individual prompt across 3 runs")
            df_agg = df.groupby('prompt', dropna=False).agg({col: 'mean' for col in score_columns}).reset_index()

    elif dimension in ['ethics', 'context']:
        if 'subdimension' not in df.columns or 'set_id' not in df.columns:
            raise ValueError(f"{dimension} requires 'subdimension' and 'set_id' columns for aggregation")

        print("    → Aggregating first prompt per (subdimension, set_id) across 3 runs")
        df_first = df.sort_values(['subdimension', 'set_id', 'run', 'prompt']).groupby(
            ['subdimension', 'set_id', 'run'], as_index=False).first()

        df_agg = df_first.groupby(['subdimension', 'set_id']).agg({col: 'mean' for col in score_columns}).reset_index()

    else:
        print("    → Default: averaging per prompt across 3 runs")
        df_agg = df.groupby('prompt', dropna=False).agg({col: 'mean' for col in score_columns}).reset_index()

    # Calculate average AI score
    df_agg["ai_average_score"] = (df_agg["gpt_score_0_100"] + df_agg["gemini_score_0_100"]) / 2

    print(f"    → Aggregated to {len(df_agg)} rows from {len(df)} raw entries")
    return df_agg

def calculate_human_ai_agreement_metrics(human_scores, ai_scores, comparison_name):
    """
    Calculate comprehensive agreement metrics between human and AI scores.
    
    Args:
        human_scores: Array of human scores
        ai_scores: Array of AI scores  
        comparison_name: String identifier for the comparison
        
    Returns:
        Dictionary of agreement metrics or None if no valid pairs
    """
    human_scores = np.array(human_scores)
    ai_scores = np.array(ai_scores)
    
    # Remove invalid scores
    valid_mask = ~(np.isnan(human_scores) | np.isnan(ai_scores))
    human_scores = human_scores[valid_mask]
    ai_scores = ai_scores[valid_mask]

    if len(human_scores) == 0:
        print(f"    ⚠️ No valid score pairs for {comparison_name}")
        return None

    print(f"    → Calculating metrics for {comparison_name} with {len(human_scores)} valid pairs")
    
    # === ICC Calculation (FIXED) ===
    # Create proper long-format data where both raters evaluate the same targets
    icc_data = []
    for i, (h_score, a_score) in enumerate(zip(human_scores, ai_scores)):
        icc_data.append({'targets': i, 'rater': 'Human', 'score': h_score})
        icc_data.append({'targets': i, 'rater': 'AI', 'score': a_score})
    
    icc_df = pd.DataFrame(icc_data)
    
    try:
        icc_result = intraclass_corr(data=icc_df, targets='targets', raters='rater', ratings='score')
        # Use ICC(3,1) for consistency - single rater, fixed raters
        icc_row = icc_result[icc_result["Type"] == "ICC3"]
        if len(icc_row) > 0:
            icc_value = icc_row["ICC"].values[0]
        else:
            print(f"    ⚠️ ICC3 not found, using first available ICC type")
            icc_value = icc_result["ICC"].values[0]
    except Exception as e:
        print(f"    ⚠️ ICC calculation failed for {comparison_name}: {e}")
        icc_value = np.nan
    
    # === Correlation Metrics ===
    try:
        pearson_corr, pearson_p = pearsonr(human_scores, ai_scores)
        spearman_corr, spearman_p = spearmanr(human_scores, ai_scores)
    except Exception as e:
        print(f"    ⚠️ Correlation calculation failed: {e}")
        pearson_corr = spearman_corr = np.nan
        pearson_p = spearman_p = np.nan
    
    # === Categorical Agreement Metrics ===
    try:
        # Cohen's Kappa - perfect for ordinal categories
        kappa = cohen_kappa_score(human_scores, ai_scores)
        
        # Weighted Cohen's Kappa - accounts for degree of disagreement
        kappa_weighted = cohen_kappa_score(human_scores, ai_scores, weights='linear')
        
        # Simple accuracy
        accuracy = np.mean(human_scores == ai_scores)
        
    except Exception as e:
        print(f"    ⚠️ Categorical metrics calculation failed: {e}")
        kappa = kappa_weighted = accuracy = np.nan
    mae = np.mean(np.abs(human_scores - ai_scores))
    rmse = np.sqrt(np.mean((human_scores - ai_scores) ** 2))
    
    # Bland-Altman statistics  
    diff = human_scores - ai_scores
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    agreement_range = 1.96 * std_diff
    
    # Descriptive statistics
    human_mean = np.mean(human_scores)
    ai_mean = np.mean(ai_scores)
    human_std = np.std(human_scores, ddof=1)
    ai_std = np.std(ai_scores, ddof=1)

    return {
        'icc_value': icc_value,
        'spearman_corr': spearman_corr, 
        'spearman_p': spearman_p,
        'cohens_kappa_weighted': kappa_weighted,
        'accuracy': accuracy,
        'systematic_bias': mean_diff,
        'agreement_range': agreement_range,
        'upper_loa': upper_loa,
        'lower_loa': lower_loa,
        'n_valid': len(human_scores),
        'human_mean': human_mean,
        'ai_mean': ai_mean,
        'human_std': human_std,
        'ai_std': ai_std
    }

# === Walk through all subfolders ===
for dirpath, _, filenames in os.walk(root_folder):
    for file in filenames:
        if file.endswith("_output_gpt.csv"):
            file_path = os.path.join(dirpath, file)
            try:
                print(f"\n🔄 Processing: {file_path}")
                df = pd.read_csv(file_path, delimiter=";", quotechar='"')
                model_name = os.path.basename(dirpath)
                dimension = file.replace("_output_gpt.csv", "").replace("_graded", "")
                print(f"  Model: {model_name}, Dimension: {dimension}")
                print(f"  Original data: {len(df)} rows")

                # Convert scores with better error handling
                def safe_convert_scores(series):
                    return pd.to_numeric(series.astype(str).str.replace(",", "."), errors='coerce')
                
                df["human_score"] = safe_convert_scores(df["human_score"])
                df["gpt_score_0_100"] = safe_convert_scores(df["gpt_score_0_100"])
                df["gemini_score_0_100"] = safe_convert_scores(df["gemini_score_0_100"])

                # Check for data issues
                print(f"  Human score range: {df['human_score'].min():.2f} - {df['human_score'].max():.2f}")
                print(f"  GPT score range: {df['gpt_score_0_100'].min():.2f} - {df['gpt_score_0_100'].max():.2f}")
                print(f"  Gemini score range: {df['gemini_score_0_100'].min():.2f} - {df['gemini_score_0_100'].max():.2f}")

                # Aggregate scores
                df_aggregated = aggregate_scores(df, dimension)

                # Extract and convert scores to same scale (0, 1, 2)
                human_scores_raw = df_aggregated["human_score"].values
                gpt_scores_raw = df_aggregated["gpt_score_0_100"].values
                gemini_scores_raw = df_aggregated["gemini_score_0_100"].values
                ai_average_scores_raw = df_aggregated["ai_average_score"].values
                
                # Convert to binned scores (0, 1, 2)
                human_scores = bin_scores_human(human_scores_raw)
                gpt_scores = bin_scores_ai(gpt_scores_raw)
                gemini_scores = bin_scores_ai(gemini_scores_raw)
                ai_average_scores = bin_scores_ai(ai_average_scores_raw)
                
                print(f"  After binning - Human: {np.unique(human_scores)}, GPT: {np.unique(gpt_scores)}")

                comparisons = [
                    ("Human_vs_GPT", human_scores, gpt_scores),
                    ("Human_vs_Gemini", human_scores, gemini_scores),
                    ("Human_vs_AI_Average", human_scores, ai_average_scores)
                ]

                for comparison_name, ref_scores, comp_scores in comparisons:
                    metrics = calculate_human_ai_agreement_metrics(ref_scores, comp_scores, comparison_name)
                    if metrics is not None:
                        results.append({
                            "Model": model_name,
                            "Dimension": dimension,
                            "Comparison": comparison_name,
                            "N_evaluations": metrics['n_valid'],
                            "ICC3": round(metrics['icc_value'], 3) if not np.isnan(metrics['icc_value']) else np.nan,
                            "Cohens_Kappa_Weighted": round(metrics['cohens_kappa_weighted'], 3) if not np.isnan(metrics['cohens_kappa_weighted']) else np.nan,
                            "Accuracy": round(metrics['accuracy'], 3) if not np.isnan(metrics['accuracy']) else np.nan,
                            "Spearman_corr": round(metrics['spearman_corr'], 3) if not np.isnan(metrics['spearman_corr']) else np.nan,
                            "Spearman_p": round(metrics['spearman_p'], 4) if not np.isnan(metrics['spearman_p']) else np.nan,
                            "Systematic_bias": round(metrics['systematic_bias'], 2),
                            "Agreement_range_95": round(metrics['agreement_range'], 2),
                            "Upper_LoA": round(metrics['upper_loa'], 2),
                            "Lower_LoA": round(metrics['lower_loa'], 2),
                            "Human_mean": round(metrics['human_mean'], 2),
                            "AI_mean": round(metrics['ai_mean'], 2),
                            "Human_std": round(metrics['human_std'], 2),
                            "AI_std": round(metrics['ai_std'], 2)
                        })

                        print(f"  → {comparison_name} metrics calculated for {metrics['n_valid']} evaluations")
                        print(f"    ICC3: {round(metrics['icc_value'], 3) if not np.isnan(metrics['icc_value']) else 'N/A'}")
                        print(f"    Weighted κ: {round(metrics['cohens_kappa_weighted'], 3) if not np.isnan(metrics['cohens_kappa_weighted']) else 'N/A'}")
                        print(f"    Spearman: {round(metrics['spearman_corr'], 3) if not np.isnan(metrics['spearman_corr']) else 'N/A'}")
                        print(f"    Accuracy: {round(metrics['accuracy'], 3) if not np.isnan(metrics['accuracy']) else 'N/A'}")
                        print(f"    Systematic bias: {round(metrics['systematic_bias'], 2)} (Human-AI)")

            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()

# === Save summary table ===
if results:
    summary = pd.DataFrame(results)
    summary.to_csv(output_path, index=False)
    print(f"\n✅ Human vs AI Evaluators Agreement Summary saved to: {output_path}")

    # === Print summary statistics ===
    print("\n📊 Inter-Rater Agreement Between Human and AI Evaluators:")
    print("Overall Statistics:")
    core_metrics = ["ICC3", "Cohens_Kappa_Weighted", "Spearman_corr", "Accuracy", "Systematic_bias"]
    valid_summary = summary[core_metrics].dropna()
    if len(valid_summary) > 0:
        print(valid_summary.describe().round(3))
        
        print("\nMean agreement metrics across all dimensions and models:")
        print(valid_summary.mean().round(3))

        print("\n📈 Agreement by Comparison Type:")
        comp_group = summary.groupby("Comparison")[["ICC3", "Cohens_Kappa_Weighted", "Spearman_corr", "Accuracy"]].mean()
        print(comp_group.round(3))

        print("\n📈 Agreement by Dimension:")
        dim_group = summary.groupby("Dimension")[["ICC3", "Cohens_Kappa_Weighted", "Spearman_corr", "Accuracy"]].mean()
        print(dim_group.round(3))

        print("\n📈 Agreement by Model:")
        model_group = summary.groupby("Model")[["ICC3", "Cohens_Kappa_Weighted", "Spearman_corr", "Accuracy"]].mean()
        print(model_group.round(3))

        print("\n📊 Number of evaluation units by dimension and comparison:")
        eval_counts = summary.groupby(["Dimension", "Comparison"])["N_evaluations"].first().reset_index()
        eval_pivot = eval_counts.pivot(index="Dimension", columns="Comparison", values="N_evaluations")
        print(eval_pivot)
        
        # Flag potential issues
        negative_icc = summary[summary['ICC3'] < 0]
        if len(negative_icc) > 0:
            print(f"\n⚠️ Warning: {len(negative_icc)} cases with negative ICC found")
            print("Negative ICC cases:")
            print(negative_icc[['Model', 'Dimension', 'Comparison', 'N_evaluations', 'ICC3', 'Systematic_bias']].to_string())
            
        # Additional diagnostics
        low_n = summary[summary['N_evaluations'] < 10]
        if len(low_n) > 0:
            print(f"\n📊 {len(low_n)} cases with <10 evaluations - results may be unstable")
            
        print(f"\n📈 Distribution of agreement levels:")
        print(f"Excellent (ICC >0.75): {len(summary[summary['ICC3'] > 0.75])}")
        print(f"Good (ICC 0.60-0.75): {len(summary[(summary['ICC3'] >= 0.60) & (summary['ICC3'] <= 0.75)])}")
        print(f"Fair (ICC 0.40-0.59): {len(summary[(summary['ICC3'] >= 0.40) & (summary['ICC3'] < 0.60)])}")
        print(f"Poor (ICC <0.40): {len(summary[summary['ICC3'] < 0.40])}")
    else:
        print("⚠️ No valid metrics calculated - check data and score conversions")
else:
    print("⚠️ No results generated - check file paths and data format")