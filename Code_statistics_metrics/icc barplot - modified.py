import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

plt.style.use('default')
plt.rcParams['axes.grid'] = False  # Disable grids globally
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Define RGB color palette
rgb_colors = [
    (0.2, 0.3, 0.8),      # Blue
    (0.8, 0.2, 0.2),      # Red
    (0.2, 0.7, 0.3),      # Green
    (0.9, 0.6, 0.1),      # Orange
    (0.6, 0.2, 0.8),      # Purple
    (0.1, 0.8, 0.8),      # Cyan
    (0.8, 0.8, 0.2),      # Yellow
    (0.8, 0.4, 0.6),      # Pink
]

# Load your AI agreement data
df = pd.read_csv("C:/Users/rache/Desktop/Master Thesis Research/LLM Scoring/irr_summary_table_modified.csv")

# Clean up names for prettier display
df["Dimension"] = df["Dimension"].str.capitalize()
df["Model"] = df["Model"].replace({
    "Gemma": "Gemma 2 9B",
    "Gemini": "Gemini 2.0",
    "LLama": "LLaMA 3 70B", 
    "Deepseek": "DeepSeek R1",
    "Mistral": "Mistral Large"
})

output_dir = "C:/Users/rache/Desktop/Master Thesis Research/LLM Scoring"

# === AI EVALUATORS AGREEMENT PLOTS ===

# 1. **ICC3 AGREEMENT BY DIMENSION AND MODEL** - How well do GPT-4 and Gemini agree?
plt.figure(figsize=(16, 8))
pivot_icc = df.pivot(index='Dimension', columns='Model', values='ICC3')

x = np.arange(len(pivot_icc.index))
width = 0.15
models = pivot_icc.columns

for i, model in enumerate(models):
    bars = plt.bar(x + i*width - (len(models)-1)*width/2, pivot_icc[model], width, 
                   label=model, alpha=0.8, color=rgb_colors[i % len(rgb_colors)])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

# Add interpretation lines
plt.axhline(y=0.75, color=(0.0, 0.6, 0.0), linestyle='--', alpha=0.7, linewidth=2, label='Excellent (≥0.75)')
plt.axhline(y=0.5, color=(1.0, 0.5, 0.0), linestyle='--', alpha=0.7, linewidth=2, label='Fair (≥0.5)')
plt.axhline(y=0.25, color=(0.8, 0.0, 0.0), linestyle='--', alpha=0.7, linewidth=2, label='Poor (≥0.25)')

plt.title('GPT-4o vs Gemini 2.5 Agreement by Model and Dimension\n(ICC3 - Intraclass Correlation)', 
          fontsize=14, fontweight='bold', pad=20)
plt.ylabel('ICC(3,1)', fontsize=12)
plt.xlabel('Evaluation Dimensions', fontsize=12)
plt.xticks(x, pivot_icc.index, rotation=45, ha='right')
plt.ylim(0, 1)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
           ncol=len(models), frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "AI_01_icc_by_dimension_model.png"), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 2. **CORRELATION COMPARISON BY MODEL AND DIMENSION** - Pearson vs Spearman
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# Pearson correlation by model and dimension
pivot_pearson = df.pivot(index='Dimension', columns='Model', values='Pearson_corr')
x = np.arange(len(pivot_pearson.index))
width = 0.15

for i, model in enumerate(pivot_pearson.columns):
    bars = ax1.bar(x + i*width - (len(pivot_pearson.columns)-1)*width/2, pivot_pearson[model], width,
                   label=model, alpha=0.8, color=rgb_colors[i % len(rgb_colors)])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

ax1.axhline(y=0.7, color=(0.0, 0.6, 0.0), linestyle='--', alpha=0.7, linewidth=2, label='Strong (≥0.7)')
ax1.axhline(y=0.5, color=(1.0, 0.5, 0.0), linestyle='--', alpha=0.7, linewidth=2, label='Moderate (≥0.5)')
ax1.set_title('Pearson Correlation (Linear Relationship)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Correlation Coefficient', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(pivot_pearson.index, rotation=45, ha='right')
ax1.set_ylim(0, 1)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

# Spearman correlation by model and dimension  
pivot_spearman = df.pivot(index='Dimension', columns='Model', values='Spearman_corr')

for i, model in enumerate(pivot_spearman.columns):
    bars = ax2.bar(x + i*width - (len(pivot_spearman.columns)-1)*width/2, pivot_spearman[model], width,
                   label=model, alpha=0.8, color=rgb_colors[i % len(rgb_colors)])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

ax2.axhline(y=0.7, color=(0.0, 0.6, 0.0), linestyle='--', alpha=0.7, linewidth=2)
ax2.axhline(y=0.5, color=(1.0, 0.5, 0.0), linestyle='--', alpha=0.7, linewidth=2)
ax2.set_title('Spearman Correlation (Rank Agreement)', fontweight='bold', fontsize=12)
ax2.set_ylabel('Correlation Coefficient', fontsize=11)
ax2.set_xlabel('Evaluation Dimensions', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(pivot_spearman.index, rotation=45, ha='right')
ax2.set_ylim(0, 1)

plt.suptitle('GPT-4o vs Gemini 2.5: Linear vs Rank-Order Agreement by Model and Dimension', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "AI_02_correlation_comparison.png"), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 3. **MEASUREMENT ERROR ANALYSIS BY MODEL AND DIMENSION** - MAE and Agreement Range
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# Mean Absolute Error by model and dimension
pivot_mae = df.pivot(index='Dimension', columns='Model', values='MAE')
x = np.arange(len(pivot_mae.index))
width = 0.15

for i, model in enumerate(pivot_mae.columns):
    bars = ax1.bar(x + i*width - (len(pivot_mae.columns)-1)*width/2, pivot_mae[model], width,
                   label=model, alpha=0.8, color=rgb_colors[i % len(rgb_colors)])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                     f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

ax1.set_title('Mean Absolute Error (Average Point Difference)', fontweight='bold', fontsize=12)
ax1.set_ylabel('MAE (0-100 scale)', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(pivot_mae.index, rotation=45, ha='right')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

# Agreement Range by model and dimension
pivot_range = df.pivot(index='Dimension', columns='Model', values='Agreement_range_95')

for i, model in enumerate(pivot_range.columns):
    bars = ax2.bar(x + i*width - (len(pivot_range.columns)-1)*width/2, pivot_range[model], width,
                   label=model, alpha=0.8, color=rgb_colors[i % len(rgb_colors)])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'±{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

ax2.set_title('95% Agreement Range (95% of differences within ±range)', fontweight='bold', fontsize=12)
ax2.set_ylabel('Agreement Range (points)', fontsize=11)
ax2.set_xlabel('Evaluation Dimensions', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(pivot_range.index, rotation=45, ha='right')

plt.suptitle('GPT-4o vs Gemini 2.5: Measurement Error Analysis by Model and Dimension', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "AI_03_measurement_error.png"), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 4. **SYSTEMATIC BIAS ANALYSIS BY MODEL AND DIMENSION** - Which AI scores higher?
plt.figure(figsize=(16, 8))
pivot_bias = df.pivot(index='Dimension', columns='Model', values='Systematic_bias')

x = np.arange(len(pivot_bias.index))
width = 0.15

for i, model in enumerate(pivot_bias.columns):
    values = pivot_bias[model]
    
    bars = plt.bar(x + i*width - (len(pivot_bias.columns)-1)*width/2, values, width,
                   label=model, alpha=0.8, color=rgb_colors[i % len(rgb_colors)])
    
    # Add value labels
    for j, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            y_pos = height + (0.5 if height >= 0 else -0.5)
            plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                     f'{height:.1f}', ha='center', va='bottom' if height >= 0 else 'top', 
                     fontweight='bold', fontsize=8)

plt.axhline(y=0, color=(0.0, 0.0, 0.0), linestyle='-', alpha=0.8, linewidth=2)
plt.title('Systematic Bias Between GPT-4o and Gemini 2.5 by Model and Dimension\n(Positive = GPT scores higher, Negative = Gemini scores higher)', 
          fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Systematic Bias (GPT - Gemini)', fontsize=12)
plt.xlabel('Evaluation Dimensions', fontsize=12)
plt.xticks(x, pivot_bias.index, rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "AI_04_systematic_bias.png"), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 5. **MODEL COMPARISON HEATMAP** - All metrics across models and dimensions
plt.figure(figsize=(14, 10))
# Create pivot for heatmap
model_dim_data = df.groupby(['Model', 'Dimension'])[['ICC3', 'Pearson_corr', 'Spearman_corr', 'MAE']].mean()

# Normalize MAE (flip scale so higher is better like other metrics)
model_dim_data['MAE_normalized'] = 1 - (model_dim_data['MAE'] / model_dim_data['MAE'].max())

# Create multi-level heatmap data
heatmap_metrics = model_dim_data[['ICC3', 'Pearson_corr', 'Spearman_corr', 'MAE_normalized']].reset_index()
heatmap_pivot = heatmap_metrics.melt(id_vars=['Model', 'Dimension'], 
                                     var_name='Metric', value_name='Score')
final_pivot = heatmap_pivot.pivot_table(index=['Model', 'Dimension'], 
                                         columns='Metric', values='Score')

# Create custom RGB colormap for heatmap
from matplotlib.colors import LinearSegmentedColormap
heatmap_colors = [(0.8, 0.2, 0.2), (1.0, 1.0, 0.8), (0.2, 0.3, 0.8)]  # Red -> Light Yellow -> Blue
custom_cmap = LinearSegmentedColormap.from_list("custom_rgb", heatmap_colors)

sns.heatmap(final_pivot, annot=True, fmt='.3f', cmap=custom_cmap, center=0.5,
            cbar_kws={'label': 'Agreement Score (0-1 scale)'}, square=False)
plt.title('AI Evaluators Agreement: Comprehensive Metric Comparison\n(GPT-4 vs Gemini across Models and Dimensions)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Agreement Metric', fontsize=12)
plt.ylabel('Model → Dimension', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "AI_05_comprehensive_heatmap.png"), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 6. **BLAND-ALTMAN STYLE PLOT BY MODEL AND DIMENSION** - Agreement limits visualization
plt.figure(figsize=(18, 10))

# Create subplot for each dimension
dimensions = df['Dimension'].unique()
n_dims = len(dimensions)
cols = 3 if n_dims > 4 else 2
rows = (n_dims + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
if rows == 1:
    axes = [axes] if cols == 1 else axes
else:
    axes = axes.flatten()

for i, dim in enumerate(dimensions):
    dim_data = df[df['Dimension'] == dim]
    
    # Plot each model's bias
    models = dim_data['Model'].unique()
    x_pos = range(len(models))
    
    axes[i].scatter(x_pos, dim_data['Systematic_bias'], alpha=0.7, s=80, 
                    c=[rgb_colors[j % len(rgb_colors)] for j in range(len(models))])
    
    # Calculate dimension-level statistics
    mean_bias = dim_data['Systematic_bias'].mean()
    mean_upper = dim_data['Upper_LoA'].mean()
    mean_lower = dim_data['Lower_LoA'].mean()
    
    # Add agreement limits
    axes[i].axhline(y=mean_bias, color=(0.2, 0.3, 0.8), linestyle='-', alpha=0.7, linewidth=2, 
                    label=f'Mean bias: {mean_bias:.1f}')
    axes[i].axhline(y=mean_upper, color=(0.8, 0.2, 0.2), linestyle='--', alpha=0.7, linewidth=2, 
                    label=f'Upper LoA: {mean_upper:.1f}')
    axes[i].axhline(y=mean_lower, color=(0.8, 0.2, 0.2), linestyle='--', alpha=0.7, linewidth=2, 
                    label=f'Lower LoA: {mean_lower:.1f}')
    axes[i].axhline(y=0, color=(0.0, 0.0, 0.0), linestyle='-', alpha=0.3, linewidth=1)
    
    # Add model labels on points
    for j, (model, bias) in enumerate(zip(models, dim_data['Systematic_bias'])):
        axes[i].text(j, bias + 0.5, model[:8], ha='center', va='bottom', fontsize=8, rotation=45)
    
    axes[i].set_title(f'{dim}\nLimits of Agreement', fontweight='bold')
    axes[i].set_ylabel('Score Difference (GPT - Gemini)')
    axes[i].set_xticks(x_pos)
    axes[i].set_xticklabels([m[:8] for m in models], rotation=45, ha='right')
    axes[i].legend(frameon=False, fontsize=8)

# Hide empty subplots
for i in range(len(dimensions), len(axes)):
    axes[i].axis('off')

plt.suptitle('Bland-Altman Analysis: GPT-4 vs Gemini Agreement Limits by Model and Dimension', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "AI_06_bland_altman_analysis.png"), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ AI Evaluators Agreement Plots Saved with RGB Colors:")
print("1. ICC3 Agreement by Dimension")
print("2. Correlation Comparison (Pearson vs Spearman)")
print("3. Measurement Error Analysis (MAE & Agreement Range)")
print("4. Systematic Bias Analysis") 
print("5. Comprehensive Metric Heatmap")
print("6. Bland-Altman Agreement Limits")

# Print key insights
print("\n📊 Key Insights for AI Evaluators Agreement:")
print(f"Overall ICC3: {df['ICC3'].mean():.3f} ± {df['ICC3'].std():.3f}")
print(f"Overall Pearson: {df['Pearson_corr'].mean():.3f} ± {df['Pearson_corr'].std():.3f}")
print(f"Overall MAE: {df['MAE'].mean():.1f} ± {df['MAE'].std():.1f} points")

print("\nBest agreeing dimension (ICC3):")
best_dim = df.groupby('Dimension')['ICC3'].mean().idxmax()
best_score = df.groupby('Dimension')['ICC3'].mean().max()
print(f"  {best_dim}: ICC3 = {best_score:.3f}")

print("\nWorst agreeing dimension (ICC3):")
worst_dim = df.groupby('Dimension')['ICC3'].mean().idxmin()
worst_score = df.groupby('Dimension')['ICC3'].mean().min()
print(f"  {worst_dim}: ICC3 = {worst_score:.3f}")

print("\nSystematic bias overview:")
bias_summary = df.groupby('Dimension')['Systematic_bias'].mean()
gpt_higher = bias_summary[bias_summary > 0]
gemini_higher = bias_summary[bias_summary < 0]
print(f"  GPT-4 scores higher in: {list(gpt_higher.index)}")
print(f"  Gemini scores higher in: {list(gemini_higher.index)}")