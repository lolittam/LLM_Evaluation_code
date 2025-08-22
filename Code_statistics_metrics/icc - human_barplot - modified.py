import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Load your data
df = pd.read_csv("C:/Users/rache/Desktop/Master Thesis Research/LLM Scoring/human_ai_agreement_summary.csv")

# Clean up names for prettier display
df["Dimension"] = df["Dimension"].str.capitalize()
df["Model"] = df["Model"].replace({
    "Gemma": "Gemma 2 9B",
    "Gemini": "Gemini 2.0", 
    "LLama": "LLaMA 3 70B",
    "Deepseek": "DeepSeek R1",
    "Mistral": "Mistral Large"
})

# Clean comparison names
df["Comparison"] = df["Comparison"].replace({
    "Human_vs_Gemini": "Human vs Gemini",
    "Human_vs_GPT": "Human vs GPT-4", 
    "Human_vs_AI_Average": "Human vs AI Average"
})

output_dir = "C:/Users/rache/Desktop/Master Thesis Research/LLM Scoring"

# 1. **WEIGHTED KAPPA BY COMPARISON TYPE**
plt.figure(figsize=(14, 8))
comparison_kappa = df.groupby(['Comparison', 'Dimension'])['Cohens_Kappa_Weighted'].mean().reset_index()
pivot_kappa = comparison_kappa.pivot(index='Dimension', columns='Comparison', values='Cohens_Kappa_Weighted')

x = np.arange(len(pivot_kappa.index))
width = 0.25
colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange

bars1 = plt.bar(x - width, pivot_kappa['Human vs AI Average'], width, 
                label='Human vs AI Average', color=colors[0], alpha=0.8)
bars2 = plt.bar(x, pivot_kappa['Human vs GPT-4'], width, 
                label='Human vs GPT-4', color=colors[1], alpha=0.8)  
bars3 = plt.bar(x + width, pivot_kappa['Human vs Gemini'], width, 
                label='Human vs Gemini', color=colors[2], alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add interpretation lines
plt.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Good Agreement (≥0.6)')
plt.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Fair Agreement (≥0.4)')
plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Poor Agreement (≥0.2)')

plt.xlabel('Evaluation Dimension', fontsize=12)
plt.ylabel('Weighted Cohen\'s Kappa', fontsize=12)
plt.title('Human-AI Agreement by Comparison Type and Dimension\n(Weighted Cohen\'s Kappa)', 
          fontsize=14, fontweight='bold')
plt.xticks(x, pivot_kappa.index, rotation=45)
plt.legend(loc='upper right')
plt.ylim(-0.1, 0.7)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_kappa_by_comparison_dimension.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# 2. **ICC(3,1) BY COMPARISON TYPE** - Same structure for ICC
plt.figure(figsize=(14, 8))
comparison_icc = df.groupby(['Comparison', 'Dimension'])['ICC3'].mean().reset_index()
pivot_icc = comparison_icc.pivot(index='Dimension', columns='Comparison', values='ICC3')

bars1 = plt.bar(x - width, pivot_icc['Human vs AI Average'], width, 
                label='Human vs AI Average', color=colors[0], alpha=0.8)
bars2 = plt.bar(x, pivot_icc['Human vs GPT-4'], width, 
                label='Human vs GPT-4', color=colors[1], alpha=0.8)
bars3 = plt.bar(x + width, pivot_icc['Human vs Gemini'], width, 
                label='Human vs Gemini', color=colors[2], alpha=0.8)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add interpretation lines for ICC
plt.axhline(y=0.75, color='green', linestyle='--', alpha=0.5, label='Excellent (≥0.75)')
plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (≥0.5)')

plt.xlabel('Evaluation Dimension', fontsize=12)
plt.ylabel('ICC(3,1)', fontsize=12)
plt.title('Human-AI Agreement by Comparison Type and Dimension\n(Intraclass Correlation Coefficient)', 
          fontsize=14, fontweight='bold')
plt.xticks(x, pivot_icc.index, rotation=45)
plt.legend(loc='upper right')
plt.ylim(-0.1, 0.8)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_icc_by_comparison_dimension.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# 3. **DUAL METRIC HEATMAP** - Show both Kappa and ICC side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Kappa heatmap
sns.heatmap(pivot_kappa.T, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.3,
            ax=axes[0], cbar_kws={'label': 'Weighted Cohen\'s Kappa'})
axes[0].set_title('Weighted Cohen\'s Kappa\nby Comparison Type', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Evaluation Dimension', fontsize=10)
axes[0].set_ylabel('Comparison Type', fontsize=10)

# ICC heatmap  
sns.heatmap(pivot_icc.T, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.3,
            ax=axes[1], cbar_kws={'label': 'ICC(3,1)'})
axes[1].set_title('ICC(3,1)\nby Comparison Type', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Evaluation Dimension', fontsize=10)
axes[1].set_ylabel('Comparison Type', fontsize=10)

plt.suptitle('Human-AI Agreement: Weighted Kappa vs ICC(3,1)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "03_dual_metric_heatmap.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# 4. **COMPARISON RANKING PLOT** - Which comparison type performs best?
plt.figure(figsize=(12, 8))
comparison_stats = df.groupby('Comparison')[['Cohens_Kappa_Weighted', 'ICC3', 'Spearman_corr', 'Accuracy']].mean().sort_values('Cohens_Kappa_Weighted')

x = np.arange(len(comparison_stats))
width = 0.2

bars1 = plt.bar(x - 1.5*width, comparison_stats['Cohens_Kappa_Weighted'], width, 
                label='Weighted Kappa', alpha=0.8, color='#2E86AB')
bars2 = plt.bar(x - 0.5*width, comparison_stats['ICC3'], width, 
                label='ICC(3,1)', alpha=0.8, color='#A23B72')
bars3 = plt.bar(x + 0.5*width, comparison_stats['Spearman_corr'], width, 
                label='Spearman r', alpha=0.8, color='#F18F01')
bars4 = plt.bar(x + 1.5*width, comparison_stats['Accuracy'], width, 
                label='Accuracy', alpha=0.8, color='#E71D36')

# Add value labels
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.xlabel('Comparison Type', fontsize=12)
plt.ylabel('Agreement Score', fontsize=12)
plt.title('Overall Performance by Comparison Type\n(Average Across All Dimensions)', 
          fontsize=14, fontweight='bold')
plt.xticks(x, comparison_stats.index, rotation=15)
plt.legend()
plt.ylim(0, 1)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "04_comparison_ranking.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# 5. **DIMENSION RANKING WITHIN EACH COMPARISON**
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
comparisons = ['Human vs AI Average', 'Human vs GPT-4', 'Human vs Gemini']

for i, comp in enumerate(comparisons):
    comp_data = df[df['Comparison'] == comp].groupby('Dimension')['Cohens_Kappa_Weighted'].mean().sort_values()
    
    bars = axes[i].barh(comp_data.index, comp_data.values, alpha=0.8, color=colors[i])
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        axes[i].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    axes[i].set_title(f'{comp}\nDimension Ranking', fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Weighted Cohen\'s Kappa', fontsize=10)
    axes[i].axvline(x=0.4, color='red', linestyle='--', alpha=0.5)
    axes[i].axvline(x=0.6, color='green', linestyle='--', alpha=0.5)
    axes[i].set_xlim(-0.1, 0.7)

plt.suptitle('Dimension Performance Within Each Comparison Type', fontsize=14, fontweight='bold')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "05_dimension_ranking_by_comparison.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ NEW Academic plots saved:")
print("1. Weighted Kappa by Comparison Type & Dimension (BAR)")
print("2. ICC(3,1) by Comparison Type & Dimension (BAR)")  
print("3. Dual metric heatmap (Kappa & ICC side by side)")
print("4. Comparison type ranking (which is best overall?)")
print("5. Dimension ranking within each comparison type")

# Print detailed statistics
print("\n📊 Detailed Statistics by Comparison Type:")
for comp in df['Comparison'].unique():
    comp_data = df[df['Comparison'] == comp]
    print(f"\n{comp}:")
    print(f"  Weighted Kappa: {comp_data['Cohens_Kappa_Weighted'].mean():.3f} ± {comp_data['Cohens_Kappa_Weighted'].std():.3f}")
    print(f"  ICC(3,1): {comp_data['ICC3'].mean():.3f} ± {comp_data['ICC3'].std():.3f}")
    print(f"  Range (Kappa): {comp_data['Cohens_Kappa_Weighted'].min():.3f} to {comp_data['Cohens_Kappa_Weighted'].max():.3f}")

print("\n🏆 Best performing dimensions by comparison:")
for comp in df['Comparison'].unique():
    best_dim = df[df['Comparison'] == comp].groupby('Dimension')['Cohens_Kappa_Weighted'].mean().idxmax()
    best_score = df[df['Comparison'] == comp].groupby('Dimension')['Cohens_Kappa_Weighted'].mean().max()
    print(f"{comp}: {best_dim} (κ_w = {best_score:.3f})")

print("\n⚠️ Poorest performing dimensions by comparison:")
for comp in df['Comparison'].unique():
    worst_dim = df[df['Comparison'] == comp].groupby('Dimension')['Cohens_Kappa_Weighted'].mean().idxmin()
    worst_score = df[df['Comparison'] == comp].groupby('Dimension')['Cohens_Kappa_Weighted'].mean().min()
    print(f"{comp}: {worst_dim} (κ_w = {worst_score:.3f})")