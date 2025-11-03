import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Stroke Risk Analysis by Age - Multiple Visualization Approaches', fontsize=16, fontweight='bold')

# 1. Stroke Rate by Age Groups
age_bins = pd.cut(df['age'], bins=np.arange(0, 101, 10), right=False)
stroke_rate = df.groupby(age_bins)['stroke'].agg(['mean', 'count']).reset_index()
stroke_rate['age_midpoint'] = stroke_rate['age'].apply(lambda x: x.mid)

axes[0,0].bar(stroke_rate['age_midpoint'], stroke_rate['mean'] * 100, 
              width=8, alpha=0.7, color='coral', edgecolor='darkred')
axes[0,0].set_title('Stroke Rate by Age Group', fontweight='bold')
axes[0,0].set_xlabel('Age')
axes[0,0].set_ylabel('Stroke Rate (%)')
axes[0,0].grid(True, alpha=0.3)

# 2. Separate histograms (normalized)
stroke_ages = df[df['stroke'] == 1]['age']
no_stroke_ages = df[df['stroke'] == 0]['age']

axes[0,1].hist(no_stroke_ages, bins=20, alpha=0.6, label='No Stroke', 
               density=True, color='lightblue', edgecolor='blue')
axes[0,1].hist(stroke_ages, bins=20, alpha=0.8, label='Stroke', 
               density=True, color='red', edgecolor='darkred')
axes[0,1].set_title('Age Distribution (Normalized)', fontweight='bold')
axes[0,1].set_xlabel('Age')
axes[0,1].set_ylabel('Density')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. Box plot comparison
stroke_data = [no_stroke_ages, stroke_ages]
box_plot = axes[0,2].boxplot(stroke_data, labels=['No Stroke', 'Stroke'], 
                             patch_artist=True, notch=True)
box_plot['boxes'][0].set_facecolor('lightblue')
box_plot['boxes'][1].set_facecolor('coral')
axes[0,2].set_title('Age Distribution Comparison', fontweight='bold')
axes[0,2].set_ylabel('Age')
axes[0,2].grid(True, alpha=0.3)

# 4. Violin plot
violin_data = pd.DataFrame({
    'age': np.concatenate([no_stroke_ages, stroke_ages]),
    'stroke_status': ['No Stroke'] * len(no_stroke_ages) + ['Stroke'] * len(stroke_ages)
})
sns.violinplot(data=violin_data, x='stroke_status', y='age', ax=axes[1,0])
axes[1,0].set_title('Age Distribution Shape Comparison', fontweight='bold')
axes[1,0].grid(True, alpha=0.3)

# 5. Cumulative distribution
ages_sorted_no_stroke = np.sort(no_stroke_ages)
ages_sorted_stroke = np.sort(stroke_ages)
y_no_stroke = np.arange(1, len(ages_sorted_no_stroke) + 1) / len(ages_sorted_no_stroke)
y_stroke = np.arange(1, len(ages_sorted_stroke) + 1) / len(ages_sorted_stroke)

axes[1,1].plot(ages_sorted_no_stroke, y_no_stroke, label='No Stroke', 
               linewidth=2, color='blue')
axes[1,1].plot(ages_sorted_stroke, y_stroke, label='Stroke', 
               linewidth=2, color='red')
axes[1,1].set_title('Cumulative Distribution Function', fontweight='bold')
axes[1,1].set_xlabel('Age')
axes[1,1].set_ylabel('Cumulative Probability')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

# 6. Statistical summary table as text
summary_stats = df.groupby('stroke')['age'].describe().round(2)
axes[1,2].axis('off')
table_text = "Statistical Summary:\n\n"
table_text += f"No Stroke (n={len(no_stroke_ages)}):\n"
table_text += f"  Mean: {summary_stats.loc[0, 'mean']:.1f} years\n"
table_text += f"  Median: {summary_stats.loc[0, '50%']:.1f} years\n"
table_text += f"  Std: {summary_stats.loc[0, 'std']:.1f} years\n\n"
table_text += f"Stroke (n={len(stroke_ages)}):\n"
table_text += f"  Mean: {summary_stats.loc[1, 'mean']:.1f} years\n"
table_text += f"  Median: {summary_stats.loc[1, '50%']:.1f} years\n"
table_text += f"  Std: {summary_stats.loc[1, 'std']:.1f} years\n\n"

# Calculate statistical significance (t-test)

t_stat, p_value = stats.ttest_ind(stroke_ages, no_stroke_ages)
table_text += f"T-test p-value: {p_value:.2e}\n"
table_text += "Difference is statistically significant" if p_value < 0.05 else "Difference is not significant"

axes[1,2].text(0.1, 0.9, table_text, transform=axes[1,2].transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
axes[1,2].set_title('Statistical Analysis', fontweight='bold')

plt.tight_layout()
plt.show()

# Additional single plots for clearer focus:

# Plot 1: Clean stroke rate by age
plt.figure(figsize=(10, 6))
plt.bar(stroke_rate['age_midpoint'], stroke_rate['mean'] * 100, 
        width=8, alpha=0.8, color='darkred', edgecolor='black', linewidth=1)
plt.title('Stroke Rate by Age Group', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Stroke Rate (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, max(stroke_rate['mean'] * 100) * 1.1)

# Add value labels on bars
for i, v in enumerate(stroke_rate['mean'] * 100):
    plt.text(stroke_rate['age_midpoint'].iloc[i], v + 0.2, f'{v:.1f}%', 
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Plot 2: Side-by-side comparison with better scaling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left: Absolute counts (log scale for better visibility)
bins = np.arange(0, 101, 5)
ax1.hist(no_stroke_ages, bins=bins, alpha=0.6, label=f'No Stroke (n={len(no_stroke_ages)})', 
         color='lightblue', edgecolor='blue')
ax1.hist(stroke_ages, bins=bins, alpha=0.9, label=f'Stroke (n={len(stroke_ages)})', 
         color='red', edgecolor='darkred')
ax1.set_yscale('log')
ax1.set_title('Age Distribution - Log Scale', fontweight='bold')
ax1.set_xlabel('Age')
ax1.set_ylabel('Count (log scale)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Percentage within each group
ax2.hist(no_stroke_ages, bins=bins, alpha=0.6, label='No Stroke', 
         density=True, color='lightblue', edgecolor='blue')
ax2.hist(stroke_ages, bins=bins, alpha=0.9, label='Stroke', 
         density=True, color='red', edgecolor='darkred')
ax2.set_title('Age Distribution - Normalized', fontweight='bold')
ax2.set_xlabel('Age')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Key Insights:")
print(f"1. Average age of stroke patients: {stroke_ages.mean():.1f} years")
print(f"2. Average age of non-stroke patients: {no_stroke_ages.mean():.1f} years")
print(f"3. Age difference: {stroke_ages.mean() - no_stroke_ages.mean():.1f} years")
print(f"4. Peak stroke rate appears in the {stroke_rate.loc[stroke_rate['mean'].idxmax(), 'age']} age group")





df = stroke_data.copy(deep=True)

# Clean data - remove extreme outliers
df = df[(df['avg_glucose_level'] > 50) & (df['avg_glucose_level'] < 300)]

# Create two focused plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Stroke Rate by Glucose Ranges
glucose_bins = pd.cut(df['avg_glucose_level'], bins=np.arange(50, 301, 25), right=False)
stroke_rate = df.groupby(glucose_bins)['stroke'].agg(['mean', 'count']).reset_index()
stroke_rate['glucose_midpoint'] = stroke_rate['avg_glucose_level'].apply(lambda x: x.mid)

# Only show bins with reasonable sample sizes
stroke_rate = stroke_rate[stroke_rate['count'] >= 20]

bars = ax1.bar(stroke_rate['glucose_midpoint'], stroke_rate['mean'] * 100, 
               width=20, alpha=0.8, color='darkred', edgecolor='black')
ax1.set_title('Stroke Rate by Glucose Level', fontweight='bold', fontsize=14)
ax1.set_xlabel('Average Glucose Level (mg/dL)')
ax1.set_ylabel('Stroke Rate (%)')
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, rate, count in zip(bars, stroke_rate['mean'] * 100, stroke_rate['count']):
    if rate > 0.5:  # Only label bars with meaningful rates
        ax1.text(bar.get_x() + bar.get_width()/2, rate + 0.1, 
                f'{rate:.1f}%\n(n={count})', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')

# Plot 2: Side-by-side Distribution Comparison
stroke_data = df[df['stroke'] == 1]['avg_glucose_level']
no_stroke_data = df[df['stroke'] == 0]['avg_glucose_level']

# Use density to make both distributions visible
bins = np.arange(50, 301, 10)
ax2.hist(no_stroke_data, bins=bins, alpha=0.6, density=True, 
         label=f'No Stroke (n={len(no_stroke_data)})', color='lightblue', edgecolor='blue')
ax2.hist(stroke_data, bins=bins, alpha=0.9, density=True, 
         label=f'Stroke (n={len(stroke_data)})', color='red', edgecolor='darkred')

ax2.set_title('Glucose Distribution Comparison', fontweight='bold', fontsize=14)
ax2.set_xlabel('Average Glucose Level (mg/dL)')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add vertical lines for means
ax2.axvline(no_stroke_data.mean(), color='blue', linestyle='--', alpha=0.8, 
           label=f'No Stroke Mean: {no_stroke_data.mean():.1f}')
ax2.axvline(stroke_data.mean(), color='red', linestyle='--', alpha=0.8,
           label=f'Stroke Mean: {stroke_data.mean():.1f}')

plt.tight_layout()
plt.show()

# Quick stats summary
print("📊 GLUCOSE vs STROKE SUMMARY")
print("=" * 40)
print(f"No Stroke - Mean glucose: {no_stroke_data.mean():.1f} mg/dL")
print(f"Stroke    - Mean glucose: {stroke_data.mean():.1f} mg/dL")
print(f"Difference: {stroke_data.mean() - no_stroke_data.mean():.1f} mg/dL")

# Find the glucose range with highest stroke rate
max_rate_idx = stroke_rate['mean'].idxmax()
max_rate_range = stroke_rate.loc[max_rate_idx, 'avg_glucose_level']
max_rate_value = stroke_rate.loc[max_rate_idx, 'mean'] * 100
print(f"Highest stroke rate: {max_rate_value:.1f}% in range {max_rate_range}")