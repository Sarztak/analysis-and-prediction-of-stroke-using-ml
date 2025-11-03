import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_smoking_stroke_risk(df):
    """
    Comprehensive analysis of smoking status and stroke risk including:
    1. Group-level stroke rates by smoking status
    2. Distribution checks and "Unknown" category analysis
    3. Age-smoking interaction effects
    4. Statistical significance testing
    """
    
    # Create a copy to avoid modifying original data
    df_analysis = df.copy()
    
    # Set up the plotting area
    fig = plt.figure(figsize=(20, 16))
    
    print("🚬 COMPREHENSIVE SMOKING STATUS vs STROKE ANALYSIS")
    print("=" * 65)
    
    # 1. OVERALL DISTRIBUTION OF SMOKING STATUS
    print(f"\n📊 SMOKING STATUS DISTRIBUTION:")
    smoking_counts = df_analysis['smoking_status'].value_counts().sort_index()
    smoking_props = df_analysis['smoking_status'].value_counts(normalize=True).sort_index() * 100
    
    for status in smoking_counts.index:
        count = smoking_counts[status]
        prop = smoking_props[status]
        print(f"   {status:15}: {count:5,} ({prop:5.1f}%)")
    
    total_cases = len(df_analysis)
    print(f"   {'Total':15}: {total_cases:5,} (100.0%)")
    
    # 2. STROKE RATES BY SMOKING STATUS
    print(f"\n🎯 STROKE RATES BY SMOKING STATUS:")
    stroke_by_smoking = df_analysis.groupby('smoking_status').agg({
        'stroke': ['count', 'sum', 'mean']
    }).round(4)
    stroke_by_smoking.columns = ['Total_Cases', 'Stroke_Cases', 'Stroke_Rate']
    stroke_by_smoking['Stroke_Rate_Pct'] = stroke_by_smoking['Stroke_Rate'] * 100
    stroke_by_smoking = stroke_by_smoking.sort_values('Stroke_Rate_Pct', ascending=False)
    
    for status, row in stroke_by_smoking.iterrows():
        print(f"   {status:15}: {row['Stroke_Cases']:3.0f}/{row['Total_Cases']:5.0f} = {row['Stroke_Rate_Pct']:5.1f}%")
    
    # 3. UNKNOWN CATEGORY DEEP DIVE
    print(f"\n🔍 'UNKNOWN' CATEGORY ANALYSIS:")
    if 'Unknown' in df_analysis['smoking_status'].values:
        unknown_data = df_analysis[df_analysis['smoking_status'] == 'Unknown']
        known_data = df_analysis[df_analysis['smoking_status'] != 'Unknown']
        
        # Age comparison
        unknown_age_mean = unknown_data['age'].mean()
        known_age_mean = known_data['age'].mean()
        age_ttest = stats.ttest_ind(unknown_data['age'].dropna(), known_data['age'].dropna())
        
        print(f"   Unknown cases: {len(unknown_data):,}")
        print(f"   Mean age (Unknown): {unknown_age_mean:.1f} years")
        print(f"   Mean age (Known): {known_age_mean:.1f} years")
        print(f"   Age difference p-value: {age_ttest.pvalue:.6f}")
        print(f"   {'Significantly older' if age_ttest.pvalue < 0.05 and unknown_age_mean > known_age_mean else 'Not significantly different'}")
        
        # Stroke rate comparison
        unknown_stroke_rate = unknown_data['stroke'].mean() * 100
        known_stroke_rate = known_data['stroke'].mean() * 100
        stroke_ttest = stats.ttest_ind(unknown_data['stroke'], known_data['stroke'])
        
        print(f"   Stroke rate (Unknown): {unknown_stroke_rate:.1f}%")
        print(f"   Stroke rate (Known): {known_stroke_rate:.1f}%")
        print(f"   Stroke rate difference p-value: {stroke_ttest.pvalue:.6f}")
        print(f"   {'Higher stroke rate in Unknown' if stroke_ttest.pvalue < 0.05 and unknown_stroke_rate > known_stroke_rate else 'No significant difference'}")
    else:
        print("   No 'Unknown' category found in data")
    
    # 4. AGE-SMOKING INTERACTION ANALYSIS
    print(f"\n🎂 AGE-SMOKING INTERACTION ANALYSIS:")
    
    # Create age groups
    df_analysis['age_group'] = pd.cut(df_analysis['age'], 
                                    bins=[0, 40, 60, 80, 100], 
                                    labels=['≤40', '41-60', '61-80', '>80'],
                                    right=True)
    
    # Focus on smoking & age > 60 interaction
    elderly_smokers = df_analysis[(df_analysis['age'] > 60) & 
                                 (df_analysis['smoking_status'].isin(['smokes', 'formerly smoked']))]
    elderly_nonsmokers = df_analysis[(df_analysis['age'] > 60) & 
                                   (df_analysis['smoking_status'] == 'never smoked')]
    young_smokers = df_analysis[(df_analysis['age'] <= 60) & 
                               (df_analysis['smoking_status'].isin(['smokes', 'formerly smoked']))]
    
    if len(elderly_smokers) > 0 and len(elderly_nonsmokers) > 0:
        elderly_smoker_stroke_rate = elderly_smokers['stroke'].mean() * 100
        elderly_nonsmoker_stroke_rate = elderly_nonsmokers['stroke'].mean() * 100
        young_smoker_stroke_rate = young_smokers['stroke'].mean() * 100 if len(young_smokers) > 0 else 0
        
        print(f"   Elderly (>60) Smokers: {len(elderly_smokers):,} cases, {elderly_smoker_stroke_rate:.1f}% stroke rate")
        print(f"   Elderly (>60) Non-smokers: {len(elderly_nonsmokers):,} cases, {elderly_nonsmoker_stroke_rate:.1f}% stroke rate")
        print(f"   Young (≤60) Smokers: {len(young_smokers):,} cases, {young_smoker_stroke_rate:.1f}% stroke rate")
        
        # Statistical test for interaction
        if len(elderly_smokers) > 0 and len(elderly_nonsmokers) > 0:
            interaction_test = stats.ttest_ind(elderly_smokers['stroke'], elderly_nonsmokers['stroke'])
            print(f"   Elderly smokers vs non-smokers p-value: {interaction_test.pvalue:.6f}")
    
    # 5. STATISTICAL TESTING
    print(f"\n🧪 STATISTICAL TESTS:")
    
    # Chi-square test for overall association
    contingency_table = pd.crosstab(df_analysis['smoking_status'], df_analysis['stroke'])
    chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"   Overall smoking-stroke association:")
    print(f"   Chi-square statistic: {chi2_stat:.4f}")
    print(f"   P-value: {chi2_p:.6f}")
    print(f"   Degrees of freedom: {dof}")
    print(f"   {'Significant association' if chi2_p < 0.05 else 'No significant association'}")
    
    # ANOVA for age differences across smoking groups
    smoking_groups = [group['age'].dropna() for name, group in df_analysis.groupby('smoking_status')]
    if len(smoking_groups) > 1:
        f_stat, anova_p = stats.f_oneway(*smoking_groups)
        print(f"   Age differences across smoking groups:")
        print(f"   F-statistic: {f_stat:.4f}")
        print(f"   P-value: {anova_p:.6f}")
        print(f"   {'Significant age differences' if anova_p < 0.05 else 'No significant age differences'}")
    
    # 6. VISUALIZATIONS
    
    # Plot 1: Stroke rates by smoking status
    ax1 = plt.subplot(3, 3, 1)
    smoking_order = stroke_by_smoking.index  # Ordered by stroke rate
    stroke_rates = stroke_by_smoking.loc[smoking_order, 'Stroke_Rate_Pct']
    colors = ['#d62728' if rate > stroke_rates.mean() else '#1f77b4' for rate in stroke_rates]
    
    bars = ax1.bar(range(len(smoking_order)), stroke_rates, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Stroke Rate by Smoking Status\n(Ordered by Risk)', fontweight='bold')
    ax1.set_ylabel('Stroke Rate (%)')
    ax1.set_xticks(range(len(smoking_order)))
    ax1.set_xticklabels(smoking_order, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, rate, status in zip(bars, stroke_rates, smoking_order):
        count = stroke_by_smoking.loc[status, 'Stroke_Cases']
        total = stroke_by_smoking.loc[status, 'Total_Cases']
        ax1.text(bar.get_x() + bar.get_width()/2, rate + 0.2, 
                f'{rate:.1f}%\n({count:.0f}/{total:.0f})', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Distribution of smoking status
    ax2 = plt.subplot(3, 3, 2)
    smoking_counts_plot = df_analysis['smoking_status'].value_counts()
    colors_dist = plt.cm.Set3(np.linspace(0, 1, len(smoking_counts_plot)))
    
    bars = ax2.bar(range(len(smoking_counts_plot)), smoking_counts_plot.values, 
                   color=colors_dist, alpha=0.8, edgecolor='black')
    ax2.set_title('Distribution of Smoking Status', fontweight='bold')
    ax2.set_ylabel('Count')
    ax2.set_xticks(range(len(smoking_counts_plot)))
    ax2.set_xticklabels(smoking_counts_plot.index, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count, status in zip(bars, smoking_counts_plot.values, smoking_counts_plot.index):
        pct = (count / total_cases) * 100
        ax2.text(bar.get_x() + bar.get_width()/2, count + 50, 
                f'{count:,}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 3: Age distribution by smoking status
    ax3 = plt.subplot(3, 3, 3)
    smoking_statuses = df_analysis['smoking_status'].unique()
    for i, status in enumerate(smoking_statuses):
        age_data = df_analysis[df_analysis['smoking_status'] == status]['age'].dropna()
        ax3.hist(age_data, bins=20, alpha=0.6, density=True, 
                label=f'{status} (n={len(age_data)})', color=colors_dist[i])
    
    ax3.set_title('Age Distribution by Smoking Status', fontweight='bold')
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Density')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Unknown category analysis
    ax4 = plt.subplot(3, 3, 4)
    if 'Unknown' in df_analysis['smoking_status'].values:
        categories = ['Known Status', 'Unknown Status']
        age_means = [known_age_mean, unknown_age_mean]
        stroke_rates_comp = [known_stroke_rate, unknown_stroke_rate]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, age_means, width, label='Mean Age', alpha=0.8, color='skyblue')
        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x + width/2, stroke_rates_comp, width, label='Stroke Rate (%)', alpha=0.8, color='coral')
        
        ax4.set_title('Unknown vs Known Status Comparison', fontweight='bold')
        ax4.set_ylabel('Mean Age', color='skyblue')
        ax4_twin.set_ylabel('Stroke Rate (%)', color='coral')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, age_means):
            ax4.text(bar.get_x() + bar.get_width()/2, val + 1, 
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        for bar, val in zip(bars2, stroke_rates_comp):
            ax4_twin.text(bar.get_x() + bar.get_width()/2, val + 0.2, 
                         f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No Unknown\nCategory Found', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Unknown Category Analysis', fontweight='bold')
    
    # Plot 5: Age-Smoking Interaction Heatmap
    ax5 = plt.subplot(3, 3, 5)
    interaction_matrix = df_analysis.groupby(['age_group', 'smoking_status'])['stroke'].mean().unstack(fill_value=0) * 100
    
    if not interaction_matrix.empty:
        im = ax5.imshow(interaction_matrix.values, cmap='Reds', aspect='auto')
        ax5.set_title('Stroke Rate by Age Group & Smoking Status', fontweight='bold')
        ax5.set_xticks(range(len(interaction_matrix.columns)))
        ax5.set_yticks(range(len(interaction_matrix.index)))
        ax5.set_xticklabels(interaction_matrix.columns, rotation=45, ha='right')
        ax5.set_yticklabels(interaction_matrix.index)
        
        # Add text annotations
        for i in range(len(interaction_matrix.index)):
            for j in range(len(interaction_matrix.columns)):
                value = interaction_matrix.iloc[i, j]
                ax5.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                        color='white' if value > interaction_matrix.values.mean() else 'black',
                        fontweight='bold')
        
        plt.colorbar(im, ax=ax5, label='Stroke Rate (%)')
    
    # Plot 6: Elderly smokers vs others
    ax6 = plt.subplot(3, 3, 6)
    if len(elderly_smokers) > 0 and len(elderly_nonsmokers) > 0:
        groups = ['Elderly\nSmokers\n(>60)', 'Elderly\nNon-smokers\n(>60)', 'Young\nSmokers\n(≤60)']
        stroke_rates_groups = [elderly_smoker_stroke_rate, elderly_nonsmoker_stroke_rate, young_smoker_stroke_rate]
        counts = [len(elderly_smokers), len(elderly_nonsmokers), len(young_smokers)]
        
        colors_risk = ['#d62728' if rate > 10 else '#ff7f0e' if rate > 5 else '#2ca02c' for rate in stroke_rates_groups]
        bars = ax6.bar(groups, stroke_rates_groups, color=colors_risk, alpha=0.8, edgecolor='black')
        
        ax6.set_title('High-Risk Group Analysis:\nAge-Smoking Interaction', fontweight='bold')
        ax6.set_ylabel('Stroke Rate (%)')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate, count in zip(bars, stroke_rates_groups, counts):
            ax6.text(bar.get_x() + bar.get_width()/2, rate + 0.3, 
                    f'{rate:.1f}%\n(n={count:,})', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 7: Box plot of age by smoking status
    ax7 = plt.subplot(3, 3, 7)
    age_by_smoking = [df_analysis[df_analysis['smoking_status'] == status]['age'].dropna() 
                     for status in smoking_statuses]
    
    box_plot = ax7.boxplot(age_by_smoking, labels=smoking_statuses, patch_artist=True, showfliers=True)
    for patch, color in zip(box_plot['boxes'], colors_dist):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    ax7.set_title('Age Distribution by Smoking Status', fontweight='bold')
    ax7.set_ylabel('Age')
    ax7.set_xticklabels(smoking_statuses, rotation=45, ha='right')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Contingency table heatmap
    ax8 = plt.subplot(3, 3, 8)
    contingency_pct = pd.crosstab(df_analysis['smoking_status'], df_analysis['stroke'], normalize='index') * 100
    
    im = ax8.imshow(contingency_pct.values, cmap='RdYlBu_r', aspect='auto')
    ax8.set_title('Stroke Rate by Smoking Status\n(Normalized by Row)', fontweight='bold')
    ax8.set_xticks([0, 1])
    ax8.set_xticklabels(['No Stroke', 'Stroke'])
    ax8.set_yticks(range(len(contingency_pct.index)))
    ax8.set_yticklabels(contingency_pct.index)
    
    # Add text annotations
    for i in range(len(contingency_pct.index)):
        for j in range(len(contingency_pct.columns)):
            value = contingency_pct.iloc[i, j]
            ax8.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                    color='white' if value < 50 else 'black', fontweight='bold')
    
    plt.colorbar(im, ax=ax8, label='Percentage')
    
    # Plot 9: Summary statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('tight')
    ax9.axis('off')
    
    # Create summary table
    summary_data = [
        ['Overall Association', f'χ²={chi2_stat:.2f}, p={chi2_p:.4f}', 
         'Significant' if chi2_p < 0.05 else 'Not Significant'],
        ['Age Differences', f'F={f_stat:.2f}, p={anova_p:.4f}' if len(smoking_groups) > 1 else 'N/A', 
         'Significant' if len(smoking_groups) > 1 and anova_p < 0.05 else 'Not Significant'],
        ['Highest Risk Group', smoking_order[0], f'{stroke_by_smoking.iloc[0]["Stroke_Rate_Pct"]:.1f}%'],
        ['Lowest Risk Group', smoking_order[-1], f'{stroke_by_smoking.iloc[-1]["Stroke_Rate_Pct"]:.1f}%']
    ]
    
    if 'Unknown' in df_analysis['smoking_status'].values:
        summary_data.append(['Unknown Category', f'{len(unknown_data):,} cases', 
                           f'{unknown_stroke_rate:.1f}% stroke rate'])
    
    table = ax9.table(cellText=summary_data,
                     colLabels=['Analysis', 'Statistic', 'Result'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax9.set_title('Statistical Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # 7. DETAILED CONCLUSIONS
    print(f"\n🎯 KEY FINDINGS & CONCLUSIONS:")
    print("-" * 50)
    
    # Risk ranking
    print(f"📈 STROKE RISK RANKING (Highest to Lowest):")
    for i, (status, row) in enumerate(stroke_by_smoking.iterrows(), 1):
        risk_level = "🔴 HIGH" if row['Stroke_Rate_Pct'] > 8 else "🟡 MEDIUM" if row['Stroke_Rate_Pct'] > 4 else "🟢 LOW"
        print(f"   {i}. {status:15} - {row['Stroke_Rate_Pct']:5.1f}% {risk_level}")
    
    # Statistical significance
    print(f"\n🧪 STATISTICAL SIGNIFICANCE:")
    print(f"   Overall smoking-stroke association: {'✅ SIGNIFICANT' if chi2_p < 0.05 else '❌ NOT SIGNIFICANT'} (p={chi2_p:.4f})")
    
    # Age-smoking interaction
    if len(elderly_smokers) > 0 and len(elderly_nonsmokers) > 0:
        print(f"\n🎂 AGE-SMOKING INTERACTION:")
        print(f"   Elderly smokers (>60): {elderly_smoker_stroke_rate:.1f}% stroke rate")
        print(f"   Elderly non-smokers (>60): {elderly_nonsmoker_stroke_rate:.1f}% stroke rate")
        risk_multiplier = elderly_smoker_stroke_rate / elderly_nonsmoker_stroke_rate if elderly_nonsmoker_stroke_rate > 0 else 0
        print(f"   Risk multiplier: {risk_multiplier:.1f}x higher in elderly smokers")
        print(f"   Statistical significance: {'✅ SIGNIFICANT' if interaction_test.pvalue < 0.05 else '❌ NOT SIGNIFICANT'}")
    
    # Unknown category insights
    if 'Unknown' in df_analysis['smoking_status'].values:
        print(f"\n🔍 UNKNOWN CATEGORY INSIGHTS:")
        print(f"   Sample size: {len(unknown_data):,} ({len(unknown_data)/total_cases*100:.1f}% of total)")
        print(f"   Age bias: {'✅ Significantly older' if age_ttest.pvalue < 0.05 and unknown_age_mean > known_age_mean else '❌ No significant age difference'}")
        print(f"   Stroke bias: {'✅ Higher stroke rate' if stroke_ttest.pvalue < 0.05 and unknown_stroke_rate > known_stroke_rate else '❌ No significant difference'}")
    
    # Recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    print(f"   1. {'Focus on elderly smokers as high-risk group' if len(elderly_smokers) > 0 and elderly_smoker_stroke_rate > 10 else 'Monitor smoking patterns across age groups'}")
    print(f"   2. {'Investigate Unknown category for potential bias' if 'Unknown' in df_analysis['smoking_status'].values and len(unknown_data) > total_cases * 0.1 else 'Unknown category appears manageable'}")
    print(f"   3. {'Include smoking status in predictive models' if chi2_p < 0.05 else 'Smoking status may not be a strong predictor'}")
    print(f"   4. Consider age-smoking interaction terms in modeling")
    
    return df_analysis, stroke_by_smoking

# Example usage with sample data (replace with your actual dataframe)
np.random.seed(42)
n_total = 5000

# Create realistic smoking distribution
smoking_probs = [0.15, 0.25, 0.45, 0.15]  # smokes, formerly smoked, never smoked, unknown
smoking_statuses = np.random.choice(['smokes', 'formerly smoked', 'never smoked', 'Unknown'], 
                                   n_total, p=smoking_probs)

# Create age with realistic distribution
ages = np.random.gamma(2, 30) + 20  # Skewed towards older ages
ages = np.clip(ages, 18, 95)

# Create stroke outcome with realistic relationships
stroke_base_rates = {'smokes': 0.12, 'formerly smoked': 0.08, 'never smoked': 0.04, 'Unknown': 0.10}
stroke_probs = np.array([stroke_base_rates[status] for status in smoking_statuses])

# Add age effect (higher risk with age)
age_effect = (ages - 40) * 0.001
age_effect = np.clip(age_effect, 0, 0.05)
stroke_probs += age_effect

# Add interaction effect (smoking + elderly = higher risk)
elderly_smoker_bonus = np.where((ages > 60) & 
                               (np.isin(smoking_statuses, ['smokes', 'formerly smoked'])), 
                               0.03, 0)
stroke_probs += elderly_smoker_bonus

# Generate stroke outcomes
strokes = np.random.binomial(1, stroke_probs)

# Create sample dataframe
df_sample = pd.DataFrame({
    'smoking_status': smoking_statuses,
    'age': ages,
    'stroke': strokes
})

print("Sample dataset created with realistic smoking-stroke relationships")
print(f"Total sample size: {len(df_sample):,}")
print(f"Overall stroke rate: {df_sample['stroke'].mean()*100:.1f}%")

# Run the analysis
result_df, stroke_summary = analyze_smoking_stroke_risk(df_sample)