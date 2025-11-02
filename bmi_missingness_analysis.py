import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def comprehensive_missingness_analysis(df):
    """
    Comprehensive analysis to determine if BMI missingness is random or systematic.
    
    This function examines:
    1. Overall missingness patterns
    2. Missingness by demographic groups
    3. Statistical tests for randomness
    4. Visualization of missing patterns
    """
    
    # Create a copy to avoid modifying original data
    df_analysis = df.copy()
    
    # Add missingness indicator
    df_analysis['bmi_missing'] = df_analysis['bmi'].isna()
    
    # Set up the plotting area
    fig = plt.figure(figsize=(20, 16))
    
    print("🔍 COMPREHENSIVE BMI MISSINGNESS ANALYSIS")
    print("=" * 60)
    
    # 1. OVERALL MISSINGNESS SUMMARY
    total_rows = len(df_analysis)
    missing_count = df_analysis['bmi'].isna().sum()
    missing_rate = (missing_count / total_rows) * 100
    
    print(f"\n📊 OVERALL MISSINGNESS:")
    print(f"   Total observations: {total_rows:,}")
    print(f"   Missing BMI values: {missing_count:,}")
    print(f"   Missing rate: {missing_rate:.2f}%")
    
    # 2. MISSINGNESS BY KEY VARIABLES
    print(f"\n📈 MISSINGNESS BY DEMOGRAPHIC GROUPS:")
    
    # Analyze by stroke status
    if 'stroke' in df_analysis.columns:
        stroke_missing = df_analysis.groupby('stroke')['bmi_missing'].agg(['count', 'sum', 'mean']).round(3)
        stroke_missing.columns = ['Total', 'Missing', 'Missing_Rate']
        stroke_missing['Missing_Rate'] = stroke_missing['Missing_Rate'] * 100
        print(f"\n   By Stroke Status:")
        for idx, row in stroke_missing.iterrows():
            status = "Stroke" if idx == 1 else "No Stroke"
            print(f"   {status:12}: {row['Missing']:3.0f}/{row['Total']:4.0f} missing ({row['Missing_Rate']:5.1f}%)")
    
    # Analyze by age groups
    if 'age' in df_analysis.columns:
        df_analysis['age_group'] = pd.cut(df_analysis['age'], 
                                        bins=[0, 30, 50, 65, 80, 100], 
                                        labels=['<30', '30-49', '50-64', '65-79', '80+'],
                                        right=False)
        age_missing = df_analysis.groupby('age_group')['bmi_missing'].agg(['count', 'sum', 'mean']).round(3)
        age_missing.columns = ['Total', 'Missing', 'Missing_Rate']
        age_missing['Missing_Rate'] = age_missing['Missing_Rate'] * 100
        print(f"\n   By Age Groups:")
        for idx, row in age_missing.iterrows():
            print(f"   {str(idx):8}: {row['Missing']:3.0f}/{row['Total']:4.0f} missing ({row['Missing_Rate']:5.1f}%)")
    
    # Analyze by gender if available
    if 'gender' in df_analysis.columns:
        gender_missing = df_analysis.groupby('gender')['bmi_missing'].agg(['count', 'sum', 'mean']).round(3)
        gender_missing.columns = ['Total', 'Missing', 'Missing_Rate']
        gender_missing['Missing_Rate'] = gender_missing['Missing_Rate'] * 100
        print(f"\n   By Gender:")
        for idx, row in gender_missing.iterrows():
            print(f"   {str(idx):8}: {row['Missing']:3.0f}/{row['Total']:4.0f} missing ({row['Missing_Rate']:5.1f}%)")
    
    # 3. STATISTICAL TESTS FOR RANDOMNESS
    print(f"\n🧪 STATISTICAL TESTS FOR RANDOMNESS:")
    
    # Chi-square test for stroke vs missingness
    if 'stroke' in df_analysis.columns:
        contingency_stroke = pd.crosstab(df_analysis['stroke'], df_analysis['bmi_missing'])
        chi2_stroke, p_stroke = stats.chi2_contingency(contingency_stroke)[:2]
        print(f"\n   Stroke vs BMI Missingness:")
        print(f"   Chi-square statistic: {chi2_stroke:.4f}")
        print(f"   P-value: {p_stroke:.6f}")
        print(f"   Interpretation: {'NOT random (systematic)' if p_stroke < 0.05 else 'Appears random'}")
    
    # T-test for age differences
    if 'age' in df_analysis.columns:
        age_missing = df_analysis[df_analysis['bmi_missing']]['age'].dropna()
        age_not_missing = df_analysis[~df_analysis['bmi_missing']]['age'].dropna()
        t_stat, p_age = stats.ttest_ind(age_missing, age_not_missing)
        print(f"\n   Age Difference Test:")
        print(f"   T-statistic: {t_stat:.4f}")
        print(f"   P-value: {p_age:.6f}")
        print(f"   Mean age (missing BMI): {age_missing.mean():.1f}")
        print(f"   Mean age (non-missing BMI): {age_not_missing.mean():.1f}")
        print(f"   Interpretation: {'Significant age difference' if p_age < 0.05 else 'No significant age difference'}")
    
    # 4. VISUALIZATIONS
    
    # Plot 1: Missing pattern by stroke status
    ax1 = plt.subplot(3, 3, 1)
    if 'stroke' in df_analysis.columns:
        stroke_missing_plot = df_analysis.groupby('stroke')['bmi_missing'].mean() * 100
        bars = ax1.bar(['No Stroke', 'Stroke'], stroke_missing_plot.values, 
                      color=['lightblue', 'coral'], alpha=0.8, edgecolor='black')
        ax1.set_title('BMI Missing Rate by Stroke Status', fontweight='bold')
        ax1.set_ylabel('Missing Rate (%)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, stroke_missing_plot.values):
            ax1.text(bar.get_x() + bar.get_width()/2, rate + 0.5, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Missing pattern by age groups
    ax2 = plt.subplot(3, 3, 2)
    if 'age' in df_analysis.columns:
        age_missing_plot = df_analysis.groupby('age_group')['bmi_missing'].mean() * 100
        bars = ax2.bar(range(len(age_missing_plot)), age_missing_plot.values, 
                      color='skyblue', alpha=0.8, edgecolor='black')
        ax2.set_title('BMI Missing Rate by Age Group', fontweight='bold')
        ax2.set_ylabel('Missing Rate (%)')
        ax2.set_xticks(range(len(age_missing_plot)))
        ax2.set_xticklabels(age_missing_plot.index, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, age_missing_plot.values):
            ax2.text(bar.get_x() + bar.get_width()/2, rate + 0.5, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Age distribution comparison
    ax3 = plt.subplot(3, 3, 3)
    if 'age' in df_analysis.columns:
        age_missing = df_analysis[df_analysis['bmi_missing']]['age'].dropna()
        age_not_missing = df_analysis[~df_analysis['bmi_missing']]['age'].dropna()
        
        ax3.hist(age_not_missing, bins=20, alpha=0.6, density=True, 
                label=f'BMI Available (n={len(age_not_missing)})', color='lightblue')
        ax3.hist(age_missing, bins=20, alpha=0.8, density=True, 
                label=f'BMI Missing (n={len(age_missing)})', color='red')
        
        ax3.axvline(age_not_missing.mean(), color='blue', linestyle='--', alpha=0.8)
        ax3.axvline(age_missing.mean(), color='red', linestyle='--', alpha=0.8)
        ax3.set_title('Age Distribution: Missing vs Available BMI', fontweight='bold')
        ax3.set_xlabel('Age')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Box plot of age by missingness
    ax4 = plt.subplot(3, 3, 4)
    if 'age' in df_analysis.columns:
        box_data = [age_not_missing, age_missing]
        box_plot = ax4.boxplot(box_data, labels=['BMI Available', 'BMI Missing'], 
                              patch_artist=True, showfliers=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('coral')
        ax4.set_title('Age Distribution by BMI Availability', fontweight='bold')
        ax4.set_ylabel('Age')
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Missing pattern heatmap (if multiple variables)
    ax5 = plt.subplot(3, 3, 5)
    # Create a simple missingness correlation matrix
    missing_cols = df_analysis.select_dtypes(include=[np.number]).columns
    missing_matrix = df_analysis[missing_cols].isna().astype(int)
    if len(missing_matrix.columns) > 1:
        corr_missing = missing_matrix.corr()
        im = ax5.imshow(corr_missing, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        ax5.set_title('Missingness Correlation Matrix', fontweight='bold')
        ax5.set_xticks(range(len(corr_missing.columns)))
        ax5.set_yticks(range(len(corr_missing.columns)))
        ax5.set_xticklabels(corr_missing.columns, rotation=45)
        ax5.set_yticklabels(corr_missing.columns)
        plt.colorbar(im, ax=ax5)
    else:
        ax5.text(0.5, 0.5, 'Only BMI has\nmissing values', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Missingness Correlation Matrix', fontweight='bold')
    
    # Plot 6: Random pattern test visualization
    ax6 = plt.subplot(3, 3, 6)
    # Create a random sample pattern for comparison
    n_sample = min(1000, len(df_analysis))
    sample_indices = np.random.choice(len(df_analysis), n_sample, replace=False)
    sample_df = df_analysis.iloc[sample_indices].copy()
    
    if 'age' in df_analysis.columns:
        # Scatter plot of age vs index to show pattern
        missing_idx = sample_df[sample_df['bmi_missing']].index
        not_missing_idx = sample_df[~sample_df['bmi_missing']].index
        
        ax6.scatter(not_missing_idx, sample_df.loc[not_missing_idx, 'age'], 
                   alpha=0.6, s=20, color='blue', label='BMI Available')
        ax6.scatter(missing_idx, sample_df.loc[missing_idx, 'age'], 
                   alpha=0.8, s=20, color='red', label='BMI Missing')
        ax6.set_title('Missing Pattern by Record Order', fontweight='bold')
        ax6.set_xlabel('Record Index')
        ax6.set_ylabel('Age')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # Plot 7: Missing counts by other variables (if available)
    ax7 = plt.subplot(3, 3, 7)
    if 'gender' in df_analysis.columns:
        gender_missing_plot = df_analysis.groupby('gender')['bmi_missing'].mean() * 100
        bars = ax7.bar(gender_missing_plot.index, gender_missing_plot.values, 
                      color='lightgreen', alpha=0.8, edgecolor='black')
        ax7.set_title('BMI Missing Rate by Gender', fontweight='bold')
        ax7.set_ylabel('Missing Rate (%)')
        ax7.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, gender_missing_plot.values):
            ax7.text(bar.get_x() + bar.get_width()/2, rate + 0.5, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    else:
        ax7.text(0.5, 0.5, 'Gender variable\nnot available', 
                ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('BMI Missing Rate by Gender', fontweight='bold')
    
    # Plot 8: Cumulative missing pattern
    ax8 = plt.subplot(3, 3, 8)
    # Show cumulative missing pattern across the dataset
    cumulative_missing = df_analysis['bmi_missing'].cumsum()
    ax8.plot(cumulative_missing, color='red', linewidth=2)
    ax8.set_title('Cumulative Missing BMI Values', fontweight='bold')
    ax8.set_xlabel('Record Number')
    ax8.set_ylabel('Cumulative Missing Count')
    ax8.grid(True, alpha=0.3)
    
    # Add expected random line
    expected_random = np.arange(len(df_analysis)) * (missing_count / len(df_analysis))
    ax8.plot(expected_random, color='blue', linestyle='--', alpha=0.7, 
            label='Expected if Random')
    ax8.legend()
    
    # Plot 9: Summary statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('tight')
    ax9.axis('off')
    
    # Create summary table
    summary_data = []
    if 'stroke' in df_analysis.columns:
        summary_data.append(['Stroke Association', f'p={p_stroke:.4f}', 
                           'Systematic' if p_stroke < 0.05 else 'Random'])
    if 'age' in df_analysis.columns:
        summary_data.append(['Age Difference', f'p={p_age:.4f}', 
                           'Significant' if p_age < 0.05 else 'Not Significant'])
    
    summary_data.extend([
        ['Total Missing', f'{missing_count:,}', f'{missing_rate:.1f}%'],
        ['Sample Size', f'{total_rows:,}', '100%']
    ])
    
    table = ax9.table(cellText=summary_data,
                     colLabels=['Test', 'Statistic', 'Result'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax9.set_title('Summary of Missingness Tests', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # 5. CONCLUSIONS AND RECOMMENDATIONS
    print(f"\n🎯 CONCLUSIONS AND RECOMMENDATIONS:")
    print("-" * 40)
    
    # Determine if missingness is random
    is_random = True
    reasons = []
    
    if 'stroke' in df_analysis.columns and p_stroke < 0.05:
        is_random = False
        reasons.append(f"BMI missingness differs significantly by stroke status (p={p_stroke:.4f})")
    
    if 'age' in df_analysis.columns and p_age < 0.05:
        is_random = False
        reasons.append(f"Age differs significantly between missing/non-missing BMI groups (p={p_age:.4f})")
    
    if is_random:
        print("✅ MISSINGNESS APPEARS TO BE RANDOM")
        print("   - No significant associations found with key variables")
        print("   - Missing Completely at Random (MCAR) or Missing at Random (MAR)")
        print("   - Simple imputation methods should work well")
        print("   - Complete case analysis may be appropriate")
    else:
        print("⚠️  MISSINGNESS APPEARS TO BE SYSTEMATIC")
        print("   - Significant associations found:")
        for reason in reasons:
            print(f"     • {reason}")
        print("   - Missing Not at Random (MNAR) likely")
        print("   - Need careful imputation strategy")
        print("   - Consider multiple imputation or specialized methods")
        print("   - Complete case analysis may introduce bias")
    
    print(f"\n📋 RECOMMENDED NEXT STEPS:")
    if is_random:
        print("   1. Consider simple imputation (mean, median, mode)")
        print("   2. Or use complete case analysis")
        print("   3. Validate results with sensitivity analysis")
    else:
        print("   1. Investigate reasons for systematic missingness")
        print("   2. Use multiple imputation methods")
        print("   3. Consider pattern-mixture models")
        print("   4. Include missingness indicators in analysis")
        print("   5. Perform sensitivity analysis for different scenarios")
    
    return df_analysis

# Example usage with sample data (replace with your actual dataframe)
# Recreate sample data similar to your original code
np.random.seed(42)
n_total = 5000
n_stroke = 250

# Create sample data
no_stroke_bmi = np.random.normal(26, 5, n_total - n_stroke)
stroke_bmi = np.random.normal(30, 6, n_stroke)

# Add some systematic missingness patterns for demonstration
df_sample = pd.DataFrame({
    'bmi': np.concatenate([no_stroke_bmi, stroke_bmi]),
    'stroke': np.concatenate([np.zeros(n_total - n_stroke), np.ones(n_stroke)]),
    'age': np.random.normal(60, 15, n_total),
    'gender': np.random.choice(['Male', 'Female'], n_total)
})

# Add systematic missing values (more missing in older patients and stroke patients)
missing_prob = 0.02 + 0.03 * df_sample['stroke'] + 0.001 * (df_sample['age'] - 40).clip(0)
missing_mask = np.random.random(n_total) < missing_prob
df_sample.loc[missing_mask, 'bmi'] = np.nan

print("Sample dataset created with systematic missingness patterns")
print(f"Total missing BMI values: {df_sample['bmi'].isna().sum()}")

# Run the analysis
result_df = comprehensive_missingness_analysis(df_sample)