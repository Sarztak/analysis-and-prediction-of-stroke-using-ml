import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_bmi(df):
    # Create the four focused plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BMI vs Stroke Analysis: Four Key Questions', fontsize=16, fontweight='bold')

    # Question 1: How is BMI distributed for stroke vs. non-stroke?
    ax1 = axes[0, 0]
    stroke_data = df[df['stroke'] == 1]['bmi'].dropna()
    no_stroke_data = df[df['stroke'] == 0]['bmi'].dropna()

    bins = np.arange(10, 81, 2)  # Wide range to capture all data including outliers
    ax1.hist(no_stroke_data, bins=bins, alpha=0.6, density=True, 
            label=f'No Stroke (n={len(no_stroke_data)})', color='lightblue', edgecolor='blue')
    ax1.hist(stroke_data, bins=bins, alpha=0.9, density=True, 
            label=f'Stroke (n={len(stroke_data)})', color='red', edgecolor='darkred')

    ax1.axvline(no_stroke_data.mean(), color='blue', linestyle='--', alpha=0.8)
    ax1.axvline(stroke_data.mean(), color='red', linestyle='--', alpha=0.8)
    ax1.set_title('BMI Distribution Comparison\n(Normalized)', fontweight='bold')
    ax1.set_xlabel('BMI')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Question 2: What's the stroke rate in different BMI bands?
    ax2 = axes[0, 1]
    # Create BMI bins - include ALL data, no filtering
    df_complete = df.dropna(subset=['bmi'])  # Only remove missing BMI values
    bmi_bins = pd.cut(df_complete['bmi'], bins=np.arange(10, 81, 5), right=False)
    stroke_rate = df_complete.groupby(bmi_bins)['stroke'].agg(['mean', 'count']).reset_index()
    stroke_rate['bmi_midpoint'] = stroke_rate['bmi'].apply(lambda x: x.mid)

    # Keep ALL bins, even with small sample sizes
    bars = ax2.bar(stroke_rate['bmi_midpoint'], stroke_rate['mean'] * 100, 
                width=4, alpha=0.8, color='darkred', edgecolor='black')
    ax2.set_title('Stroke Rate by BMI Bands\n(All Data Included)', fontweight='bold')
    ax2.set_xlabel('BMI')
    ax2.set_ylabel('Stroke Rate (%)')
    ax2.grid(True, alpha=0.3)

    # Label bars with sample sizes
    for bar, rate, count in zip(bars, stroke_rate['mean'] * 100, stroke_rate['count']):
        if count >= 5:  # Only label if we have some data
            ax2.text(bar.get_x() + bar.get_width()/2, rate + 0.2, 
                    f'{rate:.1f}%\n(n={count})', ha='center', va='bottom', 
                    fontsize=8)

    # Question 3: Are there outliers or long tails?
    ax3 = axes[1, 0]
    # Box plot to show outliers
    box_data = [no_stroke_data, stroke_data]
    box_plot = ax3.boxplot(box_data, labels=['No Stroke', 'Stroke'], 
                        patch_artist=True, showfliers=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    box_plot['boxes'][1].set_facecolor('coral')

    # Add scatter points for extreme outliers
    ax3.scatter([1]*len(no_stroke_data), no_stroke_data, alpha=0.1, s=8, color='blue')
    ax3.scatter([2]*len(stroke_data), stroke_data, alpha=0.3, s=8, color='red')

    ax3.set_title('Outliers and Distribution Shape', fontweight='bold')
    ax3.set_ylabel('BMI')
    ax3.grid(True, alpha=0.3)

    # Add statistics text
    q99_no_stroke = np.percentile(no_stroke_data, 99)
    q99_stroke = np.percentile(stroke_data, 99)
    ax3.text(0.02, 0.98, f'99th percentiles:\nNo Stroke: {q99_no_stroke:.1f}\nStroke: {q99_stroke:.1f}', 
            transform=ax3.transAxes, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Question 4: Where are the missing values located?
    ax4 = axes[1, 1]
    # Show pattern of missing values by stroke status
    missing_by_stroke = df.groupby('stroke')['bmi'].apply(lambda x: x.isna().sum()).reset_index()
    missing_by_stroke.columns = ['stroke', 'missing_count']
    total_by_stroke = df.groupby('stroke').size().reset_index()
    total_by_stroke.columns = ['stroke', 'total_count']
    missing_summary = missing_by_stroke.merge(total_by_stroke, on='stroke')
    missing_summary['missing_rate'] = missing_summary['missing_count'] / missing_summary['total_count'] * 100

    bars = ax4.bar(['No Stroke', 'Stroke'], missing_summary['missing_rate'], 
                color=['lightblue', 'coral'], alpha=0.8, edgecolor='black')
    ax4.set_title('Missing BMI Values by Group', fontweight='bold')
    ax4.set_ylabel('Missing Rate (%)')
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for bar, rate, count in zip(bars, missing_summary['missing_rate'], missing_summary['missing_count']):
        ax4.text(bar.get_x() + bar.get_width()/2, rate + 0.5, 
                f'{rate:.1f}%\n({count} missing)', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Summary statistics
    print("📊 BMI vs STROKE ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"1. DISTRIBUTION:")
    print(f"   No Stroke - Mean BMI: {no_stroke_data.mean():.1f}")
    print(f"   Stroke    - Mean BMI: {stroke_data.mean():.1f}")
    print(f"   Difference: {stroke_data.mean() - no_stroke_data.mean():.1f}")

    print(f"\n2. STROKE RATES:")
    max_rate_idx = stroke_rate['mean'].idxmax()
    max_rate_range = stroke_rate.loc[max_rate_idx, 'bmi']
    max_rate_value = stroke_rate.loc[max_rate_idx, 'mean'] * 100
    print(f"   Highest stroke rate: {max_rate_value:.1f}% in BMI range {max_rate_range}")

    print(f"\n3. OUTLIERS:")
    print(f"   No Stroke - 99th percentile: {q99_no_stroke:.1f}")
    print(f"   Stroke    - 99th percentile: {q99_stroke:.1f}")
    print(f"   Extreme values (BMI > 50): {len(df[df['bmi'] > 50])} cases")

    print(f"\n4. MISSING VALUES:")
    total_missing = df['bmi'].isna().sum()
    total_cases = len(df)
    print(f"   Total missing: {total_missing} out of {total_cases} ({total_missing/total_cases*100:.1f}%)")
    for idx, row in missing_summary.iterrows():
        group_name = "No Stroke" if row['stroke'] == 0 else "Stroke"
        print(f"   {group_name}: {row['missing_count']} missing ({row['missing_rate']:.1f}%)")

def test():
    pass