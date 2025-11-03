import pandas as pd
import seaborn as sns
from rich.traceback import install
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt 
from eda_theme import (
    set_theme,
    create_fig,
    add_n_annotation,
    save_fig,
    STROKE_COLOR,
    NO_STROKE_COLOR,
    MISSING_COLOR,
    NEUTRAL_COLOR,
)

set_theme()
install()


def plot_target_distribution(df, target_col="stroke", output_dir="reports/eda"):
    """
    Plot the distribution of the target variable including nulls.
    Displays 'Stroke' / 'No Stroke' instead of numeric codes.
    Returns paths of saved figures.
    """

    # Map 0 → "No Stroke", 1 → "Stroke"
    label_map = {0: "No Stroke", 1: "Stroke"}
    df_display = df.copy()
    df_display[target_col] = df_display[target_col].map(label_map)

    # Fill nulls explicitly as "Missing"
    counts = (
        df_display[target_col]
        .fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis(target_col)
        .reset_index(name="count")
    )

    total = counts["count"].sum()
    counts["percent"] = (counts["count"] / total * 100).round(2)

    # Color mapping
    color_map = {
        "No Stroke": NO_STROKE_COLOR,
        "Stroke": STROKE_COLOR,
        "Missing": MISSING_COLOR,
    }
    colors = [color_map.get(x, "#cccccc") for x in counts[target_col]]

    fig, ax = create_fig(figsize=(10, 10))
    wedges, texts = ax.pie(
        counts["count"],
        labels=None,                      # no labels on slices
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )

    # Title and legend
    ax.set_title("Stroke Outcome Distribution", fontsize=18, pad=20)
    legend_labels = [
        f"{k} [{v:.1f}% (N = {c})]"
        for k, v, c in zip(
            counts[target_col], counts["percent"], counts["count"]
        )
    ]
    ax.legend(legend_labels, prop={'weight': 'bold'})

    save_fig(fig, "target_piechart", output_dir)

def dataset_overview_tables(df, output_dir="reports/eda"):
    """Generate summary tables for missing values and column-type counts."""

    # ----- Missing-value table -----
    missing = (
        df.isna().sum()
        .reset_index()
        .rename(columns={"index": "Column", 0: "MissingCount"})
    )
    missing["MissingPercent"] = (missing["MissingCount"] / len(df) * 100).round(2)
    missing = missing.sort_values(by='MissingCount', ascending=False)
    breakpoint()
    missing_latex = missing.to_latex(index=False, float_format="%.2f")

    # ----- Numeric vs Categorical counts -----
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(exclude=["number"]).columns
    counts = pd.DataFrame(
        {"Type": ["Numeric", "Categorical"],
         "Count": [len(num_cols), len(cat_cols)]}
    )
    counts_latex = counts.to_latex(index=False)

    with open(f"{output_dir}/missing_values_table.tex", "w") as f:
        f.write(missing_latex)
    with open(f"{output_dir}/column_type_counts_table.tex", "w") as f:
        f.write(counts_latex)

    return missing, counts

def hypothesis_table(output_dir="reports/eda"):
    """Create and export a hypothesis table in LaTeX format."""

    data = [
        ("age", "Numeric", "Older age increases stroke risk"),
        ("avg_glucose_level", "Numeric", "Higher glucose increases stroke risk"),
        ("bmi", "Numeric", "Higher BMI increases stroke risk"),
        ("smoking_status", "Categorical", "Current smokers more likely to have stroke"),
        ("hypertension", "Categorical", "Hypertension increases stroke risk"),
        ("heart_disease", "Categorical", "Heart disease increases stroke risk"),
    ]

    df = pd.DataFrame(data, columns=["Variable", "Type", "Expected Relationship"])
    latex_str = df.to_latex(index=False, escape=False)

    with open(f"{output_dir}/hypothesis_table.tex", "w") as f:
        f.write(latex_str)

    return df


def plot_numeric_distribution(df, column, output_dir="reports/eda"):
    """
    Plot histogram for a numeric column with mean (solid) and median (dotted) lines.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")

    data = df[column].dropna()

    fig, ax = create_fig(figsize=(8, 6))
    sns.histplot(data, bins=30, color=NEUTRAL_COLOR, alpha=0.6, ax=ax)

    mean_val = data.mean()
    median_val = data.median()

    ax.axvline(mean_val, color=STROKE_COLOR, linestyle="-", linewidth=1.2, label=f"Mean = {mean_val:.1f}")
    ax.axvline(median_val, color='black', linestyle="--", linewidth=1.2, label=f"Median = {median_val:.1f}")

    if column == 'bmi':
        ax.set_title("BMI Distribution")
    else:
        ax.set_title(f"{column.replace('_', ' ').title()} Distribution")
    
    ax.set_xlabel(column.replace('_', ' ').title())
    ax.set_ylabel("Count")

    ax.legend()

    save_fig(fig, f"{column}_distribution", output_dir)
    return mean_val, median_val


def plot_numeric_vs_target(df, column, target="stroke", output_dir="reports/eda"):
    """
    Plot relationship of a numeric variable with the binary target using a strip plot.
    Red for stroke, green for no-stroke. Adds a global median line.
    """
    if column not in df.columns or target not in df.columns:
        raise ValueError("Column or target not found in dataframe.")

    label_map = {0: "No Stroke", 1: "Stroke"}
    data = df[[column, target]].dropna()
    data[target] = data[target].map(label_map)
    median_val = data[column].median()

    fig, ax = create_fig(figsize=(8, 6))
    sns.stripplot(
        data=data,
        x=column,
        y=target,
        jitter=0.25,
        alpha=0.5,
        size=3.0,
        palette={'No Stroke': NO_STROKE_COLOR, 'Stroke': STROKE_COLOR},
        ax=ax,
        orient='h'
    )

    # median reference line
    ax.axvline(median_val, color="black", linestyle="--", linewidth=1.2,
               label=f"Median = {median_val:.1f}", zorder=5)

    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel(target.replace("_", " ").title())
    ax.set_title(f"{column.replace('_', ' ').title()} vs {target.title()}")
    ax.legend()

    save_fig(fig, f"{column}_vs_{target}_stripplot", output_dir)
    return median_val


def plot_bmi_missing_trend(df, output_dir="reports/eda"):
    """
    Plot BMI missing percentage across age groups with improved binning and annotation.
    """

    if "age" not in df.columns or "bmi" not in df.columns:
        raise ValueError("DataFrame must contain 'age' and 'bmi' columns.")

    # Refined bins: narrower mid-range and drop last group (no data)
    bins = [0, 30, 45, 60, 75, 90]
    labels = ["<30", "30-44", "45-59", "60-74", "75-89"]

    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    df["bmi_missing"] = df["bmi"].isna()

    # Calculate missing percentage
    grouped = (
        df.groupby("age_group")["bmi_missing"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"bmi_missing": "Missing %"})
    )

    # Drop age groups with 0 members
    grouped = grouped[grouped["Missing %"].notna()]

    # --- Plot ---
    fig, ax = create_fig(figsize=(7, 4))
    sns.lineplot(
        data=grouped,
        x="age_group",
        y="Missing %",
        color=STROKE_COLOR,
        marker="o",
        linewidth=3,
        markersize=8,
        ax=ax,
    )

    # Annotate points — bold, offset to avoid overlap
    for i, row in grouped.iterrows():
        ax.text(
            i,
            row["Missing %"] + 0.4,  # vertical offset
            f"{row['Missing %']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    ax.set_title("Increase in BMI Missingness with Age", pad=12)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("BMI Missing (%)")
    sns.despine(ax=ax)

    save_fig(fig, "bmi_missing_trend_refined", output_dir)
    return grouped

def plot_bmi_missing_trend_quantile(df, q=5, output_dir="reports/eda"):
    """
    Plot BMI missing percentage across quantile-based (equal-frequency) age groups.
    Each age bin contains ~equal number of observations, ensuring stable percentages.
    """

    if "age" not in df.columns or "bmi" not in df.columns:
        raise ValueError("DataFrame must contain 'age' and 'bmi' columns.")

    df = df.copy()
    df["bmi_missing"] = df["bmi"].isna()

    # --- Quantile binning by age ---
    df["age_group"] = pd.qcut(df["age"], q=q, duplicates="drop")
    # Compute median age in each quantile for annotation
    bin_medians = df.groupby("age_group")["age"].median().reset_index(name="Median Age")

    # --- Compute missing percentage per age quantile ---
    grouped = (
        df.groupby("age_group")["bmi_missing"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"bmi_missing": "Missing %"})
    )
    grouped = grouped.merge(bin_medians, on="age_group")

    # --- Plot ---
    fig, ax = create_fig(figsize=(7, 4))
    sns.lineplot(
        data=grouped,
        x="Median Age",
        y="Missing %",
        color=STROKE_COLOR,
        marker="o",
        linewidth=3,
        markersize=8,
        ax=ax,
    )

    # Annotate points
    for _, row in grouped.iterrows():
        ax.text(
            row["Median Age"],
            row["Missing %"] + 0.4,
            f"{row['Missing %']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    ax.set_title("BMI Missingness Across Age Quantiles", pad=12)
    ax.set_xlabel("Median Age of Quantile Group")
    ax.set_ylabel("BMI Missing (%)")
    sns.despine(ax=ax)

    save_fig(fig, "bmi_missing_trend_quantile", output_dir)
    return grouped

def plot_bmi_missing_by_target(df, target="stroke", output_dir="reports/eda"):
    """
    Plot BMI missing percentage by target (stroke vs no-stroke).
    Adds 95% confidence intervals for stability.
    """
    if "bmi" not in df.columns or target not in df.columns:
        raise ValueError("DataFrame must contain 'bmi' and target columns.")

    df = df.copy()
    df["bmi_missing"] = df["bmi"].isna()
    label_map = {0: "No Stroke", 1: "Stroke"}
    df[target] = df[target].map(label_map)

    # Compute missing percentage by target
    grouped = (
        df.groupby(target)["bmi_missing"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "Missing Count", "count": "Total"})
    )
    grouped["Missing %"] = grouped["Missing Count"] / grouped["Total"] * 100

    # 95% confidence intervals for the proportion
    ci_low, ci_high = proportion_confint(grouped["Missing Count"], grouped["Total"], method="wilson")
    grouped["CI Low"] = ci_low * 100
    grouped["CI High"] = ci_high * 100

    # Plot
    fig, ax = create_fig(figsize=(5, 4))
    palette = {'No Stroke': NO_STROKE_COLOR, 'Stroke': STROKE_COLOR}

    sns.barplot(
        data=grouped,
        x=target,
        y="Missing %",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.6,
        errorbar=None,  # we'll plot our own CI
    )

    # Draw confidence intervals manually
    ax.errorbar(
        grouped[target],
        grouped["Missing %"],
        yerr=[grouped["Missing %"] - grouped["CI Low"], grouped["CI High"] - grouped["Missing %"]],
        fmt="none",
        ecolor="black",
        capsize=4,
        elinewidth=1.2,
    )

    for i, row in grouped.iterrows():
        ax.text(
            i,
            row["Missing %"] / 2.5,
            f"{row['Missing %']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color='white'
        )

    ax.set_title("BMI Missingness by Stroke Outcome")
    ax.set_xlabel(target.replace("_", " ").title())
    ax.set_ylabel("BMI Missing (%)")
    sns.despine(ax=ax)

    save_fig(fig, "bmi_missing_by_target", output_dir)
    return grouped

def plot_categorical_vs_target(df, column, target="stroke", output_dir="reports/eda"):
    """
    Side-by-side (dogged) bar plot of a categorical variable vs. binary target.
    Annotates each bar with count and percentage of total.
    """
    if column not in df.columns or target not in df.columns:
        raise ValueError(f"'{column}' or '{target}' not found in dataframe.")

    data = df[[column, target]].dropna()
    label_map = {0: 'No Stroke', 1: 'Stroke'}
    data[target] = data[target].map(label_map)

    # Compute counts and percentages
    grouped = (
        data.groupby([column, target])
        .size()
        .reset_index(name="Count")
    )
    total = len(data)
    grouped["Percent"] = grouped["Count"] / total * 100

    # Plot
    fig, ax = create_fig(figsize=(9, 7))
    palette = {'No Stroke': NO_STROKE_COLOR, 'Stroke': STROKE_COLOR}

    sns.barplot(
        data=grouped,
        x=column,
        y="Count",
        hue=target,
        palette=palette,
        ax=ax,
    )

    # Annotate each bar with count and % of total
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2,
                height + total * 0.002,
                f"{int(height)}\n({height / total * 100:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="black",
            )

    ax.set_title(f"{column.replace('_', ' ').title()} vs {target.title()}", pad=12)
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Count")
    ax.legend(
        title=target.replace("_", " ").title(),
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
    )
    sns.despine(ax=ax)

    save_fig(fig, f"{column}_vs_{target}_dogged", output_dir)
    return grouped


def binary_summary_table_latex_colored(df, binary_vars, target="stroke", output_dir="reports/eda"):
    """
    Export a binary summary table to LaTeX with automatic row colors per variable.
    """
    rows = []
    for col in binary_vars:
        if col not in df.columns:
            continue
        temp = (
            df.groupby(col)[target]
            .agg(["count", "sum"])
            .rename(columns={"count": "Total", "sum": "Stroke Cases"})
        )
        temp["Stroke Rate (%)"] = (temp["Stroke Cases"] / temp["Total"] * 100).round(1)
        temp["Variable"] = col
        temp = temp.reset_index().rename(columns={col: "Category"})
        rows.append(temp)

    result = pd.concat(rows, ignore_index=True)
    result = result[["Variable", "Category", "Total", "Stroke Cases", "Stroke Rate (%)"]]
    label_map = {0: "No", 1: "Yes"}
    result["Category"] = result["Category"].apply(lambda x: label_map.get(x, x))
    
    
    # create a light pastel palette (Set2)
    unique_vars = result["Variable"].unique()

    # build LaTeX with row colors
    latex_lines = []
    latex_lines.append("\\begin{tabular}{l l r r r}")
    latex_lines.append("\\toprule")
    latex_lines.append("Variable & Category & Total & Stroke Cases & Stroke Rate (\\%) \\\\")
    latex_lines.append("\\midrule")
    

    for i, var in enumerate(unique_vars):
        subset = result[result["Variable"] == var]
        
        # Add multirow for the variable name
        num_rows = len(subset)
        latex_lines.append("\\addlinespace")
        for j, (_, row) in enumerate(subset.iterrows()):
            if j == 0:
                # First row: include variable name with multirow
                line = f"\\multirow{{{num_rows}}}{{*}}{{{row['Variable'].replace('_', '\\_')}}} & {row['Category']} & {int(row['Total'])} & {int(row['Stroke Cases'])} & {row['Stroke Rate (%)']:.1f} \\\\"
            else:
                # Subsequent rows: empty first column
                line = f"& {row['Category']} & {int(row['Total'])} & {int(row['Stroke Cases'])} & {row['Stroke Rate (%)']:.1f} \\\\"
            latex_lines.append(line)
            
        
        # Add hline after each variable group (except the last one)
        if i < len(unique_vars) - 1:
            latex_lines.append("\\addlinespace")
            latex_lines.append("\\hline")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")

    latex_output = "\n".join(latex_lines)
    with open(f"{output_dir}/binary_summary_colored.tex", "w") as f:
        f.write(latex_output)

    return latex_output


def plot_correlation_heatmap(df, target="stroke", output_dir="reports/eda"):
    """
    Compute numeric correlation matrix including target variable and plot heatmap.
    Uses diverging RdYlBu_r palette with annotated coefficients.
    """
    # keep only numeric columns
    corr_df = df.select_dtypes(include="number").copy()
    breakpoint()

    # ensure target is last column
    if target in corr_df.columns:
        cols = [c for c in corr_df.columns if c != target] + [target]
        corr_df = corr_df[cols]

    # compute correlation matrix
    corr_matrix = corr_df.corr(numeric_only=True)

    # --- Plot ---
    fig, ax = create_fig(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        cmap="RdYlBu_r",
        annot=True,
        fmt=".2f",
        linewidths=0.6,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        ax=ax,
        square=True,
        center=0,
    )

    ax.set_title("Correlation Heatmap (Including Stroke Target)", pad=12)
    fig.tight_layout()
    save_fig(fig, "correlation_heatmap", output_dir)

    return corr_matrix

def plot_scatter_by_target(df, x, y, target="stroke", output_dir="reports/eda", loc='upper right'):

    required = {x, y, target}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    data = df[list(required)].dropna()
    data[target] = data[target].map({0: 'No Stroke', 1: 'Stroke'})

    fig, ax = create_fig(figsize=(7, 5))
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=target,
        palette={'No Stroke': NO_STROKE_COLOR, 'Stroke': STROKE_COLOR},
        alpha=0.45,
        s=12,               # smaller for a more jittered appearance
        linewidth=0,        # no border around points
        ax=ax,
    )

    ax.set_title(f"{x.replace('_', ' ').title()} vs {y.replace('_', ' ').title()} by {target.title()}", pad=10)
    ax.set_xlabel(x.replace('_', ' ').title())
    ax.set_ylabel(y.replace('_', ' ').title())
    ax.legend(title=target.replace('_', ' ').title(), frameon=True, facecolor="white", edgecolor="black", loc=loc)
    sns.despine(ax=ax)

    save_fig(fig, f"{x}_vs_{y}_scatter", output_dir)
    return ax

def plot_stroke_rate_heatmap(df, output_dir="reports/eda"):
    """
    Heatmap showing stroke rate (%) by age group and smoking status.
    Uses same age binning as before.
    """
    required = {"age", "smoking_status", "stroke"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    df = df.copy()

    # --- Use the same age bins you used before ---
    bins = [0, 30, 45, 60, 75, 90]
    labels = ["<30", "30-44", "45-59", "60-74", "75-89"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    # compute stroke rate (%)
    pivot = (
        df.groupby(["age_group", "smoking_status"])["stroke"]
        .mean()
        .mul(100)
        .unstack(fill_value=0)
    )

    # --- Plot ---
    fig, ax = create_fig(figsize=(8, 6))
    sns.heatmap(
        pivot,
        cmap="RdYlBu_r",
        annot=True,
        fmt=".1f",
        linewidths=0.6,
        cbar_kws={"label": "Stroke Rate (%)"},
        ax=ax,
        annot_kws={'fontweight': 'bold', 'fontsize': 12}
    )

    ax.set_title("Stroke Rate (%) by Age Group and Smoking Status", pad=12)
    ax.set_xlabel("Smoking Status", labelpad=12)
    ax.set_ylabel("Age Group", labelpad=12)
    fig.tight_layout()
    save_fig(fig, "stroke_rate_by_age_smoking", output_dir)

    return pivot


if __name__ == "__main__":
    df = pd.read_csv('./data/stroke_data.csv').drop(columns='id')
    # plot_target_distribution(df)
    # dataset_overview_tables(df)
    # df_hypothesis = hypothesis_table()
    # for col in ['age', 'avg_glucose_level', 'bmi']:
    #     # plot_numeric_distribution(df, col)
    #     plot_numeric_vs_target(df, col)
    # plot_bmi_missing_trend(df)
    # plot_bmi_missing_trend_quantile(df)
    # plot_bmi_missing_by_target(df)
    binary_cols = ['hypertension', 'heart_disease', 
                   'ever_married', 'Residence_type']
    # cols = ['gender', 'work_type', 'smoking_status']
    # for col in cols:
    #     plot_categorical_vs_target(df, col)
    # binary_summary_table_latex_colored(df, binary_cols)
    # plot_correlation_heatmap(df[['age', 'bmi', 'avg_glucose_level', 'stroke']])
    # plot_pairwise_numeric(df)
    # plot_stroke_rate_heatmap(df)
    plot_scatter_by_target(df, "age", "avg_glucose_level", loc='upper left')
    plot_scatter_by_target(df, "bmi", "avg_glucose_level", loc='upper right')