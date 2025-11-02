import pandas as pd
import seaborn as sns
from rich.traceback import install
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
    Plot the percentage of missing BMI values across age groups.
    Focuses only on missingness trend, not total counts.
    """

    if "age" not in df.columns or "bmi" not in df.columns:
        raise ValueError("DataFrame must contain 'age' and 'bmi' columns.")

    # Define age bins and labels
    bins = [0, 30, 50, 65, 80, 120]
    labels = ["<30", "30-49", "50-64", "65-79", "80+"]
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    df["bmi_missing"] = df["bmi"].isna()

    # Calculate missing percentage per group
    grouped = (
        df.groupby("age_group")["bmi_missing"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"bmi_missing": "Missing %"})
    )

    # Plot line showing increase in missingness
    fig, ax = create_fig(figsize=(8, 5))
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

    # Annotate points
    for i, row in grouped.iterrows():
        ax.text(
            i,
            row["Missing %"] + 0.5,
            f"{row['Missing %']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

    ax.set_title("Increase in BMI Missingness with Age")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("BMI Missing (%)")
    sns.despine(ax=ax)

    save_fig(fig, "bmi_missing_trend", output_dir)
    return grouped
if __name__ == "__main__":
    df = pd.read_csv('./data/stroke_data.csv').drop(columns='id')
    # plot_target_distribution(df)
    # dataset_overview_tables(df)
    # df_hypothesis = hypothesis_table()
    # for col in ['age', 'avg_glucose_level', 'bmi']:
    #     # plot_numeric_distribution(df, col)
    #     plot_numeric_vs_target(df, col)
    plot_bmi_missing_trend(df)
