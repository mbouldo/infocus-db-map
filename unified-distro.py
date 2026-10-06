import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -----------------------------
# Load unified dataset
# -----------------------------

df = pd.read_json("unified_data.jsonl", lines=True)

# {"GEOID":"36047060600","met_colorectal":"0.719","racial-segregation":"0.666299154722528","household-income":"94722","breast_cancer_screening":"0.842","currently_smoke":"0.077","obese_over_30BMI":"0.202","history_of_cancer_dx":"0.083","report_fair_or_poor_overall_health":"0.102","dx_with_diabetes":"0.08","high_cholesterol":"0.366","dental_last_year":"0.719"}
# {"GEOID":"36047005602","met_colorectal":"0.698","racial-segregation":"0.703468208092486","household-income":"97750","breast_cancer_screening":"0.806","currently_smoke":"0.102","obese_over_30BMI":"0.224","history_of_cancer_dx":"0.076","report_fair_or_poor_overall_health":"0.118","dx_with_diabetes":"0.078","high_cholesterol":"0.349","dental_last_year":"0.7"}

# -----------------------------
# Data typing
# -----------------------------

# GEOID is an identifier and remains a string
for col in df.columns:
    if col != "GEOID":
        original = df[col].copy()

        df[col] = pd.to_numeric(df[col], errors="coerce")

        coerced = original.notna() & df[col].isna()

        if coerced.any():
            print(
                f"\nWARNING: {col}: "
                f"{coerced.sum()} value(s) coerced to NaN"
            )
            print(original[coerced].to_string())

print("\nNaN counts after data typing:")

nan_counts = df.isna().sum()
nan_counts = nan_counts[nan_counts > 0]

if len(nan_counts):
    print(nan_counts.to_string())
else:
    print("No NaN values.")


# -----------------------------
# Distribution mosaic
# -----------------------------

main_ses_var = "household-income"
main_seg_var = "racial-segregation"

#plot_cols = [col for col in df.columns if col != "GEOID"]
plot_cols = [
    "racial-segregation",
    "household-income",
    "breast_cancer_screening",
    "met_colorectal",
    "currently_smoke",
    "obese_over_30BMI",
    "history_of_cancer_dx",
    "report_fair_or_poor_overall_health",
    "dx_with_diabetes",
    "high_cholesterol",
    ]




n_plots = len(plot_cols)
n_cols = 3
n_rows = int(np.ceil(n_plots / n_cols))

# Calculate histograms first so we can find the global maximum frequency
histograms = {}

global_max_frequency = 0

for col in plot_cols:
    values = df[col].dropna()

    if values.empty:
        continue

    counts, bin_edges = np.histogram(values, bins=20)

    histograms[col] = (values, counts, bin_edges)

    global_max_frequency = max(global_max_frequency, counts.max())


fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(15, 4 * n_rows)
)

axes = np.atleast_1d(axes).flatten()

for ax, col in zip(axes, plot_cols):

    if col not in histograms:
        ax.set_visible(False)
        continue

    values, counts, bin_edges = histograms[col]

    ax.hist(
        values,
        bins=bin_edges,
        edgecolor="black"
    )

    # Same y-axis across every subplot
    ax.set_ylim(0, global_max_frequency * 1.10)

    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

    # Small statistics box
    stats_text = (
        f"n = {len(values):,}\n"
        f"max = {values.max():,.3g}\n"
        f"median = {values.median():,.3g}"
    )

    ax.text(
        0.97,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.8
        )
    )

# Hide unused subplot spaces
for ax in axes[n_plots:]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()