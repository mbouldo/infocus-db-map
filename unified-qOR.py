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

for col in df.columns:
    if col != "GEOID":
        original = df[col].copy()

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        coerced = original.notna() & df[col].isna()

        if coerced.any():
            print(
                f"\nWARNING: {col}: "
                f"{coerced.sum()} value(s) coerced to NaN"
            )
            


print("\nNaN counts after data typing:")

nan_counts = df.isna().sum()
nan_counts = nan_counts[nan_counts > 0]

if len(nan_counts):
    print(nan_counts.to_string())
else:
    print("No NaN values.")


# -----------------------------
# Define variables
# -----------------------------

main_ses_var = "household-income"
main_seg_var = "racial-segregation"

plot_cols = [
    "breast_cancer_screening",
    "met_colorectal",
    "currently_smoke",
    "obese_over_30BMI",
    "history_of_cancer_dx",
    "report_fair_or_poor_overall_health",
    "dx_with_diabetes",
    "high_cholesterol",
    "dental_last_year"
]


# -----------------------------
# Global SES quartiles
# -----------------------------

df["ses_q"] = pd.qcut(
    df[main_ses_var],
    4,
    labels=["Q1", "Q2", "Q3", "Q4"],
    duplicates="drop"
)


# -----------------------------
# Global segregation quartiles
# -----------------------------

df["segregation_q"] = pd.qcut(
    df[main_seg_var],
    4,
    labels=["Q1", "Q2", "Q3", "Q4"],
    duplicates="drop"
)


# -----------------------------
# Print cutoffs
# -----------------------------

print("\nSES quartile cutoffs:")

ses_quantiles = df[main_ses_var].quantile(
    [0.25, 0.50, 0.75]
)

print(ses_quantiles.to_string())


print("\nSegregation quartile cutoffs:")

seg_quantiles = df[main_seg_var].quantile(
    [0.25, 0.50, 0.75]
)

print(seg_quantiles.to_string())


# -----------------------------
# Model each outcome
# -----------------------------

for outcome in plot_cols:

    print("\n" + "=" * 70)
    print(f"OUTCOME: {outcome}")
    print("=" * 70)

    # Outcome-specific median
    outcome_median = df[outcome].median()

    df["outcome_binary"] = (
        df[outcome] >= outcome_median
    ).astype(int)

    # -----------------------------
    # Create 4 SES plots
    # -----------------------------

    ses_groups = [
        "Q1",
        "Q2",
        "Q3",
        "Q4"
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(12, 3.8),
        sharey=True,
        constrained_layout=True
    )

    # -----------------------------
    # Fit model within each SES
    # -----------------------------

    for ax, ses in zip(axes, ses_groups):

        subset = df[
            df["ses_q"] == ses
        ].copy()

        subset = subset.dropna(
            subset=[
                outcome,
                main_ses_var,
                main_seg_var,
                "segregation_q"
            ]
        )

        # -----------------------------
        # Q1 = reference
        # -----------------------------

        subset["Q2"] = (
            subset["segregation_q"] == "Q2"
        ).astype(int)

        subset["Q3"] = (
            subset["segregation_q"] == "Q3"
        ).astype(int)

        subset["Q4"] = (
            subset["segregation_q"] == "Q4"
        ).astype(int)

        X = subset[
            ["Q2", "Q3", "Q4"]
        ]

        X = sm.add_constant(X)

        y = subset["outcome_binary"]

        # -----------------------------
        # Fit logistic model
        # -----------------------------

        try:

            model = sm.Logit(
                y,
                X
            ).fit(
                disp=False
            )

        except (
            np.linalg.LinAlgError,
            ValueError
        ):

            print(
                f"Skipping {ses}: "
                f"model could not be fit"
            )

            continue

        # -----------------------------
        # Extract OR + 95% CI
        # -----------------------------

        odds_ratios = [1.0]
        ci_lower = [1.0]
        ci_upper = [1.0]

        for q in ["Q2", "Q3", "Q4"]:

            beta = model.params[q]

            conf = model.conf_int().loc[q]

            odds_ratios.append(
                np.exp(beta)
            )

            ci_lower.append(
                np.exp(conf[0])
            )

            ci_upper.append(
                np.exp(conf[1])
            )

        # -----------------------------
        # Plot
        # -----------------------------

        x = np.arange(1, 5)

        ax.errorbar(
            x,
            odds_ratios,
            yerr=[
                np.array(odds_ratios)
                - np.array(ci_lower),

                np.array(ci_upper)
                - np.array(odds_ratios)
            ],
            fmt="o",
            markersize=4.5,
            linewidth=1.2,
            elinewidth=1.0,
            capsize=3,
            capthick=1.0
        )
        # N for each segregation group
        # -----------------------------

        group_n = (
            subset
            .groupby("segregation_q", observed=False)
            .size()
            .reindex(["Q1", "Q2", "Q3", "Q4"])
        )

        # -----------------------------
        # Plot
        # -----------------------------

        x = np.arange(1, 5)

        ax.errorbar(
            x,
            odds_ratios,
            yerr=[
                np.array(odds_ratios)
                - np.array(ci_lower),

                np.array(ci_upper)
                - np.array(odds_ratios)
            ],
            fmt="o",
            markersize=4.5,
            linewidth=1.2,
            elinewidth=1.0,
            capsize=3,
            capthick=1.0
        )

        ax.axhline(
            1,
            linestyle="--",
            linewidth=0.9,
            alpha=0.7
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            ["Q1", "Q2", "Q3", "Q4"],
            fontsize=9
        )

        # N below each segregation quartile
        for xi, q in zip(x, ["Q1", "Q2", "Q3", "Q4"]):
            ax.annotate(
                f"n={group_n[q]:,}",
                xy=(xi, 0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -16),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7.5,
                color="dimgray"
            )
        ax.axhline(
            1,
            linestyle="--",
            linewidth=0.9,
            alpha=0.7
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            ["Q1", "Q2", "Q3", "Q4"],
            fontsize=9
        )

        # Only bottom axis label, rather than repeating it
        if ax is axes[1]:
            ax.set_xlabel(
                "",
                fontsize=10
            )

        # SES income range
        income_min = subset[
            main_ses_var
        ].min()

        income_max = subset[
            main_ses_var
        ].max()

        ax.set_title(
            f"{ses}  (${income_min:,.0f}–${income_max:,.0f})",
            fontsize=10,
            fontweight="bold",
            pad=6
        )

        # Log scale
        ax.set_yscale("log")

        # Cleaner grid
        ax.grid(
            axis="y",
            which="major",
            linewidth=0.6,
            alpha=0.25
        )

        ax.grid(
            axis="y",
            which="minor",
            visible=False
        )

        # Cleaner borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Smaller tick labels
        ax.tick_params(
            axis="both",
            labelsize=9,
            length=3
        )

        # Print model information
        print(
            f"\n{ses}: "
            f"N = {len(subset):,}"
        )

        print(
            "  Q1 reference: OR = 1.000"
        )

        for q, or_value, low, high in zip(
            ["Q1", "Q2", "Q3", "Q4"],
            odds_ratios,
            ci_lower,
            ci_upper
        ):
            print(
                f"  {q}: "
                f"OR = {or_value:.4f} "
                f"(95% CI "
                f"{low:.4f}-{high:.4f})"
            )

    # -----------------------------
    # Figure formatting
    # -----------------------------
  # -----------------------------
      
    axes[0].set_ylabel(
        "Odds Ratio",
        fontsize=10
    )

    fig.suptitle(
        f"Odds of {outcome} ≥ Median ({outcome_median:.3f})",
        fontsize=12,
        fontweight="bold"
    )

    plt.show()