import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

main_ses_var = "Household Income"
main_seg_var = "Racial Segregation"

plot_cols = [
    "Met_Colon_Screen",
]


df = pd.read_json(
    "us-data/us_data.jsonl",
    lines=True,
    dtype={"GEOID": str}
)

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

nan_counts = df.isna().sum()
nan_counts = nan_counts[nan_counts > 0]

if len(nan_counts):
    print(nan_counts.to_string())
else:
    print("No NaN values.")

# Isolate only FIPS State ID 36 (New York), NJ is 34, VA 51, FL 12, AL 01 
df = df[df["GEOID"].astype(str).str.startswith("34")].copy()

# -----------------------------
# Global SES quartiles
# -----------------------------
SES_INCOME_GROUPS = ["SESQ1", "SESQ2", "SESQ3", "SESQ4"]
df["ses_q"] = pd.qcut(
    df[main_ses_var],
    4,
    labels=SES_INCOME_GROUPS,
    duplicates="drop"
)

# -----------------------------
# Global segregation quartiles
# -----------------------------

MODE = 'hard-coded'

SEG_GROUPS = ['A','B','C','D','E','F']
N_SEG = len(SEG_GROUPS)
SEG_GROUPS_POP_FIRST =  SEG_GROUPS[1:]

SEG_CUTOFFS = [-0.5,-0.25,0,0.25,0.5]

if MODE == "hard-coded":
    df["segregation_q"] = pd.cut(
        df[main_seg_var],
        bins=[-np.inf, *SEG_CUTOFFS, np.inf],
        labels=SEG_GROUPS,
        right=False
    )
    cutoffs = SEG_CUTOFFS

else:
    df["segregation_q"] = pd.qcut(
        df[main_seg_var],
        N_SEG,
        labels=SEG_GROUPS,
        duplicates="drop"
    )
    cutoffs = df[main_seg_var].quantile(
        np.arange(1, N_SEG) / N_SEG
    ).tolist()


X_AXIS_LABELS = (
    [f"< {cutoffs[0]:.2f}"] +
    [
        f"[{cutoffs[i-1]:.2f}, {cutoffs[i]:.2f})"
        for i in range(1, len(cutoffs))
    ] +
    [f">= {cutoffs[-1]:.2f}"]
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
# SEG_GROUPS -> quantiles
seg_quantiles = df[main_seg_var].quantile(
    np.arange(1, N_SEG) / N_SEG
)
print(seg_quantiles.to_string())

print("\nSegregation cutoffs:")
print(SEG_CUTOFFS)

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

    ses_groups = SES_INCOME_GROUPS

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
        # Q1 = reference across SEG_GROUPS
        # -----------------------------

        # Q1 = reference across SEG_GROUPS
        for seg in SEG_GROUPS[1:]:
            subset[seg] = (
                subset["segregation_q"] == seg
            ).astype(int)

        X = subset[SEG_GROUPS[1:]]

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

        for q in SEG_GROUPS[1:]: # model against SEG_GROUPS

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

        x = np.arange(1, (N_SEG+1)) # plot X Axis of SEG_GROUPS

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
            .reindex(SEG_GROUPS)
        )

        # -----------------------------
        # Plot
        # -----------------------------

        x = np.arange(1, (N_SEG+1))

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
            SEG_GROUPS,
            fontsize=9
        )

        # N below each segregation quartile
        for xi, q in zip(x, SEG_GROUPS):
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
            X_AXIS_LABELS,
            fontsize=8
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
            SEG_GROUPS,
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