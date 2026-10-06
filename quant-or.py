import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load data
#df = pd.read_json("met_colo_racial_seg.jsonl", lines=True)
df = pd.read_json("met_colo_racial_seg.jsonl", lines=True)


# -----------------------------
# Define outcome
# -----------------------------

outcome_median = df["outcome"].median()

df["outcome_binary"] = (df["outcome"] >= outcome_median).astype(int)

# -----------------------------
# Define global quintiles
# -----------------------------

df["segregation_q"] = pd.qcut(
    df["segregation"],
    5,
    labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
)

# -----------------------------
# Fit model within each SES
# -----------------------------

ses_groups = sorted(df["ses"].dropna().unique())

fig, axes = plt.subplots(
    1,
    len(ses_groups),
    figsize=(16, 5),
    sharey=True
)

if len(ses_groups) == 1:
    axes = [axes]

for ax, ses in zip(axes, ses_groups):

    subset = df[df["ses"] == ses].copy()

    # Q1 is the reference group
    subset["Q2"] = (subset["segregation_q"] == "Q2").astype(int)
    subset["Q3"] = (subset["segregation_q"] == "Q3").astype(int)
    subset["Q4"] = (subset["segregation_q"] == "Q4").astype(int)
    subset["Q5"] = (subset["segregation_q"] == "Q5").astype(int)

    X = subset[["Q2", "Q3", "Q4", "Q5"]]
    X = sm.add_constant(X)

    y = subset["outcome_binary"]

    try:
        model = sm.Logit(y, X).fit(disp=False)
    except (np.linalg.LinAlgError, ValueError):
        print(f"Skipping {ses}: model could not be fit")
        continue

    # Q1 reference
    odds_ratios = [1.0]
    ci_lower = [1.0]
    ci_upper = [1.0]

    for q in ["Q2", "Q3", "Q4", "Q5"]:

        beta = model.params[q]
        conf = model.conf_int().loc[q]

        odds_ratios.append(np.exp(beta))
        ci_lower.append(np.exp(conf[0]))
        ci_upper.append(np.exp(conf[1]))

    # Plot
    x = np.arange(1, 6)

    ax.errorbar(
        x,
        odds_ratios,
        yerr=[
            np.array(odds_ratios) - np.array(ci_lower),
            np.array(ci_upper) - np.array(odds_ratios)
        ],
        fmt="o",
        capsize=5
    )

    ax.axhline(1, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])

    ax.set_xlabel("Segregation Quintile")
    income_min = df[df["ses"] == ses]["income"].min()
    income_max = df[df["ses"] == ses]["income"].max()

    ax.set_title(
        f"{ses} (${income_min:,.0f} to ${income_max:,.0f})"
    )

    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Odds Ratio")

plt.suptitle(
    f"Odds of Outcome ≥ Median ({outcome_median:.3f}) by Segregation Quintile"
)

print("\nSES income ranges:")
print(
    df.groupby("ses")["income"]
      .agg(["min", "max"])
      .sort_index()
      .to_string()
)

segregation_quantiles = df["segregation"].quantile(
    [0.2, 0.4, 0.6, 0.8]
)

print("\nSegregation quintile cutoffs:")
print(segregation_quantiles.to_string())





plt.tight_layout()
plt.show()