import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter



MODEL_PARAMS = {
    "Household Income",
    "Racial Segregation",
    "Cancer_Prevalence",
    "Medicaid Enrollment",
    "Uninsured",
    "Over 64",
}

VARTYPES = {
    "Household Income": "quartile",
    "Racial Segregation": "hard_coded",
    "Cancer_Prevalence": "median",
    "Medicaid Enrollment": "median",
    "Uninsured": "median",
    "Over 64": "median",
    "Met_Colon_Screen": "median" # will be outcome var, above or below median
}

VARGROUPS = {
    "quartile": ["Q1","Q2","Q3","Q4"],
    "hard_coded": ["G1", "G2", "G3", "G4", "G5"],
    "median": ["M1", "M2"]
}

OUTCOME_VAR = "Met_Breast_Screen"

df = pd.read_json("grouped_us_tract.jsonl", lines=True)

# Binary outcome: M2 vs. M1
outcome_col = f"{OUTCOME_VAR}:group_assigned"
df["outcome"] = (df[outcome_col] == "M2").astype(int)

# Select predictors
predictor_cols = [f"{var}:group_assigned" for var in MODEL_PARAMS]
model_data = df[["outcome"] + predictor_cols].dropna().copy()

# One-hot encode categorical predictors.
# Drop the first category of each variable as its reference group.
X = pd.get_dummies(
    model_data[predictor_cols],
    drop_first=True,
    dtype=int
)

X = sm.add_constant(X)
y = model_data["outcome"]

# Fit one multivariable logistic regression
model = sm.Logit(y, X).fit()

# Calculate adjusted odds ratios and 95% confidence intervals
results = pd.DataFrame({
    "Odds_Ratio": np.exp(model.params),
    "CI_Lower": np.exp(model.conf_int()[0]),
    "CI_Upper": np.exp(model.conf_int()[1]),
    "P_Value": model.pvalues,
})

results = results.drop(index="const")

print(results.round(4))

results.to_csv("adjusted_logistic_odds_ratios.csv")

# Forest plot of adjusted odds ratios
plot_rows = []

for var in MODEL_PARAMS:
    var_col = f"{var}:group_assigned"
    groups = VARGROUPS[VARTYPES[var]]

    # First category is the reference group
    reference_group = groups[0]

    plot_rows.append({
        "Label": f"{var} ({reference_group})",
        "Odds_Ratio": 1.0,
        "CI_Lower": 1.0,
        "CI_Upper": 1.0,
        "P_Value": np.nan,
    })

    # Add the non-reference groups in their predefined order
    for group in groups[1:]:
        result_key = f"{var_col}_{group}"

        if result_key in results.index:
            plot_rows.append({
                "Label": f"{var} ({group})",
                "Odds_Ratio": results.loc[result_key, "Odds_Ratio"],
                "CI_Lower": results.loc[result_key, "CI_Lower"],
                "CI_Upper": results.loc[result_key, "CI_Upper"],
                "P_Value": results.loc[result_key, "P_Value"],
            })

plot_df = pd.DataFrame(plot_rows)

# Number rows sequentially: 1, 2, 3, ...
plot_df["Plot_Number"] = range(1, len(plot_df) + 1)

# Print the ordered table
print(plot_df.round(4).to_string(index=False))

# Create forest plot
fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.4)))

y_pos = np.arange(len(plot_df))

for i, row in plot_df.iterrows():
    is_reference = row["CI_Lower"] == row["CI_Upper"] == 1.0

    ax.errorbar(
        row["Odds_Ratio"],
        y_pos[i],
        xerr=[
            [row["Odds_Ratio"] - row["CI_Lower"]],
            [row["CI_Upper"] - row["Odds_Ratio"]],
        ],
        fmt="s" if is_reference else "o",
        color="gray" if is_reference else "C0",
        capsize=3,
    )

# Null effect line
ax.axvline(1, linestyle="--", linewidth=1, color="gray")

# Number each row, followed by its variable and group
ax.set_yticks(y_pos)
ax.set_yticklabels([
    f"({row.Plot_Number}) {row.Label}"
    for row in plot_df.itertuples()
])

# Display first predictor at the top
ax.invert_yaxis()

ax.set_xscale("log")
ax.set_xlabel("Adjusted Odds Ratio (95% CI)")
ax.set_title(f"Adjusted Odds of Achieving Above Median for {OUTCOME_VAR}")
ax.grid(axis="x", alpha=0.25)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_formatter(ScalarFormatter())

plt.tight_layout()
plt.savefig("adjusted_logistic_forest_plot.png", dpi=300)
plt.show()