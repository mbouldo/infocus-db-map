import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_json("met_colo_racial_seg.jsonl", lines=True)

# --------------------------------------------------
# 1. Create binary outcome
# --------------------------------------------------

outcome_median = df["outcome"].median()

df["MET"] = (df["outcome"] >= outcome_median).astype(int)

# --------------------------------------------------
# 2. Create absolute segregation
# --------------------------------------------------

df["abs_segregation"] = df["segregation"].abs()

# --------------------------------------------------
# 3. Global segregation quartiles
# --------------------------------------------------

q1 = df["abs_segregation"].quantile(0.25)
q2 = df["abs_segregation"].quantile(0.50)
q3 = df["abs_segregation"].quantile(0.75)

df["quartile"] = pd.cut(
    df["abs_segregation"],
    bins=[-float("inf"), q1, q2, q3, float("inf")],
    labels=["Q1", "Q2", "Q3", "Q4"],
    include_lowest=True
)

# --------------------------------------------------
# 4. Keep only Q1 and Q4
# --------------------------------------------------

model_df = df[df["quartile"].isin(["Q1", "Q4"])].copy()

model_df["Q4"] = (model_df["quartile"] == "Q4").astype(int)

# Make SES categorical with SES 1 as reference
model_df["ses"] = pd.Categorical(
    model_df["ses"],
    categories=["SES 1", "SES 2", "SES 3", "SES 4"]
)

# --------------------------------------------------
# 5. Logistic regression
#
# MET ~ Q4 + SES
# --------------------------------------------------

model = smf.logit(
    "MET ~ Q4 + C(ses, Treatment(reference='SES 1'))",
    data=model_df
).fit()

print(model.summary())

# --------------------------------------------------
# 6. Convert coefficients to odds ratios
# --------------------------------------------------

results = pd.DataFrame({
    "Coefficient": model.params,
    "OR": model.params.apply(lambda x: __import__("math").exp(x)),
    "p_value": model.pvalues,
})

# 95% CI on coefficient scale, then exponentiate
conf = model.conf_int()
results["CI_lower"] = conf[0].apply(lambda x: __import__("math").exp(x))
results["CI_upper"] = conf[1].apply(lambda x: __import__("math").exp(x))

print("\nOdds Ratios")
print(
    results[
        ["OR", "CI_lower", "CI_upper", "p_value"]
    ]
)