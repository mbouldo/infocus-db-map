import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ============================================================
# Load data
# ============================================================

with open("combined_data.jsonl", "r") as f:
    data = [json.loads(line) for line in f if line.strip()]

df = pd.DataFrame(data)

# Keep required variables
df = df[["GEOID", "outcome", "segregation", "income", "ses"]].copy()

# Numeric conversion
df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")
df["segregation"] = pd.to_numeric(df["segregation"], errors="coerce")

# Remove rows missing variables needed for analysis
df = df.dropna(subset=["outcome", "segregation", "ses"])


# ============================================================
# Outcome threshold
# ============================================================

outcome_median = df["outcome"].median()

df["met"] = (df["outcome"] > outcome_median).astype(int)

print("\n==============================")
print("OUTCOME")
print("==============================")
print(f"Outcome median: {outcome_median:.4f}")
print(f"Met (> median): {df['met'].sum()}")
print(f"Not met (<= median): {(df['met'] == 0).sum()}")
print(f"Total: {len(df)}")


# ============================================================
# Global segregation quartiles
# ============================================================

q1 = df["segregation"].quantile(0.25)
q2 = df["segregation"].quantile(0.50)
q3 = df["segregation"].quantile(0.75)

print("\n==============================")
print("GLOBAL SEGREGATION QUARTILES")
print("==============================")
print(f"Q1 cutoff: {q1:.6f}")
print(f"Q2 cutoff: {q2:.6f}")
print(f"Q3 cutoff: {q3:.6f}")

print("\nInterpretation:")
print(f"Q1: segregation <= {q1:.6f}")
print(f"Q2: {q1:.6f} < segregation <= {q2:.6f}")
print(f"Q3: {q2:.6f} < segregation <= {q3:.6f}")
print(f"Q4: segregation > {q3:.6f}")


# ============================================================
# Assign global segregation quartiles
# ============================================================

df["seg_quartile"] = pd.cut(
    df["segregation"],
    bins=[-np.inf, q1, q2, q3, np.inf],
    labels=["Q1", "Q2", "Q3", "Q4"],
    include_lowest=True
)


# ============================================================
# Counts for every SES x segregation quartile
# ============================================================

print("\n==============================")
print("COUNTS BY SES AND SEGREGATION QUARTILE")
print("==============================")

counts = (
    df.groupby(["ses", "seg_quartile"], observed=True)
      .agg(
          total=("met", "size"),
          met=("met", "sum")
      )
      .reset_index()
)

counts["not_met"] = counts["total"] - counts["met"]

print(counts.to_string(index=False))


# ============================================================
# Logistic regression within each SES
# Q1 is the reference group
# ============================================================

print("\n==============================")
print("STRATIFIED LOGISTIC REGRESSION")
print("==============================")
print("Outcome: met (> global outcome median)")
print("Predictor: global segregation quartile")
print("Reference: Q1")
print()

results = []

for ses in ["SES 1", "SES 2", "SES 3", "SES 4"]:

    subset = df[df["ses"] == ses].copy()

    print(f"\n---------- {ses} ----------")
    print(f"Total N: {len(subset)}")
    print(f"Met: {subset['met'].sum()}")
    print(f"Not met: {(subset['met'] == 0).sum()}")

    # Show group sizes
    print("\nQuartile counts:")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        qdata = subset[subset["seg_quartile"] == q]

        print(
            f"{q}: n={len(qdata)}, "
            f"met={qdata['met'].sum()}, "
            f"not_met={(qdata['met'] == 0).sum()}"
        )

    # Logistic regression
    #
    # C(seg_quartile, Treatment(reference="Q1"))
    # means Q1 is the reference category.
    #
    # Therefore:
    # Q2 coefficient = log(OR Q2 vs Q1)
    # Q3 coefficient = log(OR Q3 vs Q1)
    # Q4 coefficient = log(OR Q4 vs Q1)

    try:
        model = smf.logit(
            'met ~ C(seg_quartile, Treatment(reference="Q1"))',
            data=subset
        ).fit(disp=False)

        print("\nOdds ratios:")
        print(f"{'Comparison':<12} {'OR':>10} {'95% CI':>22} {'p':>12}")

        # Q1 reference
        print(f"{'Q1 ref':<12} {'1.000':>10} {'reference':>22} {'--':>12}")

        for q in ["Q2", "Q3", "Q4"]:

            term = (
                f'C(seg_quartile, Treatment(reference="Q1"))[T.{q}]'
            )

            beta = model.params[term]
            ci_low, ci_high = model.conf_int().loc[term]
            p = model.pvalues[term]

            odds_ratio = np.exp(beta)
            ci_low = np.exp(ci_low)
            ci_high = np.exp(ci_high)

            print(
                f"{q} vs Q1     "
                f"{odds_ratio:>10.3f} "
                f"[{ci_low:.3f}, {ci_high:.3f}] "
                f"{p:>12.4g}"
            )

            results.append({
                "SES": ses,
                "comparison": f"{q} vs Q1",
                "OR": odds_ratio,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "p": p
            })

    except Exception as e:
        print(f"\nRegression failed for {ses}: {e}")


# ============================================================
# Combined results table
# ============================================================

results_df = pd.DataFrame(results)

print("\n==============================")
print("FINAL RESULTS")
print("==============================")

print(results_df.to_string(index=False))

# ============================================================
# XY results table
# ============================================================

results_df["OR (95% CI)"] = results_df.apply(
    lambda r: f"{r['OR']:.3f} ({r['CI_low']:.3f}, {r['CI_high']:.3f})",
    axis=1
)

xy_table = (
    results_df
    .pivot(
        index="SES",
        columns="comparison",
        values="OR (95% CI)"
    )
    .reindex(["SES 1", "SES 2", "SES 3", "SES 4"])
    [["Q2 vs Q1", "Q3 vs Q1", "Q4 vs Q1"]]
)

print("\n==============================")
print("ODDS RATIOS")
print("==============================")
print(xy_table.to_string())