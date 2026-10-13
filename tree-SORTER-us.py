import pandas as pd
import numpy as np

main_seg_var = "Racial Segregation"
SEG_CUTOFFS = [-0.5,-0.25,0.25,0.5]
main_outcome_var = "Met_Colon_Screen"

#{"GEOID":"01001020100","Gini Coefficient":0.4238,"Economic Segregation":-0.06385869565217392,"Racial Economic Segregation":0.09918478260869565,"Racial Segregation":0.5845030881527232,"Gender Pay Gap":0.3368427728800659,"Uninsured Children":0,"Uninsured (Black)":0.020833333333333332,"Uninsured (Hispanic)":0.2132701421800948,"Living Below Poverty (Black)":0,"Living Below Poverty (Hispanic)":0,"Lack_English_Prof":0,"Insurance Coverage":0.8876341050254094,"Medicaid Enrollment":0.2365894974590627,"Household Income":63237,"Annual Labor Force Participation Rate":0.5566298342541437,"Annual Unemployment Rate":0.0037220843672456576,"Below Poverty":0.15547024952015356,"Received Public Assistance":0.1358695652173913,"Uninsured":0.1123658949745906,"Met_Breast_Screen":78,"Met_Colon_Screen":65,"Currently_Smoke":18.8,"BMI_Obese":39.3,"Physically_Inactive":30.3,"Binge_Drinking":15.6,"Sleep_Debt":37.9,"Cancer_Prevalence":8.6,"Bad_Health":23.1,"Poor_Physical":15.6,"Poor_Mental":20,"Depression":26.6,"Diabetes_DX":13.8,"High_BP":40.5,"High_Cholesterol":37.1,"BP_Medicine":80.6,"CHD":8.4,"Had_Stroke":4.2,"Asthma":11,"COPD":10,"No_Teeth":13.4,"Recent_Checkup":78,"Recent_Dentist":57.3,"Hearing_Disability":8.4,"Vision_Disability":6.2,"Cognitive_Disability":17.4,"Mobility_Disability":18.2,"Selfcare_Disability":4.8,"Independent_Living_Disability":10.1,"Socially_Isolated":35.7,"Food_Insecure":16.7,"Housing_Insecure":14.2,"Lacked_Reliable_Transportation":9.7,"Lacked_Social_Emotional_Support":25.9,"Total":1781,"Under 18":0.20887142055025268,"18 to 64":0.5957327344188658,"Over 64":0.19539584503088153,"Female":0.5204941044357103,"White":0.66535654126895,"Black":0.08085345311622684,"AIAN":0,"Asian":0.04435710275126333,"NHOPI":0,"Hispanic":0.1184727681078046,"Other_Races":0.09096013475575519,"Below 9th grade":0.01281085154483798,"High School":0.905802562170309,"College":0.25018839487565936,"Advanced Degree":0.10851544837980406}

VARTYPES = {
    "Household Income": "quartile",
    "Racial Segregation": "hard_coded",
    "Cancer_Prevalence": "median",
    "Medicaid Enrollment": "median",
    "Uninsured": "median",
    "Over 64": "median",
    "Met_Colon_Screen": "median", # will be outcome var, above or below median
    "Met_Breast_Screen": "median"
}
# Household Income -> Quartile
# Racial Segregation -> Hard Coded SEG_CUTOFFS

# History of Cancer Diagnosis -> Above or Below Median
# Enrolled in Medicaid -> Above or Below Median
# Uninsured -> Above or Below Median
# Cancer Prevalance -> Above or Below Median
# Over 64 -> Above or Below Median

# Quartile Values: Q1,Q2,Q3,Q4
# Hard Coded Values: G1,G2,G3,G4
# Above or below median values: M1,M2

df = pd.read_json(
    "us-data/us_data.jsonl",
    lines=True,
    dtype={"GEOID": str}
)

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


grouped = pd.DataFrame({
    "GEOID": df["GEOID"]
})

for var, group_type in VARTYPES.items():
    values = pd.to_numeric(df[var], errors="coerce")
    output_col = f"{var}:group_assigned"

    if group_type == "quartile":
        grouped[output_col] = pd.qcut(
            values.rank(method="first"),
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"]
        ).astype("string")

    elif group_type == "hard_coded":
        grouped[output_col] = pd.cut(
            values,
            bins=[-np.inf, *SEG_CUTOFFS, np.inf],
            labels=["G1", "G2", "G3", "G4", "G5"]
        ).astype("string")

    elif group_type == "median":
        median = values.median()

        grouped[output_col] = np.where(
            values.isna(),
            None,
            np.where(values <= median, "M1", "M2")
        )

    else:
        raise ValueError(f"Unknown group type: {group_type}")

grouped.to_json(
    "grouped_us_tract.jsonl",
    orient="records",
    lines=True
)

print(f"Created grouped_us_tract.jsonl with {len(grouped)} rows.")
print(grouped.head())