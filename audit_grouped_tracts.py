import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "grouped_us_tract.jsonl"
OUTPUT_DIR = Path("group_audit")
SEG_VAR = "Racial Segregation:group_assigned"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_json(INPUT_FILE, lines=True, dtype={"GEOID": str})

group_vars = [col for col in df.columns if col.endswith(":group_assigned")]

if not group_vars:
    raise ValueError("No :group_assigned columns found.")

# Audit every variable, with a larger chart for racial segregation.
for var in group_vars:
    counts = df[var].value_counts(dropna=False)

    # Put missing values last and label them explicitly.
    counts.index = [
        "Missing" if pd.isna(x) else str(x)
        for x in counts.index
    ]

    if var == SEG_VAR:
        order = ["G1", "G2", "G3", "G4", "G5", "Missing"]
        counts = counts.reindex(
            [x for x in order if x in counts.index],
            fill_value=0
        )
        figsize = (10, 6)
    else:
        figsize = (7, 4)

    fig, ax = plt.subplots(figsize=figsize)

    counts.plot(kind="bar", ax=ax)

    ax.set_title(var.replace(":group_assigned", "") + " Frequency")
    ax.set_xlabel("Assigned Group")
    ax.set_ylabel("Number of Census Tracts")
    ax.tick_params(axis="x", rotation=0)

    # Add count labels above bars.
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)

    fig.tight_layout()

    filename = var.replace(":group_assigned", "").replace(" ", "_")
    fig.savefig(OUTPUT_DIR / f"{filename}.png", dpi=150)
    plt.close(fig)

    print(f"\n{var}")
    print(counts.to_string())
    print(f"Total: {counts.sum():,}")

print(f"\nSaved {len(group_vars)} charts to {OUTPUT_DIR.resolve()}")