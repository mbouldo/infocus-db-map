import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILES = [
    "tract_disparity_f.csv",
    "tract_economy_f.csv",
    "tract_screening_f.csv"
]

OUTPUT_FILE = BASE_DIR / "us_data.jsonl"

METADATA_COLUMNS = {"FIPS", "Tract", "County", "State"}


def read_csv(filename):
    data = {}

    filepath = BASE_DIR / filename

    with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            fips = row["FIPS"].strip()

            if fips not in data:
                data[fips] = {"GEOID": fips}

            for column, value in row.items():
                if column in METADATA_COLUMNS:
                    continue

                if value is None or value.strip() == "":
                    continue

                try:
                    numeric_value = float(value)

                    if numeric_value.is_integer():
                        numeric_value = int(numeric_value)

                    data[fips][column] = numeric_value

                except ValueError:
                    data[fips][column] = value

    return data


# Combine all files by FIPS/GEOID
combined = {}

for filename in INPUT_FILES:
    file_data = read_csv(filename)

    for fips, values in file_data.items():
        if fips not in combined:
            combined[fips] = {"GEOID": fips}

        for column, value in values.items():
            if column != "GEOID":
                combined[fips][column] = value


# Write JSONL
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for fips in sorted(combined):
        f.write(
            json.dumps(combined[fips], separators=(",", ":")) + "\n"
        )


print(f"Created: {OUTPUT_FILE}")
print(f"Records: {len(combined)}")
print(f"Files: {len(INPUT_FILES)}")