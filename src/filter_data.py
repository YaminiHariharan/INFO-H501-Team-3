import json
from pathlib import Path

# Set correct base directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "yelp_academic_dataset_business.json"
OUTPUT_FILE = DATA_DIR / "yelp_in_pa_business.json"

VALID_STATES = {"IN", "PA"}

print("Filtering Yelp dataset for IN and PA restaurants...")

count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in infile:
        business = json.loads(line)

        state = business.get("state")
        categories = business.get("categories")

        if (
            state in VALID_STATES
            and categories is not None
            and ("Restaurant" in categories or "Food" in categories)
        ):
            outfile.write(json.dumps(business) + "\n")
            count += 1

print(f"Total businesses kept: {count}")
