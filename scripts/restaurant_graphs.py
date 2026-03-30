cp scripts/generate_graphs.py scripts/generate_graphs.py.bak
cat > scripts/generate_graphs.py <<'PY'
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("graphs", exist_ok=True)

df = pd.read_json("data/processed/indy_philly_restaurant_aggregated.json", lines=True)

indy = df[df["city"].astype(str).str.contains("Indianapolis", case=False, na=False)].copy()
philly = df[df["city"].astype(str).str.contains("Philadelphia", case=False, na=False)].copy()

df["city_label"] = df["city"].apply(
    lambda x: "Indianapolis, IN" if "indianapolis" in str(x).lower()
    else "Philadelphia, PA" if "philadelphia" in str(x).lower()
    else str(x)
)
indy["city_label"] = "Indianapolis, IN"
philly["city_label"] = "Philadelphia, PA"

# GRAPH 1 — Restaurant Count by City
city_counts = df["city_label"].value_counts().reindex(
    ["Indianapolis, IN", "Philadelphia, PA"]
).dropna()

plt.figure(figsize=(8, 5))
city_counts.plot(kind="bar")
plt.title("Restaurant Count: Indianapolis, IN vs Philadelphia, PA")
plt.xlabel("City")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig("graphs/restaurants_by_city.png")
plt.close()

# GRAPH 2 — Top Categories
categories = df["categories"].dropna().astype(str).str.split(", ").explode()
top_categories = categories.value_counts().head(10)

plt.figure(figsize=(10, 6))
top_categories.plot(kind="bar")
plt.title("Top Restaurant Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("graphs/top_categories.png")
plt.close()

# GRAPH 3 — Rating Distribution
plt.figure(figsize=(8, 5))
plt.hist(indy["avg_stars"].dropna(), bins=10, alpha=0.7, label="Indianapolis, IN")
plt.hist(philly["avg_stars"].dropna(), bins=10, alpha=0.7, label="Philadelphia, PA")
plt.title("Restaurant Rating Distribution")
plt.xlabel("Average Rating")
plt.ylabel("Number of Restaurants")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/rating_distribution.png")
plt.close()

# GRAPH 4 — Total Reviews
reviews = {
    "Indianapolis, IN": indy["total_reviews"].fillna(0).sum(),
    "Philadelphia, PA": philly["total_reviews"].fillna(0).sum(),
}

plt.figure(figsize=(6, 5))
plt.bar(reviews.keys(), reviews.values())
plt.title("Total Reviews by City")
plt.ylabel("Review Count")
plt.tight_layout()
plt.savefig("graphs/reviews_comparison.png")
plt.close()

# GRAPH 5 — Cuisine Side-by-Side (Top 10)
indy_categories = indy["categories"].dropna().astype(str).str.split(", ").explode()
philly_categories = philly["categories"].dropna().astype(str).str.split(", ").explode()

indy_top = indy_categories.value_counts().head(10)
philly_top = philly_categories.value_counts().head(10)

combined = pd.DataFrame({
    "Indianapolis, IN": indy_top,
    "Philadelphia, PA": philly_top
}).fillna(0)

plt.figure(figsize=(12, 6))
combined.plot(kind="bar")
plt.title("Top Cuisine Comparison: Indianapolis, IN vs Philadelphia, PA")
plt.xlabel("Cuisine Type")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("graphs/cuisine_side_by_side.png")
plt.close()

def clean_categories(frame):
    out = frame.copy()
    out["categories"] = out["categories"].fillna("").astype(str)
    out = out.assign(categories=out["categories"].str.split(", ")).explode("categories")
    out["categories"] = out["categories"].astype(str).str.strip()
    out = out[out["categories"].ne("") & out["categories"].ne("nan")]
    return out

# GRAPH 6 — Top 5 Highest-Rated Cuisines
exp = clean_categories(df)

rated = (
    exp.groupby(["city_label", "categories"])
    .agg(
        avg_rating=("avg_stars", "mean"),
        restaurant_count=("categories", "size"),
    )
    .reset_index()
)

rated = rated[rated["restaurant_count"] >= 5]

top_rated = (
    rated.sort_values(["city_label", "avg_rating"], ascending=[True, False])
    .groupby("city_label")
    .head(5)
)

top_rated_pivot = top_rated.pivot(index="categories", columns="city_label", values="avg_rating").fillna(0)

plt.figure(figsize=(12, 6))
top_rated_pivot.plot(kind="bar")
plt.title("Top 5 Highest-Rated Cuisines: Indianapolis, IN vs Philadelphia, PA")
plt.xlabel("Cuisine Type")
plt.ylabel("Average Rating")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("graphs/top_rated_cuisines.png")
plt.close()

# GRAPH 7 — Best Value Cuisines (proxy score)
# Because this processed dataset has no price column, value is estimated by
# rating weighted by review volume.
value_df = clean_categories(df)
value_df = value_df[value_df["total_reviews"].notna() & value_df["avg_stars"].notna()].copy()
value_df["value_score"] = value_df["avg_stars"] * np.log1p(value_df["total_reviews"])

value_scores = (
    value_df.groupby(["city_label", "categories"])
    .agg(
        value_score=("value_score", "mean"),
        restaurant_count=("categories", "size"),
    )
    .reset_index()
)

value_scores = value_scores[value_scores["restaurant_count"] >= 5]

best_value = (
    value_scores.sort_values(["city_label", "value_score"], ascending=[True, False])
    .groupby("city_label")
    .head(5)
)

best_value_pivot = best_value.pivot(index="categories", columns="city_label", values="value_score").fillna(0)

plt.figure(figsize=(12, 6))
best_value_pivot.plot(kind="bar")
plt.title("Best Value Cuisines: Indianapolis, IN vs Philadelphia, PA")
plt.xlabel("Cuisine Type")
plt.ylabel("Value Score (Stars x log(Reviews + 1))")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("graphs/best_value_cuisines.png")
plt.close()

# GRAPH 8 — Postal Code Hotspots by Restaurant Count
postal_counts = (
    df.groupby(["city_label", "postal_code"])
    .size()
    .reset_index(name="restaurant_count")
)

postal_counts = postal_counts[postal_counts["restaurant_count"] >= 5]

top_postal_counts = (
    postal_counts.sort_values(["city_label", "restaurant_count"], ascending=[True, False])
    .groupby("city_label")
    .head(10)
)

postal_counts_pivot = top_postal_counts.pivot(index="postal_code", columns="city_label", values="restaurant_count").fillna(0)

plt.figure(figsize=(12, 6))
postal_counts_pivot.plot(kind="bar")
plt.title("Top Postal Code Food Hotspots: Indianapolis, IN vs Philadelphia, PA")
plt.xlabel("Postal Code")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("graphs/postal_code_hotspots.png")
plt.close()

# GRAPH 9 — Postal Code Quality Hotspots
postal_quality = (
    df.groupby(["city_label", "postal_code"])
    .agg(
        avg_rating=("avg_stars", "mean"),
        restaurant_count=("postal_code", "size"),
    )
    .reset_index()
)

postal_quality = postal_quality[postal_quality["restaurant_count"] >= 5]

top_postal_quality = (
    postal_quality.sort_values(["city_label", "avg_rating"], ascending=[True, False])
    .groupby("city_label")
    .head(10)
)

postal_quality_pivot = top_postal_quality.pivot(index="postal_code", columns="city_label", values="avg_rating").fillna(0)

plt.figure(figsize=(12, 6))
postal_quality_pivot.plot(kind="bar")
plt.title("Top Postal Code Quality Hotspots: Indianapolis, IN vs Philadelphia, PA")
plt.xlabel("Postal Code")
plt.ylabel("Average Rating")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("graphs/postal_code_quality_hotspots.png")
plt.close()

print("Graphs generated successfully.")
PY
python3 scripts/generate_graphs.py
ls graphs
