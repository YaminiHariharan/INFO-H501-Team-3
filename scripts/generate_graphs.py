import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_json(
    "data/processed/indy_philly_restaurant_aggregated.json",
    lines=True
)

# Filter cities
indy = df[df["city"] == "Indianapolis"]
philly = df[df["city"] == "Philadelphia"]

# -------------------------------
# GRAPH 1: Restaurants by City
# -------------------------------
city_counts = df["city"].value_counts()

plt.figure()
city_counts.plot(kind="bar", color=["blue","red"])

plt.title("Restaurant Count: Indianapolis vs Philadelphia")
plt.xlabel("City")
plt.ylabel("Number of Restaurants")

plt.tight_layout()
plt.savefig("graphs/restaurants_by_city.png")
plt.close()

# -------------------------------
# GRAPH 2: Top Categories
# -------------------------------
categories = df["categories"].dropna().str.split(", ").explode()

top_categories = categories.value_counts().head(10)

plt.figure()
top_categories.plot(kind="bar")

plt.title("Top Restaurant Categories")
plt.xlabel("Category")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("graphs/top_categories.png")
plt.close()

# -------------------------------
# GRAPH 3: Rating Distribution
# -------------------------------
plt.figure()

plt.hist(indy["avg_stars"], bins=10, alpha=0.7, label="Indiana", color="blue")
plt.hist(philly["avg_stars"], bins=10, alpha=0.7, label="Pennsylvania", color="red")

plt.title("Restaurant Rating Distribution")
plt.xlabel("Average Rating")
plt.ylabel("Number of Restaurants")

plt.legend()

plt.tight_layout()
plt.savefig("graphs/rating_distribution.png")
plt.close()

# -------------------------------
# GRAPH 4: Total Reviews
# -------------------------------
reviews = {
    "Indianapolis": indy["total_reviews"].sum(),
    "Philadelphia": philly["total_reviews"].sum()
}

plt.figure()

plt.bar(reviews.keys(), reviews.values(), color=["blue","red"])

plt.title("Total Reviews by City")
plt.ylabel("Review Count")

plt.tight_layout()
plt.savefig("graphs/reviews_comparison.png")
plt.close()

print("Graphs generated successfully in /graphs")
