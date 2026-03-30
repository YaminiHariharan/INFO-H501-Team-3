import pandas as pd
import matplotlib.pyplot as plt

def generate_graphs(df):

    # Split into cities
    indy = df[df["city"].str.contains("Indianapolis", case=False, na=False)]
    philly = df[df["city"].str.contains("Philadelphia", case=False, na=False)]

    # -------------------------------
    # GRAPH 1 — Restaurant Count
    # -------------------------------
    counts = [len(indy), len(philly)]

    plt.figure()
    plt.bar(["Indianapolis, IN", "Philadelphia, PA"], counts)
    plt.title("Restaurant Count: IN vs PA")
    plt.savefig("graphs/restaurants_by_city.png")
    plt.close()

    # -------------------------------
    # GRAPH 2 — Top Categories (Top 10)
    # -------------------------------
    indy_categories = indy["categories"].dropna().str.split(", ").explode()
    philly_categories = philly["categories"].dropna().str.split(", ").explode()

    indy_top = indy_categories.value_counts().head(10)
    philly_top = philly_categories.value_counts().head(10)

    combined = pd.DataFrame({
        "Indianapolis, IN": indy_top,
        "Philadelphia, PA": philly_top
    }).fillna(0)

    combined.plot(kind="bar", figsize=(10,6))
    plt.title("Top 10 Cuisine Comparison: IN vs PA")
    plt.ylabel("Number of Restaurants")
    plt.tight_layout()
    plt.savefig("graphs/top_categories.png")
    plt.close()

    # -------------------------------
    # GRAPH 3 — Rating Distribution
    # -------------------------------
    plt.figure()
    plt.hist(indy["avg_stars"], alpha=0.5, label="Indianapolis, IN")
    plt.hist(philly["avg_stars"], alpha=0.5, label="Philadelphia, PA")

    plt.legend()
    plt.title("Rating Distribution")
    plt.savefig("graphs/rating_distribution.png")
    plt.close()

    # -------------------------------
    # GRAPH 4 — Reviews Comparison
    # -------------------------------
    reviews = [indy["total_reviews"].sum(), philly["total_reviews"].sum()]

    plt.figure()
    plt.bar(["Indianapolis, IN", "Philadelphia, PA"], reviews)
    plt.title("Total Reviews: IN vs PA")
    plt.savefig("graphs/reviews_comparison.png")
    plt.close()

    # -------------------------------
    # GRAPH 5 — Value (Rating vs Reviews)
    # -------------------------------
    plt.figure()

    plt.scatter(indy["total_reviews"], indy["avg_stars"], alpha=0.5, label="Indianapolis, IN")
    plt.scatter(philly["total_reviews"], philly["avg_stars"], alpha=0.5, label="Philadelphia, PA")

    plt.xlabel("Review Count")
    plt.ylabel("Average Rating")
    plt.title("Value: Rating vs Reviews (IN vs PA)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("graphs/value_reviews_vs_rating.png")
    plt.close()

    # -------------------------------
    # GRAPH 6 — Price (if available)
    # -------------------------------
    if "price" in df.columns:
        plt.figure()

        indy_price = indy["price"].dropna()
        philly_price = philly["price"].dropna()

        plt.boxplot([indy_price, philly_price],
                    labels=["Indianapolis, IN", "Philadelphia, PA"])

        plt.title("Price Distribution: IN vs PA")
        plt.ylabel("Price Level")

        plt.tight_layout()
        plt.savefig("graphs/price_distribution.png")
        plt.close()
