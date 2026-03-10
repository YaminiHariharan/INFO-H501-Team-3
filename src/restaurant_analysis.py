import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

def ensure_output_directory():
    os.makedirs("assets/images", exist_ok=True)

def load_restaurant_data():
    return pd.read_json(
        "data/processed/indy_philly_restaurant_aggregated.json",
        lines=True
    )

def plot_review_count_distribution(df):
    ensure_output_directory()

    plt.figure(figsize=(8, 5))

    colors = {"IN": "blue", "PA": "red"}

    for state in df["state"].unique():
        subset = df[df["state"] == state]
        plt.hist(
            subset["total_reviews"],
            bins=30,
            alpha=0.5,
            label=state,
            color=colors.get(state, "gray")
        )

    plt.title("Review Count Distribution (IN vs PA)")
    plt.xlabel("Total Reviews")
    plt.ylabel("Number of Restaurants")
    plt.legend()

    plt.tight_layout()
    plt.savefig("assets/images/review_count_distribution.png", dpi=300)
    plt.close()

def plot_top_reviewed_restaurants(df):
    ensure_output_directory()

    plt.figure(figsize=(10, 6))

    colors = {"IN": "blue", "PA": "red"}

    for state in ["IN", "PA"]:
        top = (
            df[df["state"] == state]
            .sort_values("total_reviews", ascending=False)
            .head(5)
        )

        plt.barh(
            top["name"],
            top["total_reviews"],
            color=colors[state],
            alpha=0.7,
            label=state
        )

    plt.title("Top Reviewed Restaurants by State")
    plt.xlabel("Total Reviews")
    plt.legend()

    # Correlation calculation
    correlation = df["total_reviews"].corr(df["avg_stars"])
    print(f"\nCorrelation between total reviews and average rating: {correlation:.3f}")

    plt.tight_layout()
    plt.savefig("assets/images/top_reviewed_restaurants.png", dpi=300)
    plt.close()

def plot_rating_vs_reviews(df):
    ensure_output_directory()

    plt.figure(figsize=(8, 5))

    colors = {"IN": "blue", "PA": "red"}

    for state in df["state"].unique():
        subset = df[df["state"] == state]
        plt.scatter(
            subset["total_reviews"],
            subset["avg_stars"],
            alpha=0.5,
            label=state,
            color=colors.get(state, "gray")
        )

    # Calculate correlation
    correlation = df["total_reviews"].corr(df["avg_stars"])

    # Display correlation on plot
    plt.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.3f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle="round", alpha=0.2)
    )

    plt.title("Average Rating vs Total Reviews (IN vs PA)")
    plt.xlabel("Total Reviews")
    plt.ylabel("Average Star Rating")
    plt.legend()

    plt.tight_layout()
    plt.savefig("assets/images/rating_vs_reviews.png", dpi=300)
    plt.close()


    plt.title("Average Rating vs Total Reviews (IN vs PA)")
    plt.xlabel("Total Reviews")
    plt.ylabel("Average Star Rating")
    plt.legend()

    plt.tight_layout()
    plt.savefig("assets/images/rating_vs_reviews.png", dpi=300)
    plt.close()

def plot_average_reviews_by_state(df):
    ensure_output_directory()

    avg_reviews = df.groupby("state")["total_reviews"].mean()

    colors = {"IN": "blue", "PA": "red"}

    plt.figure(figsize=(6,5))
    bars = plt.bar(
        avg_reviews.index,
        avg_reviews.values,
        color=[colors.get(state, "gray") for state in avg_reviews.index]
    )

    for i, value in enumerate(avg_reviews.values):
        plt.text(i, value, f"{value:.1f}", ha="center", va="bottom")

    plt.title("Average Review Count per Restaurant by State")
    plt.xlabel("State")
    plt.ylabel("Average Review Count")

    plt.tight_layout()
    plt.savefig("assets/images/avg_reviews_by_state.png", dpi=300)
    plt.close()

def plot_rating_boxplot_by_state(df):
    ensure_output_directory()

    plt.figure(figsize=(6, 5))

    data = [
        df[df["state"] == "IN"]["avg_stars"],
        df[df["state"] == "PA"]["avg_stars"]
    ]

    box = plt.boxplot(
        data,
        patch_artist=True,
        labels=["IN", "PA"]
    )

    colors = ["blue", "red"]

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    plt.title("Distribution of Restaurant Ratings by State")
    plt.ylabel("Average Star Rating")

    plt.tight_layout()
    plt.savefig("assets/images/rating_boxplot_by_state.png", dpi=300)
    plt.close()
