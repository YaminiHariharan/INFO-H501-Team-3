import os
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


def ensure_output_directory():
    os.makedirs("assets/images", exist_ok=True)


def plot_review_count_distribution(df):
    ensure_output_directory()

    plt.figure(figsize=(8, 6))

    colors = {"IN": "blue", "PA": "red"}

    for state in ["IN", "PA"]:
        subset = df[df["state"] == state]
        plt.hist(
            subset["review_count"],
            bins=30,
            alpha=0.6,
            label=state,
            color=colors[state]
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
            .sort_values("review_count", ascending=False)
            .head(5)
        )

        plt.barh(
            top["name"],
            top["review_count"],
            color=colors[state],
            alpha=0.7,
            label=state
        )

    plt.title("Top Reviewed Restaurants by State")
    plt.xlabel("Total Reviews")
    plt.legend()

    plt.tight_layout()
    plt.savefig("assets/images/top_reviewed_restaurants.png", dpi=300)
    plt.close()


def plot_rating_vs_reviews(df):
    ensure_output_directory()

    plt.figure(figsize=(8, 6))

    colors = {"IN": "blue", "PA": "red"}

    for state in ["IN", "PA"]:
        subset = df[df["state"] == state]
        plt.scatter(
            subset["review_count"],
            subset["stars"],
            alpha=0.5,
            label=state,
            color=colors[state]
        )

    correlation = df["review_count"].corr(df["stars"])

    plt.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.3f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.2)
    )

    plt.title("Rating vs Review Count (IN vs PA)")
    plt.xlabel("Total Reviews")
    plt.ylabel("Star Rating")
    plt.legend()

    plt.tight_layout()
    plt.savefig("assets/images/rating_vs_reviews.png", dpi=300)
    plt.close()

    print(f"\nCorrelation between total reviews and average rating: {correlation:.3f}")
