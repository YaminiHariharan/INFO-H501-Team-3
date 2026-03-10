import pandas as pd

from restaurant_analysis import (
    plot_review_count_distribution,
    plot_top_reviewed_restaurants,
    plot_rating_vs_reviews
)


def load_filtered_data():
    return pd.read_json("data/yelp_in_pa_business.json", lines=True)


def main():
    try:
        df = load_filtered_data()

        print("\nStatistical Summary:")
        print(f"Total Businesses: {len(df)}")
        print(f"Mean Star Rating: {df['stars'].mean():.2f}")
        print(f"Median Star Rating: {df['stars'].median():.2f}")
        print(f"Star Rating Std Dev: {df['stars'].std():.2f}")
        print(f"Indiana Businesses: {len(df[df['state']=='IN'])}")
        print(f"Pennsylvania Businesses: {len(df[df['state']=='PA'])}")

        plot_review_count_distribution(df)
        plot_top_reviewed_restaurants(df)
        plot_rating_vs_reviews(df)

        print("\nAll visualizations generated successfully.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
