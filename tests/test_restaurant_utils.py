import unittest
import pandas as pd

from restaurant_utils import RestaurantFinder, distance_miles, extract_badges


class TestRestaurantUtils(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "name": ["Pizza Place", "Burger Spot"],
                "city": ["Philly", "Indy"],
                "category": ["Pizza", "Burgers"],
                "rating": [4.8, 3.9],
                "reviews": [120, 45],
                "zip": ["19107", "46250"],
                "comment": ["funny cool useful", "nice place"],
            }
        )
        self.zip_coords = {
            "19107": (39.9495, -75.1498),
            "46250": (39.9012, -86.0803),
        }
        self.finder = RestaurantFinder(self.df, self.zip_coords)

    def test_distance_zero(self):
        d = distance_miles(39.95, -75.15, 39.95, -75.15)
        self.assertAlmostEqual(d, 0.0, places=6)

    def test_extract_badges(self):
        badges = extract_badges(self.df.iloc[0], "comment")
        self.assertEqual(set(badges), {"funny", "cool", "useful"})

    def test_filter_by_city(self):
        out = self.finder.filter(city="Philly", city_col="city")
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["name"], "Pizza Place")

    def test_top_picks(self):
        out = self.finder.top_picks(self.df, name_col="name", city_col="city", category_col="category", rating_col="rating", review_col="reviews", limit=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["name"], "Pizza Place")


if __name__ == "__main__":
    unittest.main()
