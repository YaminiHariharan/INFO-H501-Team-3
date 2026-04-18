from __future__ import annotations

import math
from typing import Optional
import pandas as pd


def find_col(df: pd.DataFrame, keywords: list[str]) -> Optional[str]:
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def infer_rating_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["rating", "stars"])


def infer_review_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["review"])


def infer_comment_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["comment", "text"])


def fallback_image(category: object) -> str:
    return f"https://source.unsplash.com/800x500/?{category}"


def extract_badges(row: pd.Series, comment_col: Optional[str]) -> list[str]:
    if not comment_col or comment_col not in row.index:
        return []
    text = str(row.get(comment_col, "")).lower()
    return [w for w in ["funny", "cool", "useful"] if w in text]


def distance_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))


# ✅ THIS IS THE IMPORTANT PART
class RestaurantFinder:

    def __init__(self, df: pd.DataFrame, zip_coords: dict):
        self.df = df
        self.zip_coords = zip_coords

    def filter(self, search, city, category, min_rating, min_reviews,
               user_zip, radius_miles, name_col, city_col, category_col,
               rating_col, review_col, zip_col, comment_col, address_col):

        df = self.df.copy()

        if search:
            df = df[df.astype(str).apply(lambda r: r.str.contains(search, case=False).any(), axis=1)]

        if city != "All" and city_col:
            df = df[df[city_col] == city]

        if category != "All" and category_col:
            df = df[df[category_col] == category]

        if rating_col:
            df = df[pd.to_numeric(df[rating_col], errors="coerce") >= min_rating]

        if review_col:
            df = df[pd.to_numeric(df[review_col], errors="coerce") >= min_reviews]

        if zip_col and user_zip in self.zip_coords:
            lat1, lon1 = self.zip_coords[user_zip]

            def within(row):
                z = str(row.get(zip_col))
                if z in self.zip_coords:
                    lat2, lon2 = self.zip_coords[z]
                    return distance_miles(lat1, lon1, lat2, lon2) <= radius_miles
                return False

            df = df[df.apply(within, axis=1)]

        return df

    def top_picks(self, df, name_col, city_col, category_col, rating_col, review_col, limit=6):

        temp = df.copy()

        if rating_col:
            temp["rating_num"] = pd.to_numeric(temp[rating_col], errors="coerce")

        if review_col:
            temp["review_num"] = pd.to_numeric(temp[review_col], errors="coerce")

        temp = temp.sort_values(["rating_num", "review_num"], ascending=[False, False])

        return temp.head(limit)
