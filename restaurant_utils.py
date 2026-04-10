from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


def find_col(df: pd.DataFrame, keywords: list[str]) -> Optional[str]:
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in keywords):
            return col
    return None


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def infer_rating_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["rating", "stars", "avg_stars", "average_rating", "avg_rating"])


def infer_review_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["review_count", "reviews", "total_reviews", "num_reviews"])


def infer_comment_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["comment", "comments", "review_text", "text", "description", "snippet", "tip", "note"])


def fallback_image(category: object) -> str:
    text = str(category or "restaurant").strip().lower().replace(" ", ",")
    if not text or text == "nan":
        text = "restaurant"
    return f"https://source.unsplash.com/1200x800/?{text}"


def extract_badges(row: pd.Series, comment_col: Optional[str]) -> list[str]:
    if not comment_col or comment_col not in row.index:
        return []
    text = str(row.get(comment_col, "")).lower()
    return [word for word in ["funny", "cool", "useful"] if word in text]


def distance_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class RestaurantFinder:
    df: pd.DataFrame
    zip_coords: dict[str, tuple[float, float]]

    def filter(
        self,
        *,
        search: str = "",
        city: str = "All",
        category: str = "All",
        min_rating: float = 0.0,
        min_reviews: int = 0,
        user_zip: Optional[str] = None,
        radius_miles: float = 5.0,
        name_col: Optional[str] = None,
        city_col: Optional[str] = None,
        category_col: Optional[str] = None,
        rating_col: Optional[str] = None,
        review_col: Optional[str] = None,
        zip_col: Optional[str] = None,
        comment_col: Optional[str] = None,
        address_col: Optional[str] = None,
    ) -> pd.DataFrame:
        filtered = self.df.copy()

        if search.strip():
            q = search.strip().lower()
            mask = pd.Series(False, index=filtered.index)
            for col in [name_col, city_col, category_col, address_col, zip_col, comment_col]:
                if col and col in filtered.columns:
                    mask |= clean_text(filtered[col]).str.contains(q, case=False, na=False)
            filtered = filtered[mask]

        if city_col and city != "All":
            filtered = filtered[clean_text(filtered[city_col]) == city]

        if category_col and category != "All":
            filtered = filtered[clean_text(filtered[category_col]) == category]

        if rating_col and rating_col in filtered.columns:
            filtered = filtered[pd.to_numeric(filtered[rating_col], errors="coerce").fillna(-1) >= min_rating]

        if review_col and review_col in filtered.columns:
            filtered = filtered[pd.to_numeric(filtered[review_col], errors="coerce").fillna(-1) >= min_reviews]

        if zip_col and zip_col in filtered.columns and user_zip in self.zip_coords:
            user_lat, user_lon = self.zip_coords[user_zip]

            def in_radius(row: pd.Series) -> bool:
                z = str(row.get(zip_col, "")).strip()[:5]
                if z in self.zip_coords:
                    lat, lon = self.zip_coords[z]
                    return distance_miles(user_lat, user_lon, lat, lon) <= radius_miles
                return False

            filtered = filtered[filtered.apply(in_radius, axis=1)]

        return filtered

    def top_picks(
        self,
        filtered: pd.DataFrame,
        *,
        name_col: Optional[str] = None,
        city_col: Optional[str] = None,
        category_col: Optional[str] = None,
        rating_col: Optional[str] = None,
        review_col: Optional[str] = None,
        limit: int = 6,
    ) -> pd.DataFrame:
        picks = filtered.copy()

        if rating_col and rating_col in picks.columns:
            picks["__rating__"] = pd.to_numeric(picks[rating_col], errors="coerce")
        if review_col and review_col in picks.columns:
            picks["__reviews__"] = pd.to_numeric(picks[review_col], errors="coerce")

        sort_cols = []
        ascending = []
        if "__rating__" in picks.columns:
            sort_cols.append("__rating__")
            ascending.append(False)
        if "__reviews__" in picks.columns:
            sort_cols.append("__reviews__")
            ascending.append(False)
        if name_col and name_col in picks.columns:
            sort_cols.append(name_col)
            ascending.append(True)

        if sort_cols:
            picks = picks.sort_values(sort_cols, ascending=ascending, na_position="last")

        cols = [c for c in [name_col, city_col, category_col, rating_col, review_col] if c and c in picks.columns]
        return picks.loc[:, cols].head(limit).copy()
