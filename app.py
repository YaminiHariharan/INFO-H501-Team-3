from __future__ import annotations

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from pathlib import Path

from restaurant_utils import (
    RestaurantFinder,
    find_col,
    infer_rating_col,
    infer_review_col,
    infer_comment_col,
    extract_badges,
    fallback_image,
)

# ---------- CONFIG ----------
st.set_page_config(page_title="Restaurant Finder", layout="wide")

# ---------- LOAD DATA ----------
def load_data():
    path = Path("data/processed/indy_philly_restaurant_aggregated.json")
    return pd.read_json(path, lines=True)

df = load_data()

# ---------- COLUMN DETECTION ----------
name_col = find_col(df, ["name"])
city_col = find_col(df, ["city"])
category_col = find_col(df, ["category"])
rating_col = infer_rating_col(df)
review_col = infer_review_col(df)
comment_col = infer_comment_col(df)
zip_col = find_col(df, ["postal_code", "zip"])
lat_col = find_col(df, ["latitude"])
lon_col = find_col(df, ["longitude"])

# ---------- ZIP COORDS ----------
ZIP_COORDS = {
    "19107": (39.9495, -75.1498),
    "46227": (39.6655, -86.1275),
    "46250": (39.9012, -86.0803),
}

# ---------- SIDEBAR ----------
st.sidebar.title("Filters")

search = st.sidebar.text_input("Search")
city = st.sidebar.selectbox("City", ["All"] + df[city_col].dropna().unique().tolist())
min_rating = st.sidebar.slider("Min Rating", 0.0, 5.0, 0.0)
user_zip = st.sidebar.selectbox("Your ZIP", list(ZIP_COORDS.keys()))
radius = st.sidebar.slider("Radius (miles)", 1, 20, 5)

# ---------- FILTER USING CLASS ----------
finder = RestaurantFinder(df, ZIP_COORDS)

filtered = finder.filter(
    search=search,
    city=city,
    category="All",
    min_rating=min_rating,
    min_reviews=0,
    user_zip=user_zip,
    radius_miles=radius,
    name_col=name_col,
    city_col=city_col,
    category_col=category_col,
    rating_col=rating_col,
    review_col=review_col,
    zip_col=zip_col,
    comment_col=comment_col,
    address_col=None,
)

picks = finder.top_picks(
    filtered,
    name_col=name_col,
    city_col=city_col,
    category_col=category_col,
    rating_col=rating_col,
    review_col=review_col,
)

# ---------- UI ----------
st.title("🍽️ Restaurant Finder")

# ---------- TOP PICKS ----------
st.subheader("🔥 Top Picks")

cols = st.columns(3)

for i, (_, row) in enumerate(picks.iterrows()):
    with cols[i % 3]:
        st.image(fallback_image(row.get(category_col)))
        st.write(f"**{row.get(name_col)}**")
        if rating_col:
            st.write(f"⭐ {row.get(rating_col)}")

# ---------- PROFILE ----------
st.subheader("Restaurant Profile")

choice = st.selectbox("Select Restaurant", filtered[name_col].dropna().unique())
row = filtered[filtered[name_col] == choice].iloc[0]

c1, c2 = st.columns(2)

with c1:
    st.image(fallback_image(row.get(category_col)))

with c2:
    st.write(f"Name: {row.get(name_col)}")
    st.write(f"City: {row.get(city_col)}")
    if rating_col:
        st.write(f"Rating: {row.get(rating_col)}")
    if review_col:
        st.write(f"Reviews: {row.get(review_col)}")

    badges = extract_badges(row, comment_col)
    if badges:
        st.write("Badges:", badges)

# ---------- MAP ----------
st.subheader("Map")

map_df = filtered.copy()
map_df = map_df.dropna(subset=[lat_col, lon_col])

if not map_df.empty:
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=map_df[lat_col].mean(),
            longitude=map_df[lon_col].mean(),
            zoom=10
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=f"[{lon_col}, {lat_col}]",
                get_radius=200,
                get_fill_color=[0, 200, 255],
                pickable=True
            )
        ]
    ))

# ---------- TABLE ----------
st.subheader("Results")
st.dataframe(filtered.head(20))
