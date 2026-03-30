import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Restaurant Explorer", layout="wide")

st.title("Restaurant Explorer")
st.write("Filter restaurants by city, cuisine, and price.")

DATA_FILE = Path("indy_philly_restaurant_aggregated.json")


@st.cache_data
def load_data(file_path: Path) -> pd.DataFrame:
    """Load JSON data into a DataFrame.

    Supports either:
    - a JSON array of objects, or
    - newline-delimited JSON (one object per line)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path.resolve()}")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw)
    except json.JSONDecodeError:
        df = pd.read_json(file_path, lines=True)

    if "price" not in df.columns:
        df["price"] = "Unknown"

    return df


try:
    df = load_data(DATA_FILE)
except Exception as e:
    st.error(str(e))
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

cities = sorted(df["city"].dropna().astype(str).unique().tolist())
selected_cities = st.sidebar.multiselect("City", cities, default=cities)

# Build a cuisine list from the categories column
all_cuisines = set()
for cats in df["categories"].dropna():
    for c in str(cats).split(","):
        all_cuisines.add(c.strip())

cuisines = sorted(all_cuisines)
selected_cuisines = st.sidebar.multiselect("Cuisine", cuisines)

price_options = sorted(df["price"].dropna().astype(str).unique().tolist())
selected_prices = st.sidebar.multiselect("Price", price_options, default=price_options)

min_stars = float(df["avg_stars"].min())
max_stars = float(df["avg_stars"].max())
stars_range = st.sidebar.slider("Stars range", min_value=min_stars, max_value=max_stars, value=(min_stars, max_stars))

filtered = df.copy()

if selected_cities:
    filtered = filtered[filtered["city"].astype(str).isin(selected_cities)]

if selected_prices:
    filtered = filtered[filtered["price"].astype(str).isin(selected_prices)]

if selected_cuisines:
    def has_any_cuisine(category_text: str) -> bool:
        category_list = [c.strip() for c in str(category_text).split(",")]
        return any(c in category_list for c in selected_cuisines)

    filtered = filtered[filtered["categories"].apply(has_any_cuisine)]

filtered = filtered[(filtered["avg_stars"] >= stars_range[0]) & (filtered["avg_stars"] <= stars_range[1])]

st.subheader(f"Results: {len(filtered)} restaurant(s)")

show_cols = ["name", "city", "categories", "avg_stars", "total_reviews", "price"]
show_cols = [c for c in show_cols if c in filtered.columns]
st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)

st.subheader("Quick summary")
col1, col2 = st.columns(2)
with col1:
    st.metric("Restaurants shown", len(filtered))
with col2:
    st.metric("Average rating", round(filtered["avg_stars"].mean(), 2) if len(filtered) else 0)

st.caption("Tip: add a real `price` column such as $, $$, $$$ to make the price filter more meaningful.")
