from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Restaurant Market Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top, rgba(56, 189, 248, 0.10), transparent 28%),
                        radial-gradient(circle at 85% 15%, rgba(168, 85, 247, 0.10), transparent 22%),
                        #0b0f14;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1f2430 0%, #151922 100%);
        }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 16px 18px;
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            letter-spacing: -0.03em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- helpers ----------
def pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    return None


def first_existing_path(paths: Iterable[str]) -> Optional[Path]:
    for p in paths:
        path = Path(p)
        if path.exists():
            return path
    return None


def clean_text_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


@st.cache_data(show_spinner=False)
def load_data_from_path(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    raise ValueError("Unsupported file type.")


def load_uploaded(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded)
    if name.endswith(".json"):
        try:
            return pd.read_json(uploaded, lines=True)
        except ValueError:
            return pd.read_json(uploaded)
    raise ValueError("Unsupported upload type.")


def infer_rating_column(df: pd.DataFrame) -> Optional[str]:
    preferred = pick_col(
        df,
        [
            "stars",
            "rating",
            "average_rating",
            "avg_rating",
            "avg_stars",
            "average_stars",
            "restaurant_rating",
        ],
    )
    if preferred:
        return preferred

    best_col = None
    best_score = 0.0
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        within_5 = s.between(0, 5).mean()
        within_10 = s.between(0, 10).mean()
        score = max(within_5, within_10 * 0.9)
        if score > best_score and s.median() <= 5:
            best_col = col
            best_score = score
    return best_col


def insight_lines(df: pd.DataFrame, city_col: Optional[str], rating_series: Optional[pd.Series], reviews_col: Optional[str], category_col: Optional[str]) -> list[str]:
    lines: list[str] = []

    if city_col and city_col in df.columns and not df.empty:
        city_counts = clean_text_series(df[city_col]).value_counts()
        if not city_counts.empty:
            lines.append(f"{city_counts.index[0]} leads this view with {int(city_counts.iloc[0]):,} restaurants.")

    if rating_series is not None and not df.empty:
        avg_rating = rating_series.loc[df.index].mean()
        if pd.notna(avg_rating):
            lines.append(f"Average rating in the current view is {avg_rating:.2f}/5.")

    if reviews_col and reviews_col in df.columns and not df.empty:
        revs = pd.to_numeric(df[reviews_col], errors="coerce")
        if revs.notna().any():
            lines.append(f"Total review activity is {int(revs.sum()):,}.")

    if category_col and category_col in df.columns and not df.empty:
        cat_counts = clean_text_series(df[category_col]).value_counts()
        if not cat_counts.empty:
            lines.append(f"Most common category: {cat_counts.index[0]}.")

    return lines[:4] if lines else ["Use the filters to surface the strongest story in the data."]


# ---------- load data ----------
st.sidebar.title("Controls")
st.sidebar.caption("Use these filters to make the story presentation-ready.")

uploaded = st.sidebar.file_uploader("Upload a data file", type=["csv", "xlsx", "xls", "json"])

if uploaded is not None:
    df = load_uploaded(uploaded)
    source_name = uploaded.name
else:
    fallback = first_existing_path([
        "data/processed/restaurants.csv",
        "data/processed/indy_philly_restaurant_aggregated.json",
    ])
    if fallback is None:
        st.error("No dataset found. Upload a file in the sidebar or place one in data/processed/.")
        st.stop()
    df = load_data_from_path(str(fallback))
    source_name = str(fallback)

city_col = pick_col(df, ["city", "city_name", "municipality", "town"])
name_col = pick_col(df, ["name", "business_name", "restaurant_name"])
reviews_col = pick_col(df, ["review_count", "reviews", "total_reviews", "num_reviews"])
category_col = pick_col(df, ["categories", "category", "cuisine", "cuisines", "primary_category"])
postal_col = pick_col(df, ["postal_code", "zip", "zipcode", "zip_code"])
address_col = pick_col(df, ["address", "full_address", "location"])
rating_col = infer_rating_column(df)
rating_series = pd.to_numeric(df[rating_col], errors="coerce") if rating_col else None

filtered = df.copy()

if city_col:
    city_values = sorted(clean_text_series(filtered[city_col]).replace("", pd.NA).dropna().unique())
    selected_cities = st.sidebar.multiselect("Cities", city_values, default=city_values)
    if selected_cities:
        filtered = filtered[clean_text_series(filtered[city_col]).isin(selected_cities)]

if rating_series is not None:
    rating_vals = rating_series.loc[filtered.index].dropna()
    if not rating_vals.empty:
        rating_floor = st.sidebar.slider(
            "Minimum rating",
            min_value=0.0,
            max_value=5.0,
            value=float(max(0.0, round(float(rating_vals.min()), 1))),
            step=0.1,
        )
        filtered = filtered[rating_series.loc[filtered.index] >= rating_floor]

if reviews_col:
    review_vals = pd.to_numeric(filtered[reviews_col], errors="coerce")
    if review_vals.notna().any():
        review_floor = st.sidebar.number_input("Minimum reviews", min_value=0, value=int(max(0, review_vals.min() or 0)), step=10)
        filtered = filtered[review_vals >= review_floor]

if category_col:
    cat_search = st.sidebar.text_input("Category contains", value="")
    if cat_search.strip():
        filtered = filtered[clean_text_series(filtered[category_col]).str.contains(cat_search.strip(), case=False, na=False)]

# ---------- hero ----------
st.markdown("""
<h1 style='font-size:48px; margin-bottom:0;'>Restaurant Market Intelligence</h1>
<p style='font-size:18px; color:#9ca3af; margin-top:0;'>A data-driven comparison of restaurant ecosystems across cities — designed for decision-making and storytelling.</p>
""", unsafe_allow_html=True)

st.markdown("""
<div style='display:flex; gap:10px; margin-bottom:10px;'>
<span style='background:#1f2937; padding:6px 12px; border-radius:999px; font-size:12px;'>📊 Data Analytics</span>
<span style='background:#1f2937; padding:6px 12px; border-radius:999px; font-size:12px;'>🏙️ City Comparison</span>
<span style='background:#1f2937; padding:6px 12px; border-radius:999px; font-size:12px;'>⭐ Ratings & Reviews</span>
</div>
""", unsafe_allow_html=True)


st.title("Restaurant Market Intelligence")
st.write("A polished dashboard for comparing restaurant scenes across cities, ratings, reviews, and cuisine categories.")

with st.expander("What this dashboard shows"):
    st.markdown(
        """
        - City-level restaurant volume
        - Rating and review activity
        - Top cuisine categories
        - A filtered table for deeper discussion
        """
    )

# ---------- KPIs ----------
metric_cols = st.columns(4)
metric_cols[0].metric("Restaurants", f"{len(filtered):,}")

if rating_series is not None and not filtered.empty:
    avg_rating = rating_series.loc[filtered.index].mean()
    metric_cols[1].metric("Average rating", f"{avg_rating:.2f}" if pd.notna(avg_rating) else "N/A")
else:
    metric_cols[1].metric("Average rating", "N/A")

if reviews_col and reviews_col in filtered.columns:
    total_reviews = pd.to_numeric(filtered[reviews_col], errors="coerce").sum()
    metric_cols[2].metric("Total reviews", f"{int(total_reviews):,}")
else:
    metric_cols[2].metric("Total reviews", "N/A")

if category_col and category_col in filtered.columns and not filtered.empty:
    metric_cols[3].metric("Top category", clean_text_series(filtered[category_col]).value_counts().index[0])
else:
    metric_cols[3].metric("Top category", "N/A")

st.caption(f"Data source: {source_name}")

# ---------- insights ----------
st.subheader("Key Insights")
ins_cols = st.columns(2)
for i, line in enumerate(insight_lines(filtered, city_col, rating_series, reviews_col, category_col)):
    with ins_cols[i % 2]:
        st.markdown(f"**{i + 1}.** {line}")

# ---------- charts ----------
left, right = st.columns((1.1, 0.9))

with left:
    st.subheader("City comparison")
    if city_col and city_col in filtered.columns and not filtered.empty:
        city_counts = clean_text_series(filtered[city_col]).value_counts().reset_index()
        city_counts.columns = ["City", "Restaurants"]
        fig = px.bar(city_counts, x="City", y="Restaurants", text="Restaurants")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("City data is unavailable.")

with right:
    st.subheader("Rating distribution")
    if rating_series is not None and not filtered.empty:
        rating_data = rating_series.loc[filtered.index].dropna()
        fig = px.histogram(rating_data, nbins=20)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Rating data is unavailable.")

bottom_left, bottom_right = st.columns((1, 1))

with bottom_left:
    st.subheader("Top categories")
    if category_col and category_col in filtered.columns and not filtered.empty:
        category_counts = clean_text_series(filtered[category_col]).value_counts().head(10).reset_index()
        category_counts.columns = ["Category", "Count"]
        fig = px.bar(category_counts.sort_values("Count", ascending=True), x="Count", y="Category", orientation="h")
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Category data is unavailable.")

with bottom_right:
    st.subheader("Review activity")
    if reviews_col and reviews_col in filtered.columns:
        review_data = pd.to_numeric(filtered[reviews_col], errors="coerce").dropna()
        if not review_data.empty:
            fig = px.box(review_data)
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Review-count data is unavailable.")
    else:
        st.info("Review-count data is unavailable.")

# ---------- table ----------
st.subheader("Filtered restaurant table")
display_cols = [c for c in [name_col, city_col, category_col, rating_col, reviews_col, postal_col, address_col] if c]
if display_cols:
    st.dataframe(filtered.loc[:, display_cols], use_container_width=True, height=420)
else:
    st.info("No displayable columns found in the dataset.")

st.download_button(
    "Download filtered data",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_restaurant_data.csv",
    mime="text/csv",
)

st.divider()
st.caption("Built for a live class presentation: clean filters, fast comparison, and a simple story your group can explain clearly.")
