from restaurant_utils import RestaurantFinder, clean_text, extract_badges, fallback_image, infer_comment_col, infer_rating_col, infer_review_col, find_col
finder = RestaurantFinder(df, ZIP_COORDS)
filtered = finder.filter(
    search=search,
    city=chosen_city,
    category=chosen_category,
    min_rating=min_rating,
    min_reviews=min_reviews,
    user_zip=user_zip,
    radius_miles=radius,
    name_col=name_col,
    city_col=city_col,
    category_col=category_col,
    rating_col=rating_col,
    review_col=review_col,
    zip_col=zip_col,
    comment_col=comment_col,
    address_col=address_col,
)

picks = finder.top_picks(
    filtered,
    name_col=name_col,
    city_col=city_col,
    category_col=category_col,
    rating_col=rating_col,
    review_col=review_col,
    limit=max_results,
)

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="Restaurant Finder",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 10%, rgba(59,130,246,0.16), transparent 22%),
                radial-gradient(circle at 85% 15%, rgba(168,85,247,0.14), transparent 20%),
                linear-gradient(180deg, #050816 0%, #0b1020 42%, #0f172a 100%);
            color: #f8fafc;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
            border: 1px solid rgba(255,255,255,0.10);
            padding: 16px 18px;
            border-radius: 20px;
            box-shadow: 0 14px 36px rgba(0,0,0,0.24);
            backdrop-filter: blur(8px);
        }
        [data-testid="stMetric"] label {
            color: rgba(226,232,240,0.72) !important;
        }
        [data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .hero-wrap {
            background: linear-gradient(135deg, rgba(59,130,246,0.22), rgba(168,85,247,0.18), rgba(16,185,129,0.10));
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 26px;
            padding: 22px 24px;
            box-shadow: 0 20px 54px rgba(0,0,0,0.28);
            margin-bottom: 14px;
        }
        .soft-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 20px;
            padding: 16px 18px;
            box-shadow: 0 14px 32px rgba(0,0,0,0.18);
            backdrop-filter: blur(10px);
        }
        .pill {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            color: #f8fafc;
            padding: 7px 12px;
            border-radius: 999px;
            font-size: 12px;
            margin-right: 8px;
            display: inline-block;
            margin-bottom: 8px;
        }
        .section-title {
            font-size: 20px;
            font-weight: 700;
            margin: 0.2rem 0 0.4rem 0;
        }
        .muted-subtitle {
            color: rgba(226,232,240,0.75);
            margin-top: 0.1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

ZIP_COORDS = {
    "19107": (39.9495, -75.1498),
    "19106": (39.9512, -75.1421),
    "19147": (39.9403, -75.1497),
    "19127": (40.0266, -75.2273),
    "19123": (39.9660, -75.1424),
    "19104": (39.9617, -75.1981),
    "19130": (39.9684, -75.1742),
    "19124": (40.0151, -75.0875),
    "19125": (39.9782, -75.1264),
    "46227": (39.6655, -86.1275),
    "46250": (39.9012, -86.0803),
    "46220": (39.8679, -86.1240),
    "46201": (39.7710, -86.0950),
    "46240": (39.9121, -86.1161),
    "46204": (39.7684, -86.1581),
}


# ---------- helpers ----------
def load_data() -> pd.DataFrame:
    candidates = [
        Path("data/processed/indy_philly_restaurant_aggregated.json"),
        Path("data/processed/restaurants.csv"),
        Path("data/processed/restaurant_data.csv"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".json":
            try:
                return pd.read_json(path, lines=True)
            except ValueError:
                return pd.read_json(path)
    st.error("No dataset found in data/processed/.")
    st.stop()


def find_col(df: pd.DataFrame, keywords: list[str]) -> Optional[str]:
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in keywords):
            return col
    return None


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


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


def infer_rating_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["rating", "stars", "avg_stars", "average_rating", "avg_rating"])


def infer_review_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["review_count", "reviews", "total_reviews", "num_reviews"])


def infer_comment_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["comment", "comments", "review_text", "text", "description", "snippet", "tip", "note"])


def distance_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def restaurant_distance(row: pd.Series, zip_col: Optional[str], user_zip: str) -> Optional[float]:
    if not zip_col or zip_col not in row.index:
        return None
    user_coords = ZIP_COORDS.get(user_zip)
    if not user_coords:
        return None
    z = str(row.get(zip_col, "")).strip()[:5]
    if z not in ZIP_COORDS:
        return None
    user_lat, user_lon = user_coords
    lat, lon = ZIP_COORDS[z]
    return distance_miles(user_lat, user_lon, lat, lon)


def safe_num(value: object) -> Optional[float]:
    n = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(n) else float(n)


# ---------- load + schema ----------
df = load_data()
name_col = find_col(df, ["restaurant_name", "business_name", "name"])
city_col = find_col(df, ["city", "city_name", "town", "municipality"])
category_col = find_col(df, ["category", "categories", "cuisine", "cuisines", "primary_category"])
rating_col = infer_rating_col(df)
review_col = infer_review_col(df)
comment_col = infer_comment_col(df)
photo_col = find_col(df, ["image_url", "photo_url", "photo", "image", "img", "picture", "thumbnail"])
address_col = find_col(df, ["address", "full_address", "location"])
zip_col = find_col(df, ["postal_code", "zip", "zipcode", "zip_code"])

rating_series = pd.to_numeric(df[rating_col], errors="coerce") if rating_col else None
review_series = pd.to_numeric(df[review_col], errors="coerce") if review_col else None


# ---------- sidebar ----------
st.sidebar.title("Quick Look")
st.sidebar.caption("Show a restaurant in under 30 seconds.")

search = st.sidebar.text_input("Search restaurant / keyword", value="")
chosen_city = st.sidebar.selectbox(
    "City",
    ["All"] + sorted(clean_text(df[city_col]).replace("", pd.NA).dropna().unique().tolist()) if city_col else ["All"],
)
chosen_category = st.sidebar.selectbox(
    "Category",
    ["All"] + sorted(clean_text(df[category_col]).replace("", pd.NA).dropna().unique().tolist()) if category_col else ["All"],
)
min_rating = st.sidebar.slider("Minimum star rating", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
min_reviews = st.sidebar.number_input("Minimum reviews", min_value=0, value=0, step=10)

zip_options = list(ZIP_COORDS.keys())
if zip_col and zip_col in df.columns:
    present_zips = set(df[zip_col].astype(str).str.slice(0, 5).tolist())
    zip_options = [z for z in ZIP_COORDS.keys() if z in present_zips] or list(ZIP_COORDS.keys())
user_zip = st.sidebar.selectbox("Your ZIP code", zip_options)
radius = st.sidebar.slider("Radius (miles)", min_value=1, max_value=20, value=5)
max_results = st.sidebar.slider("Cards shown", min_value=3, max_value=12, value=6, step=1)

filtered = df.copy()

if search.strip():
    q = search.strip().lower()
    mask = pd.Series(False, index=filtered.index)
    for col in [name_col, city_col, category_col, address_col, zip_col, comment_col]:
        if col and col in filtered.columns:
            mask |= clean_text(filtered[col]).str.contains(q, case=False, na=False)
    filtered = filtered[mask]

if city_col and chosen_city != "All":
    filtered = filtered[clean_text(filtered[city_col]) == chosen_city]

if category_col and chosen_category != "All":
    filtered = filtered[clean_text(filtered[category_col]) == chosen_category]

if rating_col:
    filtered = filtered[pd.to_numeric(filtered[rating_col], errors="coerce").fillna(-1) >= min_rating]

if review_col:
    filtered = filtered[pd.to_numeric(filtered[review_col], errors="coerce").fillna(-1) >= min_reviews]

if zip_col and zip_col in filtered.columns and user_zip in ZIP_COORDS:
    user_lat, user_lon = ZIP_COORDS[user_zip]

    def in_radius(row: pd.Series) -> bool:
        z = str(row.get(zip_col, "")).strip()[:5]
        if z in ZIP_COORDS:
            lat, lon = ZIP_COORDS[z]
            return distance_miles(user_lat, user_lon, lat, lon) <= radius
        return False

    filtered = filtered[filtered.apply(in_radius, axis=1)]


# ---------- hero ----------
st.markdown(
    """
    <div class='hero-wrap'>
      <h1 style='font-size:46px; margin-bottom:0; color:#f8fafc;'>Restaurant Finder</h1>
      <p style='font-size:17px; color:#dbeafe; margin-top:8px; max-width:980px;'>Show a client a restaurant in under 30 seconds: where it is, what it is like, and what people say about it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div>
      <span class='pill'>📍 location</span>
      <span class='pill'>⭐ star ratings</span>
      <span class='pill'>💬 reviews</span>
      <span class='pill'>✨ funny • cool • useful</span>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Restaurants", f"{len(filtered):,}")
m2.metric("Average rating", f"{rating_series.loc[filtered.index].mean():.2f}" if rating_series is not None and not filtered.empty and rating_series.loc[filtered.index].notna().any() else "N/A")
m3.metric("Total reviews", f"{int(review_series.loc[filtered.index].sum()):,}" if review_series is not None and not filtered.empty and review_series.loc[filtered.index].notna().any() else "N/A")
m4.metric("Cities", f"{clean_text(filtered[city_col]).nunique():,}" if city_col else "N/A")


# ---------- top picks ----------
st.markdown("<div class='section-title'>Top Picks</div>", unsafe_allow_html=True)
st.caption("Quick recommendations sorted by quality and popularity.")

picks = filtered.copy()
if rating_col:
    picks["__rating__"] = pd.to_numeric(picks[rating_col], errors="coerce")
if review_col:
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
picks = picks.head(max_results)

if picks.empty:
    st.info("No restaurants match those filters. Try widening the search.")
else:
    cards = st.columns(3)
    for i, (_, row) in enumerate(picks.iterrows()):
        with cards[i % 3]:
            st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
            image_url = str(row.get(photo_col)) if photo_col and photo_col in row.index and pd.notna(row.get(photo_col)) and str(row.get(photo_col)).startswith("http") else fallback_image(row.get(category_col, "restaurant"))
            st.image(image_url, use_container_width=True)
            st.write(f"**{row.get(name_col, 'N/A') if name_col else 'N/A'}**")
            if city_col:
                st.write(f"📍 {row.get(city_col, 'N/A')}")
            if category_col:
                st.write(f"🏷️ {row.get(category_col, 'N/A')}")
            if rating_col:
                r = safe_num(row.get(rating_col))
                st.write(f"⭐ {r:.2f}" if r is not None else "⭐ N/A")
            if review_col:
                rv = safe_num(row.get(review_col))
                st.write(f"💬 {int(rv):,} reviews" if rv is not None else "💬 N/A")
            badges = extract_badges(row, comment_col)
            if badges:
                st.markdown(" ".join([f"<span class='pill'>✨ {b}</span>" for b in badges]), unsafe_allow_html=True)
            else:
                st.markdown("<span class='pill'>✨ fresh pick</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ---------- restaurant profile ----------
st.markdown("<div class='section-title'>Restaurant Profile</div>", unsafe_allow_html=True)
st.caption("One-click view for a client who wants to decide quickly.")

if name_col and name_col in filtered.columns and not filtered.empty:
    selected_name = st.selectbox("Choose a restaurant", filtered[name_col].dropna().astype(str).unique())
    selected = filtered[filtered[name_col].astype(str) == str(selected_name)].iloc[0]

    left, right = st.columns([0.44, 0.56])
    with left:
        image_url = str(selected.get(photo_col)) if photo_col and photo_col in selected.index and pd.notna(selected.get(photo_col)) and str(selected.get(photo_col)).startswith("http") else fallback_image(selected.get(category_col, "restaurant"))
        st.image(image_url, use_container_width=True)
        distance = restaurant_distance(selected, zip_col, user_zip)
        if distance is not None:
            st.markdown(f"<span class='pill'>📏 {distance:.1f} miles away</span>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.write(f"### {selected.get(name_col, 'N/A')}")
        detail1, detail2, detail3, detail4 = st.columns(4)
        detail1.metric("City", selected.get(city_col, "N/A") if city_col else "N/A")
        detail2.metric("Category", selected.get(category_col, "N/A") if category_col else "N/A")
        if rating_col:
            r = safe_num(selected.get(rating_col))
            detail3.metric("Rating", f"{r:.2f}" if r is not None else "N/A")
        else:
            detail3.metric("Rating", "N/A")
        if review_col:
            rv = safe_num(selected.get(review_col))
            detail4.metric("Reviews", f"{int(rv):,}" if rv is not None else "N/A")
        else:
            detail4.metric("Reviews", "N/A")

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Address:** {selected.get(address_col, 'N/A') if address_col else 'N/A'}")
            st.write(f"**ZIP:** {selected.get(zip_col, 'N/A') if zip_col else 'N/A'}")
        with c2:
            st.write(f"**Search match:** {search if search.strip() else 'All restaurants'}")
            st.write(f"**City filter:** {chosen_city}")
            st.write(f"**Radius:** {radius} miles from {user_zip}")

        st.markdown("#### What people are saying")
        if comment_col and comment_col in selected.index and str(selected.get(comment_col, "")).strip():
            comment_text = str(selected.get(comment_col, ""))
            st.write(comment_text[:500])
        else:
            st.write("No comment text is available in this dataset, so ratings and reviews are the main signals.")

        badges = extract_badges(selected, comment_col)
        if badges:
            st.markdown(" ".join([f"<span class='pill'>✨ {b}</span>" for b in badges]), unsafe_allow_html=True)
        else:
            st.markdown("<span class='pill'>✨ fresh pick</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ---------- map view ----------
st.markdown("<div class='section-title'>Live Map View</div>", unsafe_allow_html=True)
st.caption("Approximate map based on ZIP-code center points. Great for a fast presentation demo.")

map_rows = []
if zip_col and zip_col in filtered.columns:
    for _, row in filtered.head(100).iterrows():
        z = str(row.get(zip_col, "")).strip()[:5]
        if z in ZIP_COORDS:
            lat, lon = ZIP_COORDS[z]
            map_rows.append(
                {
                    "name": str(row.get(name_col, "Restaurant")) if name_col else "Restaurant",
                    "lat": lat,
                    "lon": lon,
                    "rating": safe_num(row.get(rating_col)) if rating_col else None,
                    "city": str(row.get(city_col, "")) if city_col else "",
                    "category": str(row.get(category_col, "")) if category_col else "",
                }
            )

if map_rows and user_zip in ZIP_COORDS:
    map_df = pd.DataFrame(map_rows)
    user_lat, user_lon = ZIP_COORDS[user_zip]

    map_df["size"] = map_df["rating"].fillna(3.5).clip(2.5, 5.0) * 1200
    map_df["color_r"] = (map_df["rating"].fillna(3.5) * 45).astype(int).clip(50, 255)
    map_df["color_g"] = (220 - map_df["rating"].fillna(3.5) * 30).astype(int).clip(40, 220)
    map_df["color_b"] = 255 - (map_df["rating"].fillna(3.5) * 25).astype(int).clip(20, 180)

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_radius="size",
            get_fill_color="[color_r, color_g, color_b, 180]",
            pickable=True,
            auto_highlight=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{ "lat": user_lat, "lon": user_lon }]),
            get_position="[lon, lat]",
            get_radius=1800,
            get_fill_color="[255, 255, 255, 220]",
            pickable=False,
        ),
    ]

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=user_lat, longitude=user_lon, zoom=9, pitch=0),
        tooltip={"text": "{name}\n{city}\n{category}\nRating: {rating}"},
        map_style="mapbox://styles/mapbox/dark-v11",
    )
    st.pydeck_chart(deck, use_container_width=True)
else:
    st.info("No map points available yet. Add ZIP codes to more rows or check the ZIP mapping.")


# ---------- charts ----------
left_chart, right_chart = st.columns((1.15, 0.85))
with left_chart:
    st.markdown("<div class='section-title'>Restaurants by City</div>", unsafe_allow_html=True)
    if city_col and city_col in filtered.columns and not filtered.empty:
        counts = clean_text(filtered[city_col]).value_counts().reset_index()
        counts.columns = ["City", "Count"]
        fig = px.bar(counts, x="City", y="Count", color="Count", color_continuous_scale=["#60a5fa", "#a78bfa"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=360, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("City data is unavailable.")

with right_chart:
    st.markdown("<div class='section-title'>Star Ratings</div>", unsafe_allow_html=True)
    if rating_col and rating_col in filtered.columns:
        rating_data = pd.to_numeric(filtered[rating_col], errors="coerce").dropna()
        if not rating_data.empty:
            fig = px.histogram(rating_data, nbins=20, color_discrete_sequence=["#60a5fa"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=360, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Rating data is unavailable.")
    else:
        st.info("Rating data is unavailable.")


# ---------- results ----------
st.markdown("<div class='section-title'>Results</div>", unsafe_allow_html=True)
show_cols = [c for c in [name_col, city_col, category_col, rating_col, review_col, address_col, zip_col] if c]
if show_cols:
    st.dataframe(filtered.loc[:, show_cols].head(20), use_container_width=True, height=340)
else:
    st.dataframe(filtered.head(20), use_container_width=True, height=340)

st.download_button("Download filtered data", filtered.to_csv(index=False).encode("utf-8"), file_name="restaurant_results.csv", mime="text/csv")

st.caption("Built for a fast client walkthrough: search, choose, review stars, read comments, and decide quickly.")
