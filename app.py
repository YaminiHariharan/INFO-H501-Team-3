from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


DATA_PATH = Path("data/processed/indy_philly_with_coords.json")

ZIP_COORDS = {
    "19103": (39.9507, -75.1709),
    "19104": (39.9617, -75.1981),
    "19106": (39.9512, -75.1421),
    "19107": (39.9495, -75.1498),
    "19111": (40.0611, -75.0840),
    "19114": (40.0820, -75.0270),
    "19120": (40.0350, -75.1250),
    "19121": (39.9718, -75.1600),
    "19123": (39.9660, -75.1424),
    "19124": (40.0151, -75.0875),
    "19125": (39.9782, -75.1264),
    "19126": (40.0521, -75.1274),
    "19127": (40.0266, -75.2273),
    "19128": (40.0390, -75.2150),
    "19130": (39.9684, -75.1742),
    "19134": (39.9890, -75.1070),
    "19139": (39.9564, -75.2254),
    "19143": (39.9487, -75.2158),
    "19146": (39.9457, -75.1748),
    "19147": (39.9403, -75.1497),
    "19151": (39.9850, -75.2469),
    "19153": (39.8773, -75.2437),
    "19154": (40.0884, -74.9650),
    "46201": (39.7809, -86.1245),
    "46202": (39.7900, -86.1550),
    "46204": (39.7684, -86.1581),
    "46205": (39.8230, -86.1450),
    "46217": (39.6744, -86.1510),
    "46220": (39.8709, -86.1422),
    "46221": (39.7030, -86.2360),
    "46226": (39.8430, -86.0320),
    "46227": (39.6655, -86.1275),
    "46229": (39.7990, -85.9920),
    "46237": (39.6934, -86.0798),
    "46240": (39.9121, -86.1161),
    "46241": (39.7149, -86.2975),
    "46250": (39.9012, -86.0803),
    "46254": (39.8318, -86.2404),
    "46268": (39.8829, -86.2436),
    "46278": (39.8805, -86.2714),
    "46280": (39.9411, -86.1464),
}

CITY_LABELS = {
    "Philadelphia": "Philadelphia, PA",
    "Indianapolis": "Indianapolis, IN",
}


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
            background:
                radial-gradient(circle at 15% 10%, rgba(59,130,246,0.16), transparent 22%),
                radial-gradient(circle at 85% 15%, rgba(168,85,247,0.14), transparent 20%),
                linear-gradient(180deg, #050816 0%, #0b1020 42%, #0f172a 100%);
            color: #f8fafc;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.04));
            border: 1px solid rgba(255, 255, 255, 0.10);
            padding: 16px 18px;
            border-radius: 20px;
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.24);
            backdrop-filter: blur(8px);
        }

        [data-testid="stMetric"] label {
            color: rgba(226, 232, 240, 0.72) !important;
        }

        [data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .hero-wrap {
            background: linear-gradient(
                135deg,
                rgba(59, 130, 246, 0.22),
                rgba(168, 85, 247, 0.18),
                rgba(16, 185, 129, 0.10)
            );
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 26px;
            padding: 22px 24px;
            box-shadow: 0 20px 54px rgba(0, 0, 0, 0.28);
            margin-bottom: 14px;
        }

        .soft-card {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.04));
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 20px;
            padding: 16px 18px;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.18);
            backdrop-filter: blur(10px);
        }

        .pill {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
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
    </style>
    """,
    unsafe_allow_html=True,
)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Missing data file: {DATA_PATH}")
        st.stop()

    try:
        return pd.read_json(DATA_PATH, lines=True)
    except ValueError:
        return pd.read_json(DATA_PATH)


def find_col(df: pd.DataFrame, keywords: list[str]) -> Optional[str]:
    for col in df.columns:
        lower = col.lower()
        if any(keyword in lower for keyword in keywords):
            return col
    return None


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def safe_num(value: object) -> Optional[float]:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def fallback_image(name: str, city: str, category: str) -> Image.Image:
    width, height = 1200, 800
    img = Image.new("RGB", (width, height), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        (30, 30, width - 30, height - 30),
        radius=40,
        outline=(59, 130, 246),
        width=8,
    )
    draw.rounded_rectangle(
        (70, 70, width - 70, height - 70),
        radius=36,
        outline=(168, 85, 247),
        width=4,
    )

    try:
        title_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            64,
        )
        body_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            34,
        )
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    def wrap(text: str, limit: int = 28) -> str:
        return "\n".join(text[i : i + limit] for i in range(0, len(text), limit))

    draw.text((110, 140), wrap(name, 26), fill=(248, 250, 252), font=title_font)
    draw.text((110, 380), city, fill=(226, 232, 240), font=body_font)
    draw.text((110, 440), category, fill=(191, 219, 254), font=body_font)
    draw.text((110, 520), "Restaurant Finder", fill=(125, 211, 252), font=body_font)
    return img


def distance_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_miles = 3958.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_miles * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def score_badges(row: pd.Series) -> list[str]:
    badges: list[str] = []
    for label, col in [
        ("funny", "avg_funny"),
        ("cool", "avg_cool"),
        ("useful", "avg_useful"),
    ]:
        if col in row.index:
            value = safe_num(row.get(col))
            if value is not None and value > 0:
                badges.append(f"{label}: {value:.2f}")
    return badges


class RestaurantFinder:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def filter(
        self,
        *,
        search: str,
        city: str,
        category: str,
        min_rating: float,
        min_reviews: int,
        center_zip: str,
        radius_miles: float,
        compare_mode: str,
        name_col: Optional[str],
        city_col: Optional[str],
        category_col: Optional[str],
        rating_col: Optional[str],
        review_col: Optional[str],
        zip_col: Optional[str],
        lat_col: Optional[str],
        lon_col: Optional[str],
    ) -> pd.DataFrame:
        out = self.df.copy()

        if search.strip():
            query = search.strip().lower()
            mask = pd.Series(False, index=out.index)
            for col in [name_col, city_col, category_col]:
                if col and col in out.columns:
                    mask |= clean_text(out[col]).str.contains(query, case=False, na=False)
            out = out[mask]

        if city_col and city != "All":
            out = out[clean_text(out[city_col]) == city]

        if category_col and category != "All":
            out = out[
                clean_text(out[category_col]).str.contains(category, case=False, na=False)
            ]

        if rating_col and rating_col in out.columns:
            out = out[pd.to_numeric(out[rating_col], errors="coerce").fillna(-1) >= min_rating]

        if review_col and review_col in out.columns:
            out = out[pd.to_numeric(out[review_col], errors="coerce").fillna(-1) >= min_reviews]

        if compare_mode == "Nearby only":
            if (
                center_zip in ZIP_COORDS
                and lat_col
                and lon_col
                and lat_col in out.columns
                and lon_col in out.columns
            ):
                center_lat, center_lon = ZIP_COORDS[center_zip]
                lat_vals = pd.to_numeric(out[lat_col], errors="coerce")
                lon_vals = pd.to_numeric(out[lon_col], errors="coerce")

                distances = []
                for lat, lon in zip(lat_vals, lon_vals):
                    if pd.notna(lat) and pd.notna(lon):
                        distances.append(distance_miles(center_lat, center_lon, lat, lon))
                    else:
                        distances.append(None)

                keep = [
                    distance is not None and distance <= radius_miles
                    for distance in distances
                ]
                out = out.loc[keep].copy()
                out["distance_miles"] = [
                    distance
                    for distance in distances
                    if distance is not None and distance <= radius_miles
                ]

        return out

    def top_picks(
        self,
        df: pd.DataFrame,
        *,
        rating_col: Optional[str],
        review_col: Optional[str],
        limit: int = 6,
    ) -> pd.DataFrame:
        out = df.copy()
        if rating_col and rating_col in out.columns:
            out["__rating__"] = pd.to_numeric(out[rating_col], errors="coerce")
        if review_col and review_col in out.columns:
            out["__reviews__"] = pd.to_numeric(out[review_col], errors="coerce")

        sort_cols = []
        ascending = []
        if "__rating__" in out.columns:
            sort_cols.append("__rating__")
            ascending.append(False)
        if "__reviews__" in out.columns:
            sort_cols.append("__reviews__")
            ascending.append(False)

        if sort_cols:
            out = out.sort_values(sort_cols, ascending=ascending, na_position="last")
        return out.head(limit)


df = load_data()

name_col = find_col(df, ["name", "business_name", "restaurant_name"])
city_col = find_col(df, ["city", "city_name", "town", "municipality"])
category_col = find_col(df, ["categories", "category", "cuisine", "cuisines", "primary_category"])
rating_col = find_col(df, ["avg_stars", "stars", "rating", "average_rating", "avg_rating"])
review_col = find_col(df, ["total_reviews", "review_count", "reviews", "num_reviews"])
lat_col = "latitude" if "latitude" in df.columns else find_col(df, ["latitude", "lat"])
lon_col = "longitude" if "longitude" in df.columns else find_col(df, ["longitude", "lon", "lng"])
zip_col = find_col(df, ["postal_code", "zip", "zipcode", "zip_code"])
comment_col = find_col(df, ["comment", "comments", "review_text", "text", "description", "snippet", "tip", "note"])

if name_col is None:
    name_col = df.columns[0]

finder = RestaurantFinder(df)

st.sidebar.title("Quick Look")
st.sidebar.caption("Show a restaurant in under 30 seconds.")

compare_mode = st.sidebar.radio(
    "View mode",
    ["Compare cities", "Nearby only"],
    index=0,
)

search = st.sidebar.text_input("Search restaurant / keyword", value="")

city_options = ["All"] + sorted(
    clean_text(df[city_col]).replace("", pd.NA).dropna().unique().tolist()
) if city_col else ["All"]
chosen_city = st.sidebar.selectbox(
    "City",
    city_options,
    disabled=compare_mode == "Compare cities",
)
if compare_mode == "Compare cities":
    chosen_city = "All"

category_options = ["All"] + sorted(
    clean_text(df[category_col]).replace("", pd.NA).dropna().unique().tolist()
) if category_col else ["All"]
chosen_category = st.sidebar.selectbox("Category", category_options)

min_rating = st.sidebar.slider(
    "Minimum star rating",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.1,
)
min_reviews = st.sidebar.number_input(
    "Minimum reviews",
    min_value=0,
    value=0,
    step=10,
)

zip_options = [
    z
    for z in ZIP_COORDS.keys()
    if zip_col and zip_col in df.columns
    and z in set(df[zip_col].astype(str).str.slice(0, 5).unique())
]
if not zip_options:
    zip_options = list(ZIP_COORDS.keys())

center_zip = st.sidebar.selectbox(
    "Center ZIP code",
    zip_options,
    disabled=compare_mode == "Compare cities",
)

radius = st.sidebar.slider(
    "Radius (miles)",
    min_value=1,
    max_value=20,
    value=5,
    disabled=compare_mode == "Compare cities",
)

max_results = st.sidebar.slider(
    "Cards shown",
    min_value=3,
    max_value=12,
    value=6,
    step=1,
)

filtered = finder.filter(
    search=search,
    city=chosen_city,
    category=chosen_category,
    min_rating=min_rating,
    min_reviews=min_reviews,
    center_zip=center_zip,
    radius_miles=radius,
    compare_mode=compare_mode,
    name_col=name_col,
    city_col=city_col,
    category_col=category_col,
    rating_col=rating_col,
    review_col=review_col,
    zip_col=zip_col,
    lat_col=lat_col,
    lon_col=lon_col,
)

picks = finder.top_picks(
    filtered,
    rating_col=rating_col,
    review_col=review_col,
    limit=max_results,
)

st.markdown(
    """
    <div class='hero-wrap'>
      <h1 style='font-size:46px; margin-bottom:0; color:#f8fafc;'>Restaurant Market Intelligence</h1>
      <p style='font-size:17px; color:#dbeafe; margin-top:8px; max-width:980px;'>
      Compare restaurants instantly using ratings, reviews, and location insights.
      </p>
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

metric_cols = st.columns(4)
metric_cols[0].metric("Restaurants", f"{len(filtered):,}")
if rating_col and not filtered.empty:
    average_rating = pd.to_numeric(filtered[rating_col], errors="coerce").mean()
    metric_cols[1].metric(
        "Average rating",
        f"{average_rating:.2f}" if pd.notna(average_rating) else "N/A",
    )
else:
    metric_cols[1].metric("Average rating", "N/A")
if review_col and not filtered.empty:
    total_reviews = pd.to_numeric(filtered[review_col], errors="coerce").sum()
    metric_cols[2].metric(
        "Total reviews",
        f"{int(total_reviews):,}" if pd.notna(total_reviews) else "N/A",
    )
else:
    metric_cols[2].metric("Total reviews", "N/A")
metric_cols[3].metric(
    "Cities",
    f"{clean_text(filtered[city_col]).nunique():,}" if city_col else "N/A",
)

st.caption("Data source: data/processed/indy_philly_with_coords.json")

st.markdown("<div class='section-title'>Top Picks</div>", unsafe_allow_html=True)
st.caption("Quick recommendations sorted by quality and popularity.")

if picks.empty:
    st.info("No restaurants match those filters. Try widening the search.")
else:
    pick_cols = st.columns(3)
    for index, (_, row) in enumerate(picks.iterrows()):
        with pick_cols[index % 3]:
            st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
            st.image(
                fallback_image(
                    name=str(row.get(name_col, "Restaurant")),
                    city=str(row.get(city_col, "City")) if city_col else "City",
                    category=str(row.get(category_col, "restaurant")),
                ),
                use_container_width=True,
            )
            st.write(f"**{row.get(name_col, 'N/A')}**")
            if city_col:
                st.write(f"📍 {row.get(city_col, 'N/A')}")
            if category_col:
                st.write(f"🏷️ {row.get(category_col, 'N/A')}")
            if rating_col:
                rating_value = safe_num(row.get(rating_col))
                st.write(f"⭐ {rating_value:.2f}" if rating_value is not None else "⭐ N/A")
            if review_col:
                review_value = safe_num(row.get(review_col))
                st.write(
                    f"💬 {int(review_value):,} reviews"
                    if review_value is not None
                    else "💬 N/A"
                )
            badges = score_badges(row)
            if badges:
                st.markdown(
                    " ".join([f"<span class='pill'>✨ {badge}</span>" for badge in badges]),
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Restaurant Profile</div>", unsafe_allow_html=True)
st.caption("One-click view for a client who wants to decide quickly.")

if filtered.empty:
    st.warning("No restaurants match the current filters.")
else:
    selected_name = st.selectbox(
        "Choose a restaurant",
        filtered[name_col].dropna().astype(str).unique(),
    )
    selected = filtered[filtered[name_col].astype(str) == str(selected_name)].iloc[0]

    left, right = st.columns([0.44, 0.56])
    with left:
        st.image(
            fallback_image(
                name=str(selected.get(name_col, "Restaurant")),
                city=str(selected.get(city_col, "City")) if city_col else "City",
                category=str(selected.get(category_col, "restaurant")),
            ),
            use_container_width=True,
        )
        if lat_col and lon_col and lat_col in selected.index and lon_col in selected.index:
            latitude = safe_num(selected.get(lat_col))
            longitude = safe_num(selected.get(lon_col))
            if latitude is not None and longitude is not None and center_zip in ZIP_COORDS:
                center_lat, center_lon = ZIP_COORDS[center_zip]
                distance = distance_miles(center_lat, center_lon, latitude, longitude)
                st.markdown(
                    f"<span class='pill'>📏 {distance:.1f} miles away</span>",
                    unsafe_allow_html=True,
                )

    with right:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.write(f"### {selected.get(name_col, 'N/A')}")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("City", selected.get(city_col, "N/A") if city_col else "N/A")
        d2.metric(
            "Category",
            selected.get(category_col, "N/A") if category_col else "N/A",
        )
        if rating_col:
            rating_value = safe_num(selected.get(rating_col))
            d3.metric(
                "Rating",
                f"{rating_value:.2f}" if rating_value is not None else "N/A",
            )
        else:
            d3.metric("Rating", "N/A")
        if review_col:
            review_value = safe_num(selected.get(review_col))
            d4.metric(
                "Reviews",
                f"{int(review_value):,}" if review_value is not None else "N/A",
            )
        else:
            d4.metric("Reviews", "N/A")

        c1, c2 = st.columns(2)
        with c1:
            if lat_col and lat_col in selected.index:
                st.write(f"**Latitude:** {selected.get(lat_col)}")
            if lon_col and lon_col in selected.index:
                st.write(f"**Longitude:** {selected.get(lon_col)}")
        with c2:
            if zip_col and zip_col in selected.index:
                st.write(f"**ZIP:** {selected.get(zip_col)}")
            st.write(f"**Center ZIP:** {center_zip}")

        st.markdown("#### What people are saying")
        if comment_col and comment_col in selected.index and str(selected.get(comment_col, "")).strip():
            st.write(str(selected.get(comment_col, ""))[:500])
        else:
            st.write(
                "No comment text is available in this dataset, so ratings and review counts are the main signals."
            )

        badges = score_badges(selected)
        if badges:
            st.markdown(
                " ".join([f"<span class='pill'>✨ {badge}</span>" for badge in badges]),
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Live Map View</div>", unsafe_allow_html=True)
st.caption("Interactive maps using real latitude/longitude coordinates.")

if lat_col and lon_col and lat_col in filtered.columns and lon_col in filtered.columns:
    map_df = filtered.copy()
    map_df[lat_col] = pd.to_numeric(map_df[lat_col], errors="coerce")
    map_df[lon_col] = pd.to_numeric(map_df[lon_col], errors="coerce")
    map_df = map_df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

    if map_df.empty:
        st.info("No valid latitude/longitude values after filtering.")
    else:
        overview = px.scatter_mapbox(
            map_df,
            lat=lat_col,
            lon=lon_col,
            hover_name=name_col,
            hover_data={
                city_col: True if city_col else False,
                category_col: True if category_col else False,
                rating_col: True if rating_col else False,
                review_col: True if review_col else False,
            },
            color=city_col if city_col else None,
            zoom=8.5,
            height=500,
        )
        overview.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=10, r=10, t=35, b=10),
        )
        st.plotly_chart(overview, use_container_width=True)

        philly = (
            map_df[clean_text(map_df[city_col]) == "Philadelphia"]
            if city_col
            else pd.DataFrame()
        )
        indy = (
            map_df[clean_text(map_df[city_col]) == "Indianapolis"]
            if city_col
            else pd.DataFrame()
        )

        city_left, city_right = st.columns(2)
        with city_left:
            st.subheader(CITY_LABELS["Philadelphia"])
            if not philly.empty:
                fig = px.scatter_mapbox(
                    philly,
                    lat=lat_col,
                    lon=lon_col,
                    hover_name=name_col,
                    hover_data={
                        category_col: True if category_col else False,
                        rating_col: True if rating_col else False,
                        review_col: True if review_col else False,
                    },
                    zoom=10,
                    height=420,
                )
                fig.update_layout(
                    mapbox_style="open-street-map",
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Philadelphia restaurants are in the current filtered results.")

        with city_right:
            st.subheader(CITY_LABELS["Indianapolis"])
            if not indy.empty:
                fig = px.scatter_mapbox(
                    indy,
                    lat=lat_col,
                    lon=lon_col,
                    hover_name=name_col,
                    hover_data={
                        category_col: True if category_col else False,
                        rating_col: True if rating_col else False,
                        review_col: True if review_col else False,
                    },
                    zoom=10,
                    height=420,
                )
                fig.update_layout(
                    mapbox_style="open-street-map",
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Indianapolis restaurants are in the current filtered results.")
else:
    st.info("Latitude/longitude columns not found in the loaded file.")

left_chart, right_chart = st.columns((1.15, 0.85))

with left_chart:
    st.subheader("Restaurants by City")
    if city_col and city_col in filtered.columns and not filtered.empty:
        counts = clean_text(filtered[city_col]).value_counts().reset_index()
        counts.columns = ["City", "Count"]
        fig = px.bar(counts, x="City", y="Count", text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("City data is unavailable.")

with right_chart:
    st.subheader("Star Ratings")
    if rating_col and rating_col in filtered.columns:
        rating_data = pd.to_numeric(filtered[rating_col], errors="coerce").dropna()
        if not rating_data.empty:
            fig = px.histogram(rating_data, nbins=20)
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Rating data is unavailable.")
    else:
        st.info("Rating data is unavailable.")

st.subheader("Results")
show_cols = [
    c
    for c in [name_col, city_col, category_col, rating_col, review_col, lat_col, lon_col]
    if c
]
if show_cols:
    st.dataframe(filtered.loc[:, show_cols].head(20), use_container_width=True, height=340)
else:
    st.dataframe(filtered.head(20), use_container_width=True, height=340)

st.download_button(
    "Download filtered data",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="restaurant_results.csv",
    mime="text/csv",
)

st.caption(
    "Built for a fast client walkthrough: search, choose, review stars, read scores, and decide quickly."
)
