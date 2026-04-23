import streamlit as st
import pandas as pd
import pydeck as pdk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.set_page_config(
    page_title="Nom Nom Navigator",
    page_icon="🗺️",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS — Clean & Editorial
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* ── Page shell ── */
[data-testid="stAppViewContainer"] { background-color: #1C1C1A; }
[data-testid="stSidebar"]          { background-color: #141412; }
[data-testid="stSidebar"] section  { padding-top: 1.5rem; }
#MainMenu, footer                  { visibility: hidden; }

/* ── Page header ── */
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 40px;
    font-weight: 700;
    color: #F0EDE6;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 2px;
}
.page-sub {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: #A09D96;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 20px;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Inter', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #B0ADA5;
    padding-bottom: 10px;
    border-bottom: 1px solid #E6E4DE;
    margin-bottom: 16px;
    margin-top: 8px;
}

/* ── Info strip ── */
.info-strip {
    background: #F2EFE8;
    border-left: 3px solid #C8AB84;
    padding: 10px 16px;
    border-radius: 0 4px 4px 0;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: #666;
    margin-bottom: 24px;
    line-height: 1.5;
}

/* ── Restaurant card ── */
.card-rank {
    font-family: 'Playfair Display', serif;
    font-size: 30px;
    font-weight: 700;
    color: #DDD9D0;
    line-height: 1;
    padding-top: 3px;
}
.card-name {
    font-family: 'Playfair Display', serif;
    font-size: 17px;
    font-weight: 600;
    color: #F0EDE6;
    margin: 0 0 6px 0;
    line-height: 1.25;
}
.card-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 2px;
}
.badge {
    font-family: 'Inter', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: #2E2E2B; 
    color: #999;
    padding: 2px 7px;
    border-radius: 2px;
}
.price-badge {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: #4A7C59;
}
.star-badge {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: #C8AB84;
}
.card-address {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: #B0ADA5;
    margin-top: 5px;
}

/* ── Detail panel ── */
.detail-label {
    font-family: 'Inter', sans-serif;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #B0ADA5;
    margin-bottom: 3px;
}
.detail-value {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color:  #D0CCC4;
}
.chip-on {
    display: inline-block;
    background: #EEF5F1;
    color: #3D7A55;
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 500;
    padding: 3px 9px;
    border-radius: 2px;
    margin: 3px 4px 3px 0;
}
.chip-off {
    display: inline-block;
    background: #F4F3F0;
    color: #C8C5BC;
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    padding: 3px 9px;
    border-radius: 2px;
    margin: 3px 4px 3px 0;
    text-decoration: line-through;
}
.cat-chip {
    display: inline-block;
    background: #F2EFE8;
    color: #7A7469;
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 400;
    padding: 3px 9px;
    border-radius: 2px;
    margin: 3px 4px 3px 0;
}

/* ── Sidebar tweaks ── */
.sidebar-heading {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 600;
    color: #1C1C1A;
    margin-bottom: 2px;
}
.sidebar-tagline {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: #A09D96;
    font-style: italic;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load Data & Model
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_json("df_restaurants_only.json", lines=True)
    df['price_range'] = df['price_range'].fillna(2)
    df['categories_list'] = df['categories'].str.split(', ')
    return df

@st.cache_resource
def load_model(df):
    cuisine_dummies = pd.get_dummies(df['cuisine_group'], prefix='cuisine')
    bool_cols = [
        'outdoor_seating', 'reservations', 'delivery', 'takeout',
        'ambience_romantic', 'ambience_casual', 'ambience_classy',
        'good_for_lunch', 'good_for_dinner', 'good_for_brunch',
        'is_vegetarian_friendly', 'is_halal'
    ]
    scaler = MinMaxScaler()
    df['price_norm'] = scaler.fit_transform(df[['price_range']].fillna(2))
    feature_matrix = pd.concat([
        cuisine_dummies,
        df[['price_norm'] + bool_cols].fillna(0).astype(float),
    ], axis=1)
    weight_map = {
        'price_norm': 3.0, 'ambience_classy': 3.0, 'reservations': 3.0,
        'good_for_brunch': 3.0, 'outdoor_seating': 2.0, 'delivery': 2.0,
        'ambience_casual': 2.0, 'good_for_lunch': 2.0, 'good_for_dinner': 2.0,
        'takeout': 1.0, 'ambience_romantic': 1.0,
    }
    for col, weight in weight_map.items():
        if col in feature_matrix.columns:
            feature_matrix[col] *= weight
    for col in cuisine_dummies.columns:
        feature_matrix[col] *= 3.0
    feature_matrix.index = df['business_id']
    sim_matrix = cosine_similarity(feature_matrix.values)
    sim_df = pd.DataFrame(sim_matrix, index=feature_matrix.index, columns=feature_matrix.index)
    return sim_df

df = load_data()
item_sim_df = load_model(df)


# ─────────────────────────────────────────────
# Recommender Functions
# ─────────────────────────────────────────────
def cold_start_recs(city, cuisine_pref=None, price_pref=None, min_rating=1.0, top_n=5):
    pool = df[
        (df['city'] == city) &
        (df['is_open'] == 1) &
        (df['stars'] >= min_rating) &
        (~df['name'].str.contains('Reading Terminal Market', na=False))
    ]
    if cuisine_pref and cuisine_pref != 'Any':
        pool = pool[pool['cuisine_group'] == cuisine_pref]
    if price_pref:
        pool = pool[pool['price_range'] == price_pref]
    pool = pool.sort_values('bayes_score', ascending=False)
    recs, seen = [], set()
    for _, row in pool.iterrows():
        if row['cuisine_group'] not in seen:
            recs.append(row)
            seen.add(row['cuisine_group'])
        if len(recs) == top_n:
            break
    return pd.DataFrame(recs)

def content_based_recs(liked_business_ids, city, top_n=5):
    valid_ids = [b for b in liked_business_ids if b in item_sim_df.index]
    if not valid_ids:
        return None
    sim_scores = item_sim_df[valid_ids].mean(axis=1)
    city_businesses = df[df['city'] == city]['business_id'].values
    sim_scores = sim_scores[sim_scores.index.isin(city_businesses)]
    sim_scores = sim_scores.drop(labels=valid_ids, errors='ignore')
    sim_scores = sim_scores[df.set_index('business_id')['bayes_score'] >= 3.5]
    top_candidates = sim_scores.nlargest(top_n * 10).index.tolist()
    recs, cuisine_counts = [], {}
    for biz_id in top_candidates:
        row = df[df['business_id'] == biz_id].iloc[0]
        cuisine = row['cuisine_group']
        count = cuisine_counts.get(cuisine, 0)
        if count < 2:
            recs.append(row)
            cuisine_counts[cuisine] = count + 1
        if len(recs) == top_n:
            break
    return pd.DataFrame(recs)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
CUISINE_EMOJI = {
    'American': '🍔', 'Italian': '🍕', 'Asian': '🍜',
    'Mexican & Latin': '🌮', 'Casual': '🥪', 'Seafood': '🦞',
    'Steakhouse': '🥩', 'Indian & Middle Eastern': '🫔',
    'Vegetarian & Vegan': '🥗', 'Other': '🍽️',
}

FEATURE_MAP = {
    'outdoor_seating':      '🌿 Outdoor Seating',
    'reservations':         '📅 Reservations',
    'delivery':             '🛵 Delivery',
    'takeout':              '🥡 Takeout',
    'ambience_romantic':    '💑 Romantic',
    'ambience_casual':      '😊 Casual',
    'ambience_classy':      '🎩 Classy',
    'good_for_lunch':       '☀️ Lunch',
    'good_for_dinner':      '🌙 Dinner',
    'good_for_brunch':      '🥞 Brunch',
    'is_vegetarian_friendly': '🥦 Vegetarian Friendly',
    'is_halal':             '🌙 Halal',
}

def price_str(val):
    try:
        return '$' * int(val)
    except Exception:
        return 'N/A'


def render_detail_panel(row):
    score = round(float(row.get('bayes_score', 0)), 2)

    # ── Key metrics row ──
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, "Yelp Stars",  f"★ {row.get('stars', 'N/A')}"),
        (c2, "Bayes Score", str(score)),
        (c3, "Reviews",     f"{int(row.get('review_count', 0)):,}"),
        (c4, "Price",       price_str(row.get('price_range', 2))),
    ]
    for col, label, val in metrics:
        with col:
            st.markdown(
                f"<div class='detail-label'>{label}</div>"
                f"<div class='detail-value'>{val}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr style='border:none;border-top:1px solid #E6E4DE;margin:14px 0;'>", unsafe_allow_html=True)

    # ── Categories ──
    cats = str(row.get('categories', '') or '')
    if cats:
        chips = "".join(
            f"<span class='cat-chip'>{c.strip()}</span>"
            for c in cats.split(',') if c.strip()
        )
        st.markdown(
            "<div class='detail-label' style='margin-bottom:6px;'>Categories</div>"
            + chips,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    # ── Features ──
    chips = ""
    for col, label in FEATURE_MAP.items():
        val = row.get(col, 0)
        is_on = pd.notna(val) and float(val) == 1.0
        css_class = 'chip-on' if is_on else 'chip-off'
        chips += f"<span class='{css_class}'>{label}</span>"

    st.markdown(
        "<div class='detail-label' style='margin-bottom:6px;'>Features</div>" + chips,
        unsafe_allow_html=True,
    )


def render_card(rank, row):
    cuisine  = row.get('cuisine_group', 'Other')
    emoji    = CUISINE_EMOJI.get(cuisine, '🍽️')
    stars    = row.get('stars', 0)
    address  = row.get('address', '')
    city_name= row.get('city', '')
    rank_str = f"{rank:02d}"

    with st.container(border=True):
        left, right = st.columns([1, 11])
        with left:
            st.markdown(f"<div class='card-rank'>{rank_str}</div>", unsafe_allow_html=True)
        with right:
            st.markdown(
                f"<p class='card-name'>{emoji} {row['name']}</p>"
                f"<div class='card-meta'>"
                f"  <span class='badge'>{cuisine}</span>"
                f"  <span class='price-badge'>{price_str(row.get('price_range', 2))}</span>"
                f"  <span class='star-badge'>★ {stars}</span>"
                f"</div>"
                f"<div class='card-address'>📍 {address}, {city_name}</div>",
                unsafe_allow_html=True,
            )

        with st.expander("View full details →"):
            render_detail_panel(row)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-heading'>🗺️ Nom Nom Navigator</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-tagline'>Life's too short for bad restaurants</div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='section-label'>City</div>", unsafe_allow_html=True)
    city = st.selectbox("City", options=sorted(df['city'].unique()), label_visibility="collapsed")

    st.markdown("<div class='section-label' style='margin-top:20px;'>Filters</div>", unsafe_allow_html=True)
    cuisine = st.selectbox(
        "Cuisine",
        options=['Any'] + sorted(df['cuisine_group'].dropna().unique().tolist()),
    )
    price = st.selectbox(
        "Price Range",
        options=[1, 2, 3, 4],
        format_func=lambda x: ['$ — Budget', '$$ — Mid-range', '$$$ — Upscale', '$$$$ — Fine Dining'][x - 1],
    )
    min_rating = st.slider("Minimum Rating", 1.0, 5.0, 3.5, step=0.5)

    st.divider()
    st.markdown("<div class='section-label'>Personalise</div>", unsafe_allow_html=True)
    st.caption("Add 3+ restaurants you love for tailored picks")
    all_restaurants = df[df['city'] == city]['name'].sort_values().unique()
    liked_names = st.multiselect(
        "Liked restaurants",
        options=all_restaurants,
        label_visibility="collapsed",
    )


# ─────────────────────────────────────────────
# Run Recommender
# ─────────────────────────────────────────────
liked_ids = (
    df[df['name'].isin(liked_names)]
    .drop_duplicates('business_id')['business_id']
    .tolist()
)

if len(liked_ids) >= 3:
    results = content_based_recs(liked_ids, city)
    mode = "personalized"
else:
    results = cold_start_recs(city, cuisine_pref=cuisine, price_pref=price, min_rating=min_rating)
    mode = "popular"


# ─────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────
st.markdown("<div class='page-title'>Nom Nom Navigator</div>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Discover your next great meal</div>", unsafe_allow_html=True)

if mode == "personalized":
    st.markdown(
        f"<div class='info-strip'>✨ Showing personalised recommendations based on "
        f"<strong>{len(liked_ids)}</strong> restaurants you liked.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<div class='info-strip'>📈 Showing top-rated spots in <strong>{city}</strong>. "
        f"Add 3+ liked restaurants in the sidebar for personalised picks.</div>",
        unsafe_allow_html=True,
    )

if results is not None and len(results) > 0:

    # ── Two-column layout: map left, ranked cards right ──────────────
    col_map, col_cards = st.columns([5, 5], gap="large")

    with col_map:
        st.markdown("<div class='section-label'>On the Map</div>", unsafe_allow_html=True)

        # All open restaurants in the city, colored by star rating
        all_city = (
            df[(df['city'] == city) & (df['is_open'] == 1)]
            .drop_duplicates('business_id')[['latitude', 'longitude', 'name', 'stars', 'cuisine_group']]
            .dropna(subset=['latitude', 'longitude'])
            .copy()
        )

        # purple (1★) → green (5★), matching Image 2 palette
        def rating_color(stars):
            t = (float(stars) - 1) / 4.0
            r = int(180 - 150 * t)
            g = int(80  + 160 * t)
            b = int(180 - 160 * t)
            return [r, g, b, 180]

        all_city['color'] = all_city['stars'].apply(rating_color)

        # Top recommended restaurants — highlighted in amber on top
        rec_map = results[['latitude', 'longitude', 'name', 'stars', 'cuisine_group']].dropna()
        center_lat = all_city['latitude'].mean()
        center_lon = all_city['longitude'].mean()

        st.pydeck_chart(pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=11,
                pitch=0,
            ),
            layers=[
                # All city restaurants — small, rating-colored
                pdk.Layer(
                    'ScatterplotLayer',
                    data=all_city,
                    get_position='[longitude, latitude]',
                    get_color='color',
                    get_radius=80,
                    pickable=True,
                    opacity=0.7,
                ),
                # Recommended restaurants — larger amber dots on top
                pdk.Layer(
                    'ScatterplotLayer',
                    data=rec_map,
                    get_position='[longitude, latitude]',
                    get_color='[210, 105, 40, 255]',
                    get_radius=220,
                    pickable=True,
                ),
            ],
            tooltip={"text": "🍽️ {name}\n⭐ {stars}\n🍴 {cuisine_group}"},
        ), use_container_width=True)
        st.caption("🟠 Your recommendations · 🟢 Highly rated · 🟣 Lower rated")

        # ── Mini summary table below map ──
        st.markdown("<div class='section-label' style='margin-top:24px;'>Quick Glance</div>", unsafe_allow_html=True)
        summary = results[['name', 'cuisine_group', 'stars', 'price_range']].copy()
        summary['price_range'] = summary['price_range'].apply(
            lambda x: '$' * int(x) if pd.notna(x) else 'N/A'
        )
        summary.columns = ['Name', 'Cuisine', '★', 'Price']
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with col_cards:
        st.markdown("<div class='section-label'>Top Picks</div>", unsafe_allow_html=True)
        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            render_card(rank, row)

else:
    st.warning("No restaurants found matching your filters — try loosening the criteria.")