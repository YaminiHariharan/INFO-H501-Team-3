# 🍽️ Restaurant Finder & Market Analysis

## 📊 Overview

This project builds an interactive Streamlit web application to explore and compare restaurant data across Indianapolis, Indiana and Philadelphia, Pennsylvania.

The goal is to help users quickly:
- discover restaurants
- compare quality and popularity
- make decisions based on ratings, reviews, and location

The app is designed for **fast client decision-making (under 30 seconds)** using filters, rankings, and a live map.

---

## 👤 Stakeholder Value

This tool is useful for:
- customers looking for restaurants nearby
- users comparing food quality across cities
- anyone wanting a quick and visual restaurant decision tool

The app provides:
- search and filtering
- star ratings and review counts
- location-based (ZIP radius) filtering
- visual insights and maps

---

## 🗂️ Data Description

The dataset contains restaurant information including:
- name
- city
- category (cuisine)
- rating (stars)
- review count
- ZIP code (used for location)
- optional comment text

ZIP codes are mapped to approximate geographic coordinates for location-based filtering and visualization.

---

## ⚙️ Algorithm Description

The app processes user input through several steps:

1. **Filtering (pandas)**
   - search keyword
   - city selection
   - category selection
   - minimum rating
   - minimum review count

2. **Location Filtering**
   - user selects a ZIP code
   - distance is calculated using the Haversine formula
   - only restaurants within a chosen radius are shown

3. **Ranking**
   - restaurants are sorted by:
     - rating (primary)
     - review count (secondary)
   - used to generate “Top Picks”

4. **Visualization**
   - charts using Plotly
   - map using PyDeck (ZIP-based coordinates)

---

## 🛠️ Tools Used

- **Python** → core programming
- **pandas** → data manipulation
- **Streamlit** → web app interface
- **Plotly** → charts and visualizations
- **PyDeck** → map visualization
- **GitHub** → version control

---

## 🧪 Testing

Basic tests are included to verify:
- filtering logic
- distance calculations
- ranking behavior

Tests are located in the `/tests` directory.

---

## ⚠️ Ethical Considerations

- Location data is approximate (ZIP-level), not exact addresses
- Ratings and reviews may reflect bias or incomplete user feedback
- The app does not guarantee restaurant quality — it only summarizes available data

To mitigate risks:
- the app clearly displays ratings and review counts
- users can filter results based on multiple criteria
- limitations are acknowledged in the interface

---

## 🚀 Key Insights

- More restaurants does not always mean better quality
- High review counts indicate reliability, not necessarily higher ratings
- Less common cuisines often have the highest ratings
- Restaurant quality and density vary significantly by location

---

## 🧠 Conclusion

This project combines data analysis and an interactive web app to provide a complete restaurant exploration experience.

By integrating:
- filtering
- ranking
- location-based search
- visualization

it allows users to make fast, informed dining decisions.

---

## 📌 Future Improvements

- exact latitude/longitude for each restaurant
- real-time data from APIs (Google Places, Yelp)
- user personalization and recommendations
- improved sentiment analysis of reviews
