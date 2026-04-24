# Restaurant Market Intelligence

## Overview
This project is a Streamlit web application that helps users quickly explore and compare restaurants in Indianapolis, IN and Philadelphia, PA. The app allows users to filter restaurants by location, category, ratings, and proximity, and visualize results through interactive maps and charts.

## Purpose
The goal is to help users make fast, informed dining decisions by combining restaurant data with location-based filtering and visual insights.

## Data Description
The dataset includes restaurant information such as:
- Name
- City
- Category (cuisine)
- Ratings
- Review counts
- Latitude and longitude coordinates

Data was cleaned and processed using pandas to ensure consistency and usability in the app.

## Algorithm Description
The app uses:
- Filtering logic based on user input (city, rating, reviews, ZIP radius)
- A ranking system that prioritizes higher-rated and more-reviewed restaurants
- Distance calculations using latitude/longitude (Haversine formula)
- Visualization through Plotly maps and charts

## Tools Used
- Python – core programming language
- Streamlit – web app framework
- pandas – data manipulation
- Plotly – data visualization
- Git/GitHub – version control and collaboration

## Ethical Considerations
- The app relies on user-generated reviews, which may introduce bias
- Some locations use ZIP code approximations instead of exact coordinates
- Data may not reflect real-time restaurant changes or availability

## Limitations
- Uses ZIP center points rather than exact locations
- No live API integration (static dataset)
- Generated images instead of real restaurant photos

## Future Improvements
- Integrate real-time restaurant APIs
- Add user personalization features
- Improve map accuracy with exact coordinates
