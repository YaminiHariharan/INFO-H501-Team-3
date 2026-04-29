# Nom Nom Navigator 🗺️

## Abstract
Nom Nom Navigator is a restaurant recommendation web app built for food lovers 
in Indianapolis and Philadelphia. The app helps users discover highly-rated 
restaurants through both popularity-based and personalized content-based 
filtering. Users can filter by cuisine, price range, and minimum rating, or 
add restaurants they already love to receive tailored recommendations.

The primary stakeholder is a casual diner who wants trusted, personalized 
restaurant suggestions without having to scroll through hundreds of Yelp 
reviews. The app surfaces hidden gems alongside well-known spots, saving time 
and reducing decision fatigue.

## Data Description
The data was sourced from the **Yelp Open Dataset**, a publicly available 
collection of business, review, and user data. We filtered the dataset to 
include only restaurants in Indianapolis and Philadelphia, resulting in 
approximately 8,000+ restaurant records. 

Key cleaning steps included:
- Filtering by city and "Restaurant" category tag
- Imputing missing price range values with the median
- Engineering a `cuisine_group` feature by mapping Yelp category tags to 
  broader cuisine categories
- Computing a **Bayesian average score** to balance star ratings with review 
  volume and reduce noise from low-review restaurants
- Engineering boolean features for ambience, meal type, and dietary 
  preferences (vegetarian-friendly, halal)

## Algorithm Description
The app uses two recommendation strategies:

**Cold Start (default):** When a user has not provided liked restaurants, the 
app filters the dataset by city, cuisine, price, and minimum rating, then 
ranks results by Bayesian score. This ensures new users still receive 
high-quality recommendations.

**Content-Based Filtering (personalized):** When a user selects 3 or more 
restaurants they like, the app builds a feature matrix using cuisine dummies, 
price, ambience, and meal-type attributes. It then computes **cosine 
similarity** between all restaurants and returns the most similar restaurants 
to the user's liked set that they haven't seen yet. Results are filtered to 
ensure cuisine diversity and a minimum Bayesian score of 3.5.

## Tools Used
| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas | Data cleaning and manipulation |
| Scikit-learn | Cosine similarity, MinMaxScaler for feature engineering |
| Streamlit | Web app framework and UI |
| PyDeck | Interactive map visualizations |
| Jupyter Notebook | Exploratory data analysis and preprocessing |
| GitHub | Version control and collaboration |
| Yelp Open Dataset | Source data for restaurants and reviews |

## Ethical Concerns

**Cuisine Representation:** The `cuisine_map` used to group restaurants 
collapses many culturally distinct cuisines (Ethiopian, Filipino, Peruvian, 
etc.) into a generic "Other" category. This reduces the visibility of 
minority-owned restaurants in recommendations.

**Mitigation Strategies:**
- Expand the cuisine taxonomy to explicitly represent underrepresented cuisines
- Be transparent with users about the data source and its limitations
