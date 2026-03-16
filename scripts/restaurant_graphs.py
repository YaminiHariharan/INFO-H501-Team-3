import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_json("data/processed/indy_philly_restaurant_aggregated.json", lines=True)

# Filter Indianapolis and Philadelphia
indy = df[df["city"] == "Indianapolis"]
philly = df[df["city"] == "Philadelphia"]

# Count restaurants by city
city_counts = df["city"].value_counts()

plt.figure()

city_counts.plot(kind="bar", color=["blue","red"])

plt.title("Restaurant Counts by City")
plt.xlabel("City")
plt.ylabel("Number of Restaurants")

plt.tight_layout()

plt.savefig("graphs/restaurants_by_city.png")
