df = pd.read_csv("business.csv")

# Filter states
df = df[df['state'].isin(['IN', 'PA'])]

# Filter restaurants
df = df[df['categories'].str.contains('Restaurant|Food', case=False, na=False)]
