import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# 1. Define Parameters
cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
start_date = datetime(2023, 7, 15)
end_date = datetime(2025, 7, 14)

# Define base characteristics for each city
city_data = {
    "New York": {"base_price": 150, "base_volume": 50},
    "Los Angeles": {"base_price": 125, "base_volume": 40},
    "Chicago": {"base_price": 110, "base_volume": 35},
    "Houston": {"base_price": 105, "base_volume": 30},
    "Phoenix": {"base_price": 100, "base_volume": 30}
}

# Generate a date range for the two-year period
dates = pd.date_range(start_date, end_date, freq='D')

# 2. Generate the Dataset
data = []
for city in cities:
    base_price = city_data[city]["base_price"]
    base_volume = city_data[city]["base_volume"]
    for date in dates:
        # Simulate price with random daily variation using a normal distribution
        price = np.round(np.random.normal(loc=base_price, scale=20), 2)
        foo = (price / base_price)**2
        # Simulate volume to be positively correlated with price, with some noise
        volume = int(np.round(base_volume * (price / base_price)**2))
        volume = max(volume, 0)  # Ensure volume is not negative

        data.append({"date": date, "city": city, "average_price": price, "volume": volume})

# 3. Create a Pandas DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Open a file in write mode and save the data
with open("/Users/ryanborman/Documents/GitHub/last_mile/datasets/data.json", "w") as json_file:
    json.dump(city_data, json_file)
# Save the dataset to a CSV file (optional)
df.to_csv("/Users/ryanborman/Documents/GitHub/last_mile/datasets/datasetsdelivery_data.csv", index=False)