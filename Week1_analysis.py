import pandas as pd

# Load logistics dataset
data = pd.read_csv("logistics_data.csv")

# Inspect the dataset
print(data.head())
print(data.info())

# Calculate basic performance measures
average_delivery_time = data["Delivery_Time"].mean()
average_transport_cost = data["Transportation_Cost"].mean()

print("Average Delivery Time:", average_delivery_time)
print("Average Transportation Cost:", average_transport_cost)
