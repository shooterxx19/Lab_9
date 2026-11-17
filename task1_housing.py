import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
homes = pd.read_csv("real_estate.csv")
homes["distance_100m"] = homes["X3 distance to the nearest MRT station"] / 100
X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

# (a) Fit LinearRegression on all 414 rows
model = LinearRegression()
model.fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")

# (b) Predict price at 500 m (distance_100m = 5)
new_data = pd.DataFrame({"distance_100m": [5]})
pred_500m = model.predict(new_data)[0]
print(f"Predicted price at 500 m: {pred_500m:.2f}")

# (c) Plot observations and fitted line
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.6, label="Observed Transactions")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Fitted Line")
plt.xlabel("Distance to Nearest MRT Station (100 m)")
plt.ylabel("House Price of Unit Area (10,000 NTD/ping)")
plt.title("Housing Price vs. Distance to MRT Station")
plt.legend()
plt.tight_layout()
plt.savefig("task1_housing.png")
plt.close()