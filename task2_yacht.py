import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
yachts = pd.read_csv("yacht_hydrodynamics.data", sep=r"\s+", header=None)
X = yachts[[5]]  # Froude number
y = yachts[6]    # Residuary resistance per unit displacement weight

# (a) Build linear baseline and degree-2 polynomial pipeline
linear_model = LinearRegression()
quadratic_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression()),
])

# (b) 5-fold shuffled CV with random_state=42
folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"

linear_rmse = -cross_val_score(linear_model, X, y, cv=folds, scoring=scoring)
quadratic_rmse = -cross_val_score(quadratic_model, X, y, cv=folds, scoring=scoring)

print("Linear Fold RMSEs:", linear_rmse.round(2))
print(f"Linear Mean RMSE: {linear_rmse.mean():.2f}")
print("Quadratic Fold RMSEs:", quadratic_rmse.round(2))
print(f"Quadratic Mean RMSE: {quadratic_rmse.mean():.2f}")

# (c) Choose the better model
better = "Quadratic" if quadratic_rmse.mean() < linear_rmse.mean() else "Linear"
print(f"Better model based on mean CV RMSE: {better}")