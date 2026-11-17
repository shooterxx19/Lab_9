import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Load dataset and remove duplicates
concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()
target = "Concrete compressive strength"
X = concrete.drop(columns=target)
y = concrete[target]

# (a) 80/20 split and linear baseline CV
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")

folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"

linear_rmse = -cross_val_score(
    LinearRegression(), X_train, y_train, cv=folds, scoring=scoring
)
print(f"Linear Baseline Mean CV RMSE: {linear_rmse.mean():.2f} MPa")

# (b) Tune Polynomial + StandardScaler + Ridge pipeline
pipeline = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge()),
])

param_grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100],
}

search = GridSearchCV(pipeline, param_grid, cv=folds, scoring=scoring, refit=True)
search.fit(X_train, y_train)

# Print top 3 settings
results = pd.DataFrame(search.cv_results_)
top3 = results.sort_values("rank_test_score").head(3)
print("\nTop 3 Grid Search Settings:")
for _, row in top3.iterrows():
    print(
        f"Rank {int(row['rank_test_score'])}: "
        f"degree={row['param_poly__degree']}, alpha={row['param_ridge__alpha']} "
        f"-> Mean CV RMSE: {-row['mean_test_score']:.2f} MPa"
    )

# (c) Evaluate on held-out test set and plot
test_preds = search.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
test_mae = mean_absolute_error(y_test, test_preds)
print(f"\nTuned Model Test RMSE: {test_rmse:.2f} MPa")
print(f"Tuned Model Test MAE: {test_mae:.2f} MPa")

# Print example degree-2 feature names for part (d)
poly_features = search.best_estimator_.named_steps["poly"].get_feature_names_out(X.columns)
interactions = [f for f in poly_features if " " in f]
print(f"\nExample degree-2 interaction term: {interactions[0]}")

plt.figure(figsize=(7, 7))
plt.scatter(y_test, test_preds, alpha=0.6, label="Test Predictions")
min_val = min(y_test.min(), test_preds.min())
max_val = max(y_test.max(), test_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y = x (Ideal)")
plt.xlabel("Actual Compressive Strength (MPa)")
plt.ylabel("Predicted Compressive Strength (MPa)")
plt.title("Actual vs. Predicted Concrete Compressive Strength")
plt.legend()
plt.tight_layout()
plt.savefig("task3_concrete.png")
plt.close()