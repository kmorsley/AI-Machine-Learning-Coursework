# Homework 4 - Multiple Linear Regression with Scikit-learn
# Keishana Morsley | UTA ID: 1001807020

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Keep input/output paths relative to this script so the folder is portable.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Homework4_Multiple_Regression_Data.xlsx"

# Predictors and target required by the assignment.
predictors = [
    "Advertising_Spend",
    "Website_Visits",
    "Discount_Rate_Pct",
    "Sales_Staff_Hours",
    "Customer_Count",
]
target = "Sales_Revenue"

# -----------------------------------------------------------------------------
# Part 1 - Load and Understand the Data
# -----------------------------------------------------------------------------
print("\nPART 1 - LOAD AND UNDERSTAND THE DATA")
print("=" * 60)

df = pd.read_excel(DATA_FILE, sheet_name="Business_Data")

print("\nFirst five observations:")
print(df.head())

print(f"\nNumber of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\nColumn names:")
print(list(df.columns))
print("\nData types:")
print(df.dtypes)

analysis_columns = predictors + [target]
print("\nDescriptive statistics:")
print(df[analysis_columns].describe())

print("\nMissing values by column:")
print(df.isna().sum())

# -----------------------------------------------------------------------------
# Part 2 - Explore Relationships
# -----------------------------------------------------------------------------
print("\n\nPART 2 - EXPLORE RELATIONSHIPS")
print("=" * 60)

correlation_matrix = df[analysis_columns].corr()
correlation_matrix.to_csv(BASE_DIR / "correlation_matrix.csv")
print("\nCorrelation matrix:")
print(correlation_matrix)

sales_correlations = (
    correlation_matrix[target]
    .drop(target)
    .sort_values(key=lambda s: s.abs(), ascending=False)
)
print("\nPredictor correlations with Sales_Revenue (strongest absolute relationship first):")
print(sales_correlations)
print(
    f"\nThe strongest relationship with Sales_Revenue is "
    f"{sales_correlations.index[0]} (r = {sales_correlations.iloc[0]:.3f})."
)

plt.figure(figsize=(9, 7))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix: Business Predictors and Sales Revenue")
plt.tight_layout()
plt.savefig(BASE_DIR / "correlation_heatmap.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(df["Advertising_Spend"], df[target], alpha=0.75)
plt.title("Advertising Spend vs. Sales Revenue")
plt.xlabel("Advertising Spend ($)")
plt.ylabel("Sales Revenue ($)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(BASE_DIR / "advertising_sales_scatterplot.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# Part 3 - Create Training and Testing Data
# -----------------------------------------------------------------------------
print("\n\nPART 3 - CREATE TRAINING AND TESTING DATA")
print("=" * 60)

X = df[predictors]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Training observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")

# -----------------------------------------------------------------------------
# Part 4 - Build the Multiple Linear Regression Model
# -----------------------------------------------------------------------------
print("\n\nPART 4 - BUILD THE MULTIPLE LINEAR REGRESSION MODEL")
print("=" * 60)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nIntercept: {model.intercept_:.6f}")
print("\nRegression coefficients:")
for name, coefficient in zip(predictors, model.coef_):
    print(f"{name}: {coefficient:.6f}")

print("\nFitted regression equation:")
equation = f"Predicted Sales Revenue = {model.intercept_:.4f}"
for name, coefficient in zip(predictors, model.coef_):
    sign = "+" if coefficient >= 0 else "-"
    equation += f" {sign} {abs(coefficient):.4f}({name})"
print(equation)

# -----------------------------------------------------------------------------
# Part 5 - Generate Predictions
# -----------------------------------------------------------------------------
print("\n\nPART 5 - GENERATE PREDICTIONS")
print("=" * 60)

test_predictions = model.predict(X_test)

predictions_df = X_test.copy()
predictions_df.insert(0, "Observation_ID", df.loc[X_test.index, "Observation_ID"])
predictions_df["Actual_Sales_Revenue"] = y_test
predictions_df["Predicted_Sales_Revenue"] = test_predictions
predictions_df["Residual"] = (
    predictions_df["Actual_Sales_Revenue"]
    - predictions_df["Predicted_Sales_Revenue"]
)
predictions_df.to_csv(BASE_DIR / "regression_predictions.csv", index=False)

print("\nFirst five testing predictions:")
print(predictions_df.head())

# -----------------------------------------------------------------------------
# Part 6 - Evaluate Model Performance
# -----------------------------------------------------------------------------
print("\n\nPART 6 - EVALUATE MODEL PERFORMANCE")
print("=" * 60)

train_predictions = model.predict(X_train)

train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, train_predictions)

test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, test_predictions)

metrics_df = pd.DataFrame(
    {
        "Dataset": ["Training", "Testing"],
        "MSE": [train_mse, test_mse],
        "RMSE": [train_rmse, test_rmse],
        "R2": [train_r2, test_r2],
    }
)
metrics_df.to_csv(BASE_DIR / "regression_metrics.csv", index=False)

print("\nModel performance:")
print(metrics_df.to_string(index=False))

# -----------------------------------------------------------------------------
# Part 7 - Coefficient and Residual Visualizations
# -----------------------------------------------------------------------------
print("\n\nPART 7 - COEFFICIENT AND RESIDUAL VISUALIZATIONS")
print("=" * 60)

plt.figure(figsize=(9, 6))
plt.bar(predictors, model.coef_)
plt.axhline(0, linewidth=1)
plt.title("Multiple Linear Regression Coefficients")
plt.xlabel("Predictor")
plt.ylabel("Estimated Coefficient")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(BASE_DIR / "regression_coefficients.png", dpi=200)
plt.close()

plt.figure(figsize=(7, 6))
plt.scatter(y_test, test_predictions, alpha=0.80)
minimum = min(y_test.min(), test_predictions.min())
maximum = max(y_test.max(), test_predictions.max())
plt.plot([minimum, maximum], [minimum, maximum], linewidth=1.5)
plt.title("Actual vs. Predicted Sales Revenue")
plt.xlabel("Actual Sales Revenue ($)")
plt.ylabel("Predicted Sales Revenue ($)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(BASE_DIR / "actual_vs_predicted.png", dpi=200)
plt.close()

residuals = y_test.to_numpy() - test_predictions
plt.figure(figsize=(7, 6))
plt.scatter(test_predictions, residuals, alpha=0.80)
plt.axhline(0, linewidth=1.5)
plt.title("Residual Plot")
plt.xlabel("Predicted Sales Revenue ($)")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(BASE_DIR / "residual_plot.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# Part 8 - Make Business Predictions
# -----------------------------------------------------------------------------
print("\n\nPART 8 - MAKE BUSINESS PREDICTIONS")
print("=" * 60)

business_scenarios = pd.DataFrame(
    {
        "Scenario": ["A", "B", "C"],
        "Advertising_Spend": [20000, 35000, 50000],
        "Website_Visits": [30000, 40000, 50000],
        "Discount_Rate_Pct": [8, 10, 12],
        "Sales_Staff_Hours": [650, 750, 850],
        "Customer_Count": [1000, 1200, 1400],
    }
)

business_scenarios["Predicted_Sales_Revenue"] = model.predict(
    business_scenarios[predictors]
)
business_scenarios.to_csv(BASE_DIR / "business_predictions.csv", index=False)

print("\nBusiness scenario predictions:")
print(business_scenarios.to_string(index=False))

# -----------------------------------------------------------------------------
# Part 9 - GitHub Copilot Review
# -----------------------------------------------------------------------------
print("\n\nPART 9 - GITHUB COPILOT REVIEW")
print("=" * 60)
print(
    "See Code_Changes_Keishana_Morsley_1001807020.md for the documented "
    "Copilot review and code changes."
)

print("\nHomework 4 analysis completed successfully.")
print(f"Output files were saved to: {BASE_DIR}")
