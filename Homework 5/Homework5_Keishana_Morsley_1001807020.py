# Homework 5 - Gradient Descent and Learning Rate Comparison
# Keishana Morsley | UTA ID: 1001807020

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# Keep input/output paths relative to this script so the folder is portable.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Homework4_Multiple_Regression_Data-1.xlsx"


# -----------------------------------------------------------------------------
# Part 1 - Load the Homework 4 Dataset
# -----------------------------------------------------------------------------

print("\nPART 1 - LOAD THE HOMEWORK 4 DATASET")
print("=" * 60)

# Load the same dataset used in Homework 4.
df = pd.read_excel(DATA_FILE, sheet_name="Business_Data")

# Homework 5 uses only Advertising_Spend as X and Sales_Revenue as y.
X = df[["Advertising_Spend"]]
y = df["Sales_Revenue"]

# Display the first five observations of the two required variables.
print("\nFirst five observations:")
print(df[["Advertising_Spend", "Sales_Revenue"]].head())

# Split the data into 80% training and 20% testing data.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")

# -----------------------------------------------------------------------------
# Part 2 - Standardize X and Y
# -----------------------------------------------------------------------------

print("\n\nPART 2 - STANDARDIZE X AND Y")
print("=" * 60)

# Create separate scalers for X and y.
x_scaler = StandardScaler()
y_scaler = StandardScaler()

# Fit the scalers using the training data only.
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
).flatten()

y_test_scaled = y_scaler.transform(
    y_test.to_numpy().reshape(-1, 1)
).flatten()

print("\nFirst five standardized X training values:")
print(X_train_scaled[:5])

print("\nFirst five standardized y training values:")
print(y_train_scaled[:5])

# -----------------------------------------------------------------------------
# Part 3 - Create a Simple Scikit-learn Benchmark
# -----------------------------------------------------------------------------

print("\n\nPART 3 - CREATE A SIMPLE SCIKIT-LEARN BENCHMARK")
print("=" * 60)

benchmark_model = LinearRegression()

benchmark_model.fit(X_train, y_train)

benchmark_predictions = benchmark_model.predict(X_test)

benchmark_mse = mean_squared_error(y_test, benchmark_predictions)
benchmark_rmse = np.sqrt(benchmark_mse)
benchmark_r2 = r2_score(y_test, benchmark_predictions)

print(f"\nTest RMSE: {benchmark_rmse:.6f}")
print(f"Test R²: {benchmark_r2:.6f}")

# -----------------------------------------------------------------------------
# Part 4 - Implement Gradient Descent
# -----------------------------------------------------------------------------

print("\n\nPART 4 - IMPLEMENT GRADIENT DESCENT")
print("=" * 60)

def generate_predictions(X, slope, intercept):
    return slope * X + intercept


def calculate_cost(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted) ** 2)


def update_parameters(X, y, slope, intercept, learning_rate):
    n = len(X)

    predictions = generate_predictions(X, slope, intercept)

    slope_gradient = (-2 / n) * np.sum(X * (y - predictions))
    intercept_gradient = (-2 / n) * np.sum(y - predictions)

    slope = slope - learning_rate * slope_gradient
    intercept = intercept - learning_rate * intercept_gradient

    return slope, intercept


def gradient_descent(X, y, learning_rate, iterations=500):
    slope = 0.0
    intercept = 0.0
    cost_history = []

    for _ in range(iterations):
        slope, intercept = update_parameters(
            X,
            y,
            slope,
            intercept,
            learning_rate
        )

        predictions = generate_predictions(X, slope, intercept)
        cost = calculate_cost(y, predictions)
        cost_history.append(cost)

    return slope, intercept, cost_history

# -----------------------------------------------------------------------------
# Part 5 - Compare Three Learning Rates
# -----------------------------------------------------------------------------

print("\n\nPART 5 - COMPARE THREE LEARNING RATES")
print("=" * 60)

learning_rates = [0.001, 0.01, 0.10]
comparison_results = []
gd_results = {}

X_train_gd = X_train_scaled.flatten()
X_test_gd = X_test_scaled.flatten()

for learning_rate in learning_rates:
    slope, intercept, cost_history = gradient_descent(
        X_train_gd,
        y_train_scaled,
        learning_rate,
        iterations=500
    )

    test_predictions_scaled = generate_predictions(
        X_test_gd,
        slope,
        intercept
    )

    test_predictions = y_scaler.inverse_transform(
        test_predictions_scaled.reshape(-1, 1)
    ).flatten()

    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_r2 = r2_score(y_test, test_predictions)

    final_cost = cost_history[-1]

    comparison_results.append(
        {
            "Learning Rate": learning_rate,
            "Final Cost": final_cost,
            "Test RMSE": test_rmse,
            "Test R²": test_r2,
            "Final Slope": slope,
            "Final Intercept": intercept
        }
    )

    gd_results[learning_rate] = {
        "slope": slope,
        "intercept": intercept,
        "cost_history": cost_history,
        "test_rmse": test_rmse,
        "test_r2": test_r2
    }

learning_rate_comparison = pd.DataFrame(comparison_results)

learning_rate_comparison.to_csv(
    BASE_DIR / "learning_rate_comparison.csv",
    index=False
)

print("\nLearning rate comparison:")
print(learning_rate_comparison.to_string(index=False))

# -----------------------------------------------------------------------------
# Part 6 - Plot Learning-Rate Convergence
# -----------------------------------------------------------------------------

print("\n\nPART 6 - PLOT LEARNING-RATE CONVERGENCE")
print("=" * 60)

plt.figure(figsize=(9, 6))

for learning_rate in learning_rates:
    plt.plot(
        gd_results[learning_rate]["cost_history"],
        label=f"Learning Rate = {learning_rate}"
    )

plt.title("Gradient Descent Learning-Rate Convergence")
plt.xlabel("Iteration")
plt.ylabel("Cost / MSE")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    BASE_DIR / "learning_rate_convergence.png",
    dpi=200
)

plt.close()

print("\nSaved: learning_rate_convergence.png")

# -----------------------------------------------------------------------------
# Part 7 - Compare Gradient Descent with Scikit-learn
# -----------------------------------------------------------------------------

print("\n\nPART 7 - COMPARE GRADIENT DESCENT WITH SCIKIT-LEARN")
print("=" * 60)

model_comparison = pd.DataFrame(
    [
        {
            "Model": "Scikit-learn Simple Linear Regression",
            "Test RMSE": benchmark_rmse,
            "Test R²": benchmark_r2
        },
        {
            "Model": "GD - Learning Rate 0.001",
            "Test RMSE": gd_results[0.001]["test_rmse"],
            "Test R²": gd_results[0.001]["test_r2"]
        },
        {
            "Model": "GD - Learning Rate 0.01",
            "Test RMSE": gd_results[0.01]["test_rmse"],
            "Test R²": gd_results[0.01]["test_r2"]
        },
        {
            "Model": "GD - Learning Rate 0.10",
            "Test RMSE": gd_results[0.10]["test_rmse"],
            "Test R²": gd_results[0.10]["test_r2"]
        }
    ]
)

model_comparison.to_csv(
    BASE_DIR / "model_comparison.csv",
    index=False
)

print("\nModel comparison:")
print(model_comparison.to_string(index=False))

# -----------------------------------------------------------------------------
# Part 8 - Short Regularization Demonstration
# -----------------------------------------------------------------------------

print("\n\nPART 8 - SHORT REGULARIZATION DEMONSTRATION")
print("=" * 60)

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train, y_train)

ridge_predictions = ridge_model.predict(X_test)

ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_predictions))
ridge_r2 = r2_score(y_test, ridge_predictions)

ridge_result = pd.DataFrame(
    [
        {
            "Model": "Ridge Regression (alpha=1.0)",
            "Test RMSE": ridge_rmse,
            "Test R²": ridge_r2
        }
    ]
)

model_comparison = pd.concat(
    [model_comparison, ridge_result],
    ignore_index=True
)

model_comparison.to_csv(
    BASE_DIR / "model_comparison.csv",
    index=False
)

print("\nUpdated model comparison:")
print(model_comparison.to_string(index=False))

# -----------------------------------------------------------------------------
# Part 9 - GitHub Copilot Review
# -----------------------------------------------------------------------------

print("\n\nPART 9 - GITHUB COPILOT REVIEW")
print("=" * 60)

print(
    "See Code_Changes_Keishana_Morsley_1001807020.md for the documented "
    "Copilot review and code changes."
)

print("\nHomework 5 analysis completed successfully.")
print(f"Output files were saved to: {BASE_DIR}")