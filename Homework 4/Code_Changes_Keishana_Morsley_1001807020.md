# Homework 4 – GitHub Copilot Review

**Student:** Keishana Morsley  
**UTA ID:** 1001807020  
**Assignment:** Homework 4 – Multiple Linear Regression with Scikit-learn

## Meaningful Changes Made to the Copilot-Suggested Code

### 1. Replaced a hard-coded computer path with a portable relative path
The initial approach referenced the dataset through a location tied to the local computer. I changed the program to use `Path(__file__).resolve().parent` and construct both the dataset path and output paths from the Python file's folder.

**Why this improved the program:** The completed Homework 4 folder can now be moved, zipped, cloned from GitHub, or opened on another computer without changing the path in the code. It also ensures that all required output files are created inside the Homework 4 folder.

### 2. Rebuilt the testing-predictions output to preserve Observation_ID
The prediction workflow originally focused only on the five X variables and the target. I changed the output construction so that the original `Observation_ID` is recovered from the test-set index and included in `regression_predictions.csv`. I also calculated the residual explicitly as `Actual_Sales_Revenue - Predicted_Sales_Revenue`.

**Why this improved the program:** Each prediction can now be traced back to its original business observation, and the output matches the assignment's required columns and residual definition.

### 3. Added separate training and testing evaluation instead of evaluating only the test set
The first version emphasized testing performance. I added predictions and MSE, RMSE, and R² calculations for both the training and testing datasets and placed the results together in `regression_metrics.csv`.

**Why this improved the program:** Comparing training and testing performance gives a clearer view of model generalization and directly addresses the assignment's bias-versus-variance and model-evaluation objectives.

### 4. Improved the actual-versus-predicted and residual visualizations
I added a 45-degree reference line to the actual-versus-predicted chart and a horizontal zero-reference line to the residual plot. I also added descriptive chart titles, axis labels, and layout adjustments.

**Why this improved the program:** The reference lines make it easier to visually judge prediction accuracy and whether residuals are centered around zero, while the labels make the figures easier to interpret without referring back to the source code.

## Final Review
After the changes, the program was checked against the Homework 4 requirements. It uses the five required predictors, excludes `Observation_ID` and `Region` from the regression model, uses an 80/20 split with `random_state=42`, fits `LinearRegression`, calculates the required metrics, makes the three required business predictions, and creates the required CSV and PNG files.
