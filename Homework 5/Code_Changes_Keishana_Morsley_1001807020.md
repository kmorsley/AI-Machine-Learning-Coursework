# Homework 5 - GitHub Copilot Review
**Keishana Morsley**  

## Change 1 - Gradient Descent and Learning Rate Implementation

Copilot suggested the general structure for the gradient descent process. I modified the code so that the slope and intercept both begin at 0, as required by the assignment, and so that the same gradient descent function can accept different learning rates.

I also added a `cost_history` list so that the MSE is recorded during all 500 iterations. This allows the three learning rates to be compared using the convergence plot.

## Change 2 - Model Comparison Output

I modified the suggested comparison code so that the Scikit-learn benchmark, all three gradient descent models, and the Ridge Regression model are stored in one DataFrame.

I also added code to save the final results as `model_comparison.csv`, which makes it easier to directly compare the Test RMSE and Test R² values for each model.