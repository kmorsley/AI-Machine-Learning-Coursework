import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PART 1: LOAD AND UNDERSTAND THE DATA
# Load the CSV file (script runs inside the Homework 3 folder)
df = pd.read_csv('Homework3_PCA_Customer_Data(Customer_Data).csv')

print('First 5 Rows of the Dataset:')
print(df.head())
print('\n')

print('Dataset Shape (rows, columns):')
print(df.shape)
print('\n')

print('Column Names:')
print(df.columns.tolist())
print('\n')

print('Data Types:')
print(df.dtypes)
print('\n')

print('Missing Values:')
print(df.isnull().sum())
print('\n')

print('Descriptive Statistics:')
print(df.describe())

# PART 2: CORRELATION ANALYSIS
pca_variables = [
    'Annual_Income', 'Annual_Spend', 'Purchases_12M', 'Avg_Order_Value',
    'Website_Visits_3M', 'App_Sessions_3M', 'Email_Click_Rate_Pct',
    'Loyalty_Points', 'Satisfaction_Score', 'Support_Contacts_12M',
    'Days_Since_Last_Purchase', 'Return_Rate_Pct'
]

existing_vars = [col for col in pca_variables if col in df.columns]
missing_vars = [col for col in pca_variables if col not in df.columns]
if missing_vars:
    print('Warning: The following expected variables were not found:')
    print(missing_vars)

pca_df = df[existing_vars].copy()

# Clean formatting (remove $ , %), then convert to numeric
for col in existing_vars:
    if pca_df[col].dtype == object or str(pca_df[col].dtype).startswith('string'):
        pca_df[col] = pca_df[col].astype(str).str.replace(r'[\\$,]', '', regex=True)
        pca_df[col] = pca_df[col].str.replace('%', '', regex=False)
        pca_df[col] = pca_df[col].str.strip()
    pca_df[col] = pd.to_numeric(pca_df[col], errors='coerce')

print('\nData types after cleaning:')
print(pca_df.dtypes)

# Pearson correlation matrix
corr_matrix = pca_df.corr(method='pearson')
print('\nPearson Correlation Matrix:')
print(corr_matrix)

corr_matrix.to_csv('correlation_matrix.csv', index=True)

# Heatmap
plt.figure(figsize=(10, 8))
sns.set_theme(style='white')
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Heatmap — PCA Variables', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved correlation matrix to: correlation_matrix.csv')
print('Saved correlation heatmap to: correlation_heatmap.png')

# PART 3: STANDARDIZE USING SCIKIT-LEARN
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
standardized_array = scaler.fit_transform(pca_df)
standardized_df = pd.DataFrame(standardized_array, columns=pca_df.columns, index=pca_df.index)

print('\nFirst 5 rows of standardized PCA variables:')
print(standardized_df.head())
print('\nMeans (should be ~0):')
print(standardized_df.mean().round(6))
print('\nStdevs (should be ~1):')
print(standardized_df.std(ddof=0).round(6))
standardized_df.to_csv('standardized_customer_features.csv', index=False)
print('\nSaved standardized features to: standardized_customer_features.csv')

# PART 4: PCA
from sklearn.decomposition import PCA
pca = PCA()
# Fit PCA
pca.fit(standardized_df)
# Transform data
pc_scores = pca.transform(standardized_df)

explained_var_ratio = pca.explained_variance_ratio_
print('\nExplained variance ratio for each principal component:')
print(explained_var_ratio)

cumulative_explained = explained_var_ratio.cumsum()
print('\nCumulative explained variance:')
print(cumulative_explained)

pc_labels = [f'PC{i+1}' for i in range(len(explained_var_ratio))]
pca_summary = pd.DataFrame({
    'Principal Component': pc_labels,
    'Explained Variance Ratio': explained_var_ratio,
    'Cumulative Explained Variance': cumulative_explained
})
print('\nPCA explained variance summary:')
print(pca_summary)

pca_summary.to_csv('pca_explained_variance.csv', index=False)
print('\nSaved PCA explained variance to: pca_explained_variance.csv')

# ============================================================================
# PART 5: DETERMINE NUMBER OF PRINCIPAL COMPONENTS
# ============================================================================

# Import numpy for array operations
import numpy as np

# Use the explained variance ratios and cumulative explained variance
# These were already computed in Part 4
# Find the smallest number of components needed to reach at least 80% variance
threshold = 0.80
components_needed = int(np.argmax(cumulative_explained >= threshold) + 1)

# Print the number of components required
print(f'\nNumber of principal components needed to explain at least {int(threshold*100)}% variance: {components_needed}')

# Print the cumulative explained variance at that number of components
cum_at_k = cumulative_explained[components_needed - 1]
print(f'Cumulative explained variance at {components_needed} components: {cum_at_k:.4f}')

# Create scree plot: explained variance ratio for each PC
plt.figure(figsize=(8, 5))
pc_indices = np.arange(1, len(explained_var_ratio) + 1)
plt.bar(pc_indices, explained_var_ratio, color='C0', alpha=0.7)
plt.plot(pc_indices, explained_var_ratio, marker='o', color='C0')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.xticks(pc_indices)
plt.tight_layout()

# Save the scree plot
plt.savefig('pca_scree_plot.png', dpi=300)
plt.close()
print('Saved scree plot to: pca_scree_plot.png')

# Create cumulative explained variance plot
plt.figure(figsize=(8, 5))
plt.plot(pc_indices, cumulative_explained, marker='o', linestyle='-', color='C1')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.xticks(pc_indices)

# Add horizontal reference line at 80%
plt.axhline(y=threshold, color='gray', linestyle='--', linewidth=1)
plt.annotate(f'80% threshold', xy=(1, threshold + 0.02), color='gray')
plt.tight_layout()

# Save the cumulative variance plot
plt.savefig('pca_cumulative_variance.png', dpi=300)
plt.close()
print('Saved cumulative explained variance plot to: pca_cumulative_variance.png')

# ============================================================================
# PART 6: EXAMINE PCA LOADINGS
# ============================================================================

# Create a DataFrame of PCA loadings
# pca.components_ has shape (n_components, n_features)
# Transpose so that variables are rows and components are columns
pc_labels = [f'PC{i+1}' for i in range(pca.components_.shape[0])]
loadings_df = pd.DataFrame(
    pca.components_.T,  # Transpose to get variables as rows
    columns=pc_labels,
    index=pca_variables
)

print('\nPCA Loadings:')
print(loadings_df)

# Save loadings to CSV
loadings_df.to_csv('pca_loadings.csv')
print('\nSaved PCA loadings to: pca_loadings.csv')

# For the retained components (from Part 5),
# identify variables with largest absolute loadings
print(f'\nLargest Absolute Loadings for the {components_needed} Retained Components:')
for i in range(components_needed):
    pc_name = f'PC{i+1}'
    # Get absolute values of loadings for this component
    abs_loadings = loadings_df[pc_name].abs()
    # Get the top 3 variables with largest absolute loadings
    top_vars = abs_loadings.nlargest(3)
    print(f'\n{pc_name}:')
    for var, loading in top_vars.items():
        actual_loading = loadings_df.loc[var, pc_name]
        print(f'  {var}: {actual_loading:.4f} (|{loading:.4f}|)')

# ============================================================================
# PART 7: CREATE PCA SCORES
# ============================================================================

# Use only the retained principal components
# pc_scores contains scores for all components; keep only the first components_needed
retained_pc_scores = pc_scores[:, :components_needed]

# Create column names for the retained components
retained_pc_labels = [f'PC{i+1}' for i in range(components_needed)]

# Create a DataFrame with retained PCA scores
pca_scores_df = pd.DataFrame(retained_pc_scores, columns=retained_pc_labels)

# Add Customer_ID and Customer_Segment as the first two columns
pca_scores_df.insert(0, 'Customer_Segment', df['Customer_Segment'])
pca_scores_df.insert(0, 'Customer_ID', df['Customer_ID'])

print('\nFirst 5 rows of PCA scores (retained components):')
print(pca_scores_df.head())

# Save PCA scores to CSV
pca_scores_df.to_csv('pca_scores.csv', index=False)
print('\nSaved PCA scores to: pca_scores.csv')

# ============================================================================
# PART 8: PCA VISUALIZATIONS
# ============================================================================

# Create a scatterplot of PC1 versus PC2, colored by Customer_Segment
plt.figure(figsize=(10, 7))

# Get unique customer segments and assign colors
segments = pca_scores_df['Customer_Segment'].unique()
colors = plt.cm.Set1(range(len(segments)))

# Plot each segment with a different color
for segment, color in zip(segments, colors):
    mask = pca_scores_df['Customer_Segment'] == segment
    plt.scatter(
        pca_scores_df[mask]['PC1'],
        pca_scores_df[mask]['PC2'],
        label=segment,
        alpha=0.7,
        s=80,
        color=color
    )

plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.title('PCA Scatterplot: PC1 vs PC2', fontsize=14, fontweight='bold')
plt.legend(title='Customer_Segment', loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the scatterplot
plt.savefig('pca_scatterplot.png', dpi=300)
plt.close()
print('Saved PCA scatterplot to: pca_scatterplot.png')
