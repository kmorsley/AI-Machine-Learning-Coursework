# Homework 3: Principal Component Analysis of Customer Data

## Dataset Overview

The analysis examined 240 customer observations with 14 variables total. Twelve numerical variables were selected for Principal Component Analysis:

Annual_Income, Annual_Spend, Purchases_12M, Avg_Order_Value, Website_Visits_3M, App_Sessions_3M, Email_Click_Rate_Pct, Loyalty_Points, Satisfaction_Score, Support_Contacts_12M, Days_Since_Last_Purchase, and Return_Rate_Pct.

Customer_ID and Customer_Segment were retained for identification and interpretation purposes but excluded from the PCA inputs.

## Objective

The goal of this analysis was to apply Principal Component Analysis to reduce the dimensionality of customer behavioral data. By identifying and retaining the principal components that explain the majority of the variance, PCA simplifies the data structure while preserving meaningful information for business interpretation and future analysis.

## Data Standardization

All 12 numerical variables were standardized using scikit-learn's StandardScaler before PCA was applied.

Standardization was necessary because:

- The variables measure different quantities using different units, including currency, counts, percentages, and days.
- PCA is sensitive to scale, so variables with larger numerical magnitudes could otherwise have a disproportionate influence on the principal components.
- Standardization places the variables on a comparable scale before PCA is performed.

Verification of the standardized data showed that the means were approximately 0 and the standard deviations were approximately 1.

## Correlation Patterns Observed

The Pearson correlation analysis revealed several important relationships among the customer variables.

- **Spending-related variables:** Annual_Income, Annual_Spend, Purchases_12M, Avg_Order_Value, and Loyalty_Points showed strong positive relationships with one another. These relationships suggest that several measures of customer purchasing behavior and value contain overlapping information.

- **Digital engagement variables:** Website_Visits_3M, App_Sessions_3M, and Email_Click_Rate_Pct showed related patterns and represent customer interaction with digital channels.

- **Purchase recency:** Days_Since_Last_Purchase was negatively related to several spending-related variables. This suggests that customers who purchased more recently also tended to demonstrate stronger purchasing behavior.

- **Customer satisfaction and returns:** Satisfaction_Score showed a negative relationship with Return_Rate_Pct, suggesting that higher satisfaction tends to be associated with lower product-return activity.

These correlations support the use of PCA because several variables appear to contain related or overlapping information.

## Principal Components Retained

To retain at least 80% of the total variance, three principal components were selected.

- **PC1:** 54.44% of variance explained
- **PC1 + PC2:** 71.15% cumulative variance explained
- **PC1 + PC2 + PC3:** 84.40% cumulative variance explained

Reducing the original 12 numerical variables to three principal components therefore retained 84.40% of the information represented by the total variance in the standardized dataset.

## Business Interpretation of Retained Components

### PC1: Customer Value, Spending, and Loyalty

PC1 explained 54.44% of the total variance.

The largest observed loadings included:

- Annual_Spend: 0.3610
- Purchases_12M: 0.3607
- Loyalty_Points: 0.3463

These variables indicate that PC1 represents a dimension related to customer spending, purchasing activity, and loyalty.

Higher positive scores on PC1 generally indicate customers with stronger spending, purchasing, and loyalty characteristics, while lower scores represent customers with lower levels of these characteristics.

From a business perspective, PC1 can therefore be interpreted as a measure of overall customer value and purchasing strength.

### PC2: Digital Engagement and Interaction

PC2 explained 16.71% of the total variance.

The largest observed loadings included:

- Email_Click_Rate_Pct: 0.5205
- App_Sessions_3M: 0.5080
- Website_Visits_3M: 0.4451

These variables indicate that PC2 primarily captures customer activity across digital channels.

Customers with higher PC2 scores tend to demonstrate greater engagement through email, mobile-app activity, and website visits.

From a business perspective, this component can help distinguish digitally active customers from customers who interact less frequently through online channels.

### PC3: Customer Service Intensity, Returns, and Satisfaction

PC3 explained 13.25% of the total variance.

The largest observed loadings included:

- Return_Rate_Pct: 0.5286
- Support_Contacts_12M: 0.4835
- Satisfaction_Score: -0.4049

These loadings suggest that PC3 represents customer service intensity, return activity, and satisfaction.

Higher positive PC3 scores are associated with more returns and customer-support contacts, while Satisfaction_Score loads in the opposite direction.

From a business perspective, this component may help identify customers experiencing greater service friction compared with customers who report stronger satisfaction and require fewer support interactions.

## PCA Scatterplot Interpretation

The PCA scatterplot displays customers using PC1 on the x-axis and PC2 on the y-axis, with Customer_Segment used to distinguish the customer groups.

Because PC1 primarily represents customer value and purchasing behavior while PC2 represents digital engagement, the plot provides a two-dimensional view of how customers differ across these characteristics.

The visualization can be used to observe whether customer segments tend to occupy different areas of the PCA space or whether there is overlap between groups.

Overlap would suggest that customers assigned to different segments may still share similar spending and digital-engagement characteristics, while greater separation would suggest more distinct behavioral profiles.

## Conclusion

Principal Component Analysis successfully reduced 12 related customer variables to three interpretable principal components while retaining 84.40% of the total variance.

The results suggest that major patterns in the customer data can be summarized through three broad dimensions:

1. Customer value, spending, and loyalty
2. Digital engagement and interaction
3. Customer service intensity, returns, and satisfaction

These reduced dimensions provide a simpler way to understand major patterns in customer behavior and can support future customer segmentation or related business analysis.

Overall, PCA demonstrated how a larger set of related customer measures can be summarized into a smaller number of meaningful dimensions while preserving most of the information contained in the original numerical variables.