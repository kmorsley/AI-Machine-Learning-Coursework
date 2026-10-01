import pandas as pd

# Load the raw customer dataset
df = pd.read_csv("Homework2_Customer_Data_Raw.csv")

print("\n===== FIRST FIVE ROWS =====")
print(df.head())

print("\n===== DATASET SIZE =====")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== NUMERICAL DESCRIPTIVE STATISTICS =====")
print(df.describe())

print("\n===== CATEGORICAL UNIQUE VALUES =====")

categorical_columns = [
    "Membership_Type",
    "Region",
    "Preferred_Device",
    "Marketing_Channel"
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].unique())

    print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
duplicate_count = df.duplicated().sum()
print("Number of duplicate rows:", duplicate_count)

print("\n===== INVALID AGE VALUES =====")
invalid_age = df[(df["Age"] < 0) | (df["Age"] > 100)]
print(invalid_age[["Customer_ID", "Age"]])

print("\n===== INVALID SATISFACTION SCORES =====")
invalid_satisfaction = df[
    (df["Satisfaction_Score"] < 1) |
    (df["Satisfaction_Score"] > 5)
]
print(invalid_satisfaction[["Customer_ID", "Satisfaction_Score"]])

print("\n===== ALL CATEGORICAL VALUES =====")

for column in [
    "Membership_Type",
    "Region",
    "Preferred_Device",
    "Marketing_Channel"
]:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))

    print("\n===== START DATA CLEANING =====")

# Make a copy so the raw data stays unchanged
clean_df = df.copy()

# Remove exact duplicate rows
rows_before = len(clean_df)
duplicate_count = clean_df.duplicated().sum()
clean_df = clean_df.drop_duplicates()
rows_after = len(clean_df)

print("\nDuplicate rows detected:", duplicate_count)
print("Rows before removing duplicates:", rows_before)
print("Rows after removing duplicates:", rows_after)

# Standardize categorical values
clean_df["Membership_Type"] = (
    clean_df["Membership_Type"]
    .str.strip()
    .str.title()
)

clean_df["Region"] = (
    clean_df["Region"]
    .str.strip()
    .str.title()
)

clean_df["Preferred_Device"] = (
    clean_df["Preferred_Device"]
    .str.strip()
    .str.title()
)

clean_df["Marketing_Channel"] = (
    clean_df["Marketing_Channel"]
    .str.strip()
    .str.title()
)

print("\n===== CLEANED CATEGORICAL VALUES =====")

for column in [
    "Membership_Type",
    "Region",
    "Preferred_Device",
    "Marketing_Channel"
]:
    print(f"\n{column}:")
    print(clean_df[column].value_counts(dropna=False))

# Convert invalid Age values to missing
clean_df.loc[
    (clean_df["Age"] < 0) | (clean_df["Age"] > 100),
    "Age"
] = pd.NA

# Convert invalid Satisfaction_Score values to missing
clean_df.loc[
    (clean_df["Satisfaction_Score"] < 1) |
    (clean_df["Satisfaction_Score"] > 5),
    "Satisfaction_Score"
] = pd.NA

print("\n===== MISSING VALUES BEFORE TREATMENT =====")
print(clean_df.isnull().sum())

# Fill missing numerical values with median
numerical_columns = clean_df.select_dtypes(include="number").columns

for column in numerical_columns:
    clean_df[column] = clean_df[column].fillna(
        clean_df[column].median()
    )

# Fill missing categorical values with mode
categorical_columns = clean_df.select_dtypes(include="object").columns

for column in categorical_columns:
    if clean_df[column].isnull().any():
        clean_df[column] = clean_df[column].fillna(
            clean_df[column].mode()[0]
        )

print("\n===== MISSING VALUES AFTER TREATMENT =====")
print(clean_df.isnull().sum())

# Save cleaned dataset
clean_df.to_csv("cleaned_customer_data.csv", index=False)

print("\ncleaned_customer_data.csv saved successfully.")

print("\n===== OUTLIER DETECTION USING IQR =====")

outlier_columns = [
    "Annual_Income",
    "Avg_Session_Minutes",
    "Total_Spend_Last_6M"
]

outlier_counts = {}

for column in outlier_columns:
    Q1 = clean_df[column].quantile(0.25)
    Q3 = clean_df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = clean_df[
        (clean_df[column] < lower_bound) |
        (clean_df[column] > upper_bound)
    ]

    outlier_counts[column] = len(outliers)

    print(f"\n{column}")
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)
    print("Number of Outliers:", len(outliers))

    if len(outliers) > 0:
        print("Potential Outliers:")
        print(outliers[["Customer_ID", column]])

        from sklearn.preprocessing import StandardScaler

print("\n===== FEATURE SCALING =====")

scaling_columns = [
    "Age",
    "Annual_Income",
    "Website_Visits_Last_Month",
    "Avg_Session_Minutes",
    "Purchases_Last_6M",
    "Total_Spend_Last_6M"
]

# Create a copy so the cleaned dataset stays unchanged
standardized_df = clean_df.copy()

# Create the StandardScaler object
scaler = StandardScaler()

# Standardize only the selected numerical columns
standardized_df[scaling_columns] = scaler.fit_transform(
    clean_df[scaling_columns]
)

# Save standardized dataset
standardized_df.to_csv(
    "standardized_customer_data.csv",
    index=False
)

print("standardized_customer_data.csv saved successfully.")

print("\nStandardized values preview:")
print(standardized_df[scaling_columns].head())

print("\n===== ONE-HOT ENCODING =====")

encoding_columns = [
    "Region",
    "Preferred_Device",
    "Marketing_Channel"
]

encoded_df = pd.get_dummies(
    clean_df,
    columns=encoding_columns,
    dtype=int
)

encoded_df.to_csv(
    "encoded_customer_data.csv",
    index=False
)

print("encoded_customer_data.csv saved successfully.")

print("\nEncoded columns preview:")
print(encoded_df.head())

import matplotlib.pyplot as plt

print("\n===== DATA VISUALIZATIONS =====")

# 1. Histogram: Total Spend Last 6 Months
plt.figure()
plt.hist(clean_df["Total_Spend_Last_6M"], bins=10)
plt.title("Distribution of Total Spending in the Last 6 Months")
plt.xlabel("Total Spend Last 6 Months")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("total_spend_histogram.png")
plt.close()

print("total_spend_histogram.png saved successfully.")

# 2. Boxplot: Annual Income
plt.figure()
plt.boxplot(clean_df["Annual_Income"].dropna())
plt.title("Annual Income Distribution")
plt.ylabel("Annual Income")
plt.tight_layout()
plt.savefig("annual_income_boxplot.png")
plt.close()

print("annual_income_boxplot.png saved successfully.")

# 3. Bar Chart: Membership Type
membership_counts = clean_df["Membership_Type"].value_counts()

plt.figure()
plt.bar(membership_counts.index, membership_counts.values)
plt.title("Number of Customers by Membership Type")
plt.xlabel("Membership Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("membership_bar_chart.png")
plt.close()

print("membership_bar_chart.png saved successfully.")

# 4. Scatter Plot: Annual Income vs Total Spend
plt.figure()
plt.scatter(
    clean_df["Annual_Income"],
    clean_df["Total_Spend_Last_6M"]
)
plt.title("Annual Income vs Total Spending")
plt.xlabel("Annual Income")
plt.ylabel("Total Spend Last 6 Months")
plt.tight_layout()
plt.savefig("income_spend_scatterplot.png")
plt.close()

print("income_spend_scatterplot.png saved successfully.")

print("\n===== PREPROCESSING SUMMARY =====")

# Count invalid values identified before cleaning
invalid_age_count = len(invalid_age)
invalid_satisfaction_count = len(invalid_satisfaction)

# Total missing values before treatment
missing_before = df.isnull().sum().sum()

# Total missing values after treatment
missing_after = clean_df.isnull().sum().sum()

summary_data = {
    "Metric": [
        "Original Number of Observations",
        "Duplicate Rows Detected",
        "Final Number of Observations",
        "Missing Values Detected",
        "Missing Values Remaining After Preprocessing",
        "Invalid Age Values Identified",
        "Invalid Satisfaction Scores Identified",
        "Annual Income Outliers",
        "Average Session Minutes Outliers",
        "Total Spend Last 6M Outliers"
    ],
    "Value": [
        len(df),
        duplicate_count,
        len(clean_df),
        missing_before,
        missing_after,
        invalid_age_count,
        invalid_satisfaction_count,
        outlier_counts["Annual_Income"],
        outlier_counts["Avg_Session_Minutes"],
        outlier_counts["Total_Spend_Last_6M"]
    ]
}

summary_df = pd.DataFrame(summary_data)

summary_df.to_csv(
    "preprocessing_summary.csv",
    index=False
)

print(summary_df)

print("\npreprocessing_summary.csv saved successfully.")