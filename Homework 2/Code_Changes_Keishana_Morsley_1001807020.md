# GitHub Copilot Code Review
Keishana Morsley
UTA ID: 1001807020

## Change 1: Simplified the Data Cleaning Logic
Copilot suggested a longer approach for cleaning categorical values. I simplified the code by using string methods such as `.str.strip()` and `.str.title()`.

This made the code shorter, easier to read, and consistently corrected capitalization and extra spaces.

## Change 2: Kept Outliers Instead of Automatically Removing Them
I reviewed the outlier detection logic and made sure the program only identified and reported outliers rather than deleting them.

This was important because unusual observations may still represent valid customer behavior and the assignment specifically required investigation rather than automatic removal.

## Change 3: Kept the Cleaned Dataset Separate from the Standardized Dataset
I created a copy of the cleaned dataset before applying StandardScaler instead of replacing the original cleaned values.

This preserved the original cleaned dataset while also creating a separate standardized dataset for later analysis.

## Additional Review
I reviewed the generated code for unnecessary repetition and made sure each section had a clear purpose. I also added descriptive print statements so the output would be easier to interpret.

The final program was organized into separate sections for inspection, cleaning, outlier detection, scaling, encoding, visualization, and summary creation.