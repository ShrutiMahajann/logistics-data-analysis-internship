# ============================================================
# WEEK 2: DATA COLLECTION, CLEANING & PREPROCESSING
# Project: Logistics Delivery Performance Analysis
# Dataset: Delivery_Logistics.csv
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

# The Python file is inside Week2_Data_Preprocessing,
# while the CSV file is in the main repository folder.
file_path = "../Delivery_Logistics.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# 2. DATASET OVERVIEW
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe(include="all"))


# ------------------------------------------------------------
# 3. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing_Values": missing_values,
    "Missing_Percentage": missing_percentage.round(2)
})

print(missing_report)

total_missing = df.isnull().sum().sum()

print("\nTotal missing values:", total_missing)

if total_missing == 0:
    print("No missing values were found.")
else:
    print("Missing values require treatment.")


# ------------------------------------------------------------
# 4. CHECK DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE RECORD ANALYSIS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")
else:
    print("No duplicate rows found.")


# ------------------------------------------------------------
# 5. CHECK DELIVERY ID DUPLICATES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DELIVERY ID CHECK")
print("=" * 60)

duplicate_delivery_ids = df["delivery_id"].duplicated().sum()

print("Repeated delivery IDs:", duplicate_delivery_ids)

# Note:
# A repeated delivery_id is reported but not automatically deleted,
# because duplicate IDs do not necessarily mean duplicate records.


# ------------------------------------------------------------
# 6. CLEAN TEXT / CATEGORICAL COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CATEGORICAL DATA CLEANING")
print("=" * 60)

categorical_columns = [
    "delivery_partner",
    "package_type",
    "vehicle_type",
    "delivery_mode",
    "region",
    "weather_condition",
    "delayed",
    "delivery_status"
]

for column in categorical_columns:
    # Convert to string, remove extra spaces and standardize case
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

print("Categorical columns cleaned successfully.")

print("\nUnique values after cleaning:")

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].unique())


# ------------------------------------------------------------
# 7. FIX DELIVERY TIME COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DELIVERY TIME FORMAT CLEANING")
print("=" * 60)

time_columns = [
    "delivery_time_hours",
    "expected_time_hours"
]

# The dataset stores these hour values in a datetime-like
# text format such as:
# 1970-01-01 00:00:00.000000008
#
# The nanosecond portion represents the original hour value
# in this dataset.

for column in time_columns:

    parsed_time = pd.to_datetime(
        df[column],
        errors="coerce"
    )

    # Extract the nanosecond component as the hour value
    df[column] = parsed_time.dt.nanosecond.astype(float)

print("Delivery-time columns converted to numeric hours.")

print("\nDelivery time range:")
print(df["delivery_time_hours"].describe())

print("\nExpected time range:")
print(df["expected_time_hours"].describe())


# ------------------------------------------------------------
# 8. CONVERT NUMERICAL COLUMNS TO NUMERIC FORMAT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL DATA TYPE CONVERSION")
print("=" * 60)

numeric_columns = [
    "delivery_id",
    "distance_km",
    "package_weight_kg",
    "delivery_time_hours",
    "expected_time_hours",
    "delivery_rating",
    "delivery_cost"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

print("Numerical columns converted successfully.")


# ------------------------------------------------------------
# 9. HANDLE MISSING VALUES CREATED DURING CONVERSION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("POST-CONVERSION MISSING VALUE CHECK")
print("=" * 60)

conversion_missing = df[numeric_columns].isnull().sum()

print(conversion_missing)

# Fill numerical missing values using the median.
# Median is less affected by extreme values than the mean.

for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(
            df[column].median()
        )

# Fill categorical missing values using the mode.

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

print("\nMissing values after treatment:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 10. CHECK NUMERICAL RANGES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL RANGE VALIDATION")
print("=" * 60)

# Delivery rating should normally be between 1 and 5.
invalid_ratings = (
    (df["delivery_rating"] < 1) |
    (df["delivery_rating"] > 5)
).sum()

print("Invalid delivery ratings:", invalid_ratings)

# Distance cannot be negative.
invalid_distance = (
    df["distance_km"] < 0
).sum()

print("Negative distance values:", invalid_distance)

# Package weight cannot be negative.
invalid_weight = (
    df["package_weight_kg"] < 0
).sum()

print("Negative package weight values:", invalid_weight)

# Delivery cost cannot be negative.
invalid_cost = (
    df["delivery_cost"] < 0
).sum()

print("Negative delivery cost values:", invalid_cost)


# ------------------------------------------------------------
# 11. OUTLIER DETECTION USING IQR
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("OUTLIER DETECTION USING IQR")
print("=" * 60)

outlier_columns = [
    "distance_km",
    "package_weight_kg",
    "delivery_time_hours",
    "expected_time_hours",
    "delivery_cost"
]

outlier_summary = {}

for column in outlier_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    outliers = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    )

    outlier_count = outliers.sum()

    outlier_summary[column] = {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Outlier_Count": outlier_count
    }

outlier_report = pd.DataFrame(outlier_summary).T

print(outlier_report)


# ------------------------------------------------------------
# 12. OUTLIER TREATMENT
# ------------------------------------------------------------

# Instead of deleting potentially valid logistics observations,
# extreme values are capped using IQR boundaries.

for column in outlier_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    df[column] = df[column].clip(
        lower=lower_bound,
        upper=upper_bound
    )

print("\nOutlier treatment completed using IQR capping.")


# ------------------------------------------------------------
# 13. CREATE DELIVERY DELAY DIFFERENCE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DELIVERY TIME FEATURE CREATION")
print("=" * 60)

df["delay_hours"] = (
    df["delivery_time_hours"] -
    df["expected_time_hours"]
)

print("New column 'delay_hours' created.")

print("\nDelay hours summary:")
print(df["delay_hours"].describe())


# ------------------------------------------------------------
# 14. VALIDATE DELAY FLAG
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DELAY FLAG VALIDATION")
print("=" * 60)

print("Delayed values:")
print(df["delayed"].value_counts())

print("\nDelivery status values:")
print(df["delivery_status"].value_counts())


# ------------------------------------------------------------
# 15. NORMALIZATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL FEATURE NORMALIZATION")
print("=" * 60)

# Normalization scales selected numerical variables between 0 and 1.

scaling_columns = [
    "distance_km",
    "package_weight_kg",
    "delivery_time_hours",
    "expected_time_hours",
    "delivery_cost"
]

scaler = MinMaxScaler()

normalized_data = scaler.fit_transform(
    df[scaling_columns]
)

normalized_columns = [
    column + "_normalized"
    for column in scaling_columns
]

df[normalized_columns] = normalized_data

print("Normalization completed.")

print("\nNormalized columns:")
print(normalized_columns)


# ------------------------------------------------------------
# 16. FINAL DATA QUALITY CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL DATA QUALITY CHECK")
print("=" * 60)

print("Final dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nFinal data types:")
print(df.dtypes)


# ------------------------------------------------------------
# 17. SAVE CLEANED DATASET
# ------------------------------------------------------------

output_file = "../Delivery_Logistics_cleaned.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("Cleaned dataset saved as:")
print(output_file)

print("\nFinal dataset shape:", df.shape)

print("\nWeek 2 preprocessing completed successfully.")
