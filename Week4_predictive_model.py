# ============================================================
# WEEK 4: PREDICTIVE MODELING AND OPTIMIZATION IN LOGISTICS
# Project: Logistics Delivery Performance Analysis
# Target: Delivery Time Prediction
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

# The dataset is stored in the project's dataset folder.
df = pd.read_csv("../dataset/Delivery_Logistics.csv")

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. BASIC DATA INSPECTION
# ============================================================

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())


# ============================================================
# 3. CONVERT DELIVERY TIME VARIABLES
# ============================================================

# The source dataset stores hour values in a datetime-like format.
# Convert them into numerical hour values.

for column in ["delivery_time_hours", "expected_time_hours"]:

    parsed_time = pd.to_datetime(
        df[column],
        errors="coerce"
    )

    df[column] = parsed_time.dt.nanosecond.astype(float)


# ============================================================
# 4. DEFINE TARGET VARIABLE
# ============================================================

# Target variable:
# We want to predict the actual delivery time.

target = "delivery_time_hours"


# ============================================================
# 5. SELECT FEATURES
# ============================================================

features = [

    "distance_km",
    "package_weight_kg",
    "expected_time_hours",

    "delivery_partner",
    "package_type",
    "vehicle_type",
    "delivery_mode",
    "region",
    "weather_condition",

    "delivery_rating",
    "delivery_cost"
]


X = df[features]
y = df[target]


# ============================================================
# 6. DEFINE NUMERICAL FEATURES
# ============================================================

numeric_features = [

    "distance_km",
    "package_weight_kg",
    "expected_time_hours",
    "delivery_rating",
    "delivery_cost"

]


# ============================================================
# 7. DEFINE CATEGORICAL FEATURES
# ============================================================

categorical_features = [

    "delivery_partner",
    "package_type",
    "vehicle_type",
    "delivery_mode",
    "region",
    "weather_condition"

]


# ============================================================
# 8. DATA PREPROCESSING PIPELINE
# ============================================================

# Numerical preprocessing:
# - Fill missing values with median
# - Standardize numerical variables

numeric_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )

    ]
)


# Categorical preprocessing:
# - Fill missing values with most frequent category
# - Convert categories into numerical dummy variables

categorical_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]
)


# Combine preprocessing pipelines

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )

    ]
)


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 10. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Linear Regression":

        LinearRegression(),


    "Random Forest":

        RandomForestRegressor(

            n_estimators=60,

            max_depth=10,

            min_samples_leaf=3,

            random_state=42,

            n_jobs=-1

        )

}


# ============================================================
# 11. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

pipelines = {}


for model_name, model in models.items():

    print("\n===================================")
    print("Training:", model_name)
    print("===================================")


    # Create complete machine-learning pipeline

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )

        ]

    )


    # Train model

    pipeline.fit(
        X_train,
        y_train
    )


    # Make predictions

    predictions = pipeline.predict(
        X_test
    )


    # Calculate evaluation metrics

    mae = mean_absolute_error(
        y_test,
        predictions
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    # Store results

    results.append({

        "Model": model_name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2

    })


    pipelines[model_name] = pipeline


    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


# Lower RMSE is better.

results_df = results_df.sort_values(
    by="RMSE"
)


print("\n===================================")
print("MODEL COMPARISON")
print("===================================")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 13. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = pipelines[
    best_model_name
]


print(
    "\nSelected Best Model:",
    best_model_name
)


# ============================================================
# 14. FINAL PREDICTIONS
# ============================================================

best_predictions = best_model.predict(
    X_test
)


# ============================================================
# 15. FINAL MODEL EVALUATION
# ============================================================

final_mae = mean_absolute_error(
    y_test,
    best_predictions
)


final_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        best_predictions
    )
)


final_r2 = r2_score(
    y_test,
    best_predictions
)


print("\n===================================")
print("FINAL MODEL PERFORMANCE")
print("===================================")

print("Model:", best_model_name)

print(
    "Mean Absolute Error:",
    final_mae
)

print(
    "Root Mean Squared Error:",
    final_rmse
)

print(
    "R-squared:",
    final_r2
)


# ============================================================
# 16. FIVE-FOLD CROSS-VALIDATION
# ============================================================

print("\n===================================")
print("5-FOLD CROSS VALIDATION")
print("===================================")


kfold = KFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


cv_scores = cross_val_score(

    best_model,

    X,

    y,

    cv=kfold,

    scoring="neg_mean_squared_error",

    n_jobs=-1

)


cv_rmse = np.sqrt(
    -cv_scores
)


print(
    "Cross-validation RMSE values:"
)

print(cv_rmse)


print(
    "Mean Cross-validation RMSE:",
    cv_rmse.mean()
)


# ============================================================
# 17. ACTUAL VS PREDICTED VISUALIZATION
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(

    y_test,

    best_predictions,

    alpha=0.4

)


# Perfect prediction reference line

minimum = min(
    y_test.min(),
    best_predictions.min()
)

maximum = max(
    y_test.max(),
    best_predictions.max()
)


plt.plot(

    [minimum, maximum],

    [minimum, maximum]

)


plt.xlabel(
    "Actual Delivery Time (hours)"
)


plt.ylabel(
    "Predicted Delivery Time (hours)"
)


plt.title(
    "Actual vs Predicted Delivery Time"
)


plt.tight_layout()


plt.savefig(
    "actual_vs_predicted.png",
    dpi=200
)


plt.show()


# ============================================================
# 18. RESIDUAL ANALYSIS
# ============================================================

residuals = (

    y_test.to_numpy()

    - best_predictions

)


plt.figure(
    figsize=(8, 6)
)


plt.scatter(

    best_predictions,

    residuals,

    alpha=0.4

)


plt.axhline(
    0
)


plt.xlabel(
    "Predicted Delivery Time (hours)"
)


plt.ylabel(
    "Residual"
)


plt.title(
    "Prediction Residual Analysis"
)


plt.tight_layout()


plt.savefig(
    "prediction_residuals.png",
    dpi=200
)


plt.show()


# ============================================================
# 19. CREATE PREDICTION OUTPUT FILE
# ============================================================

prediction_results = X_test.copy()


prediction_results[
    "actual_delivery_time_hours"
] = y_test.values


prediction_results[
    "predicted_delivery_time_hours"
] = best_predictions


prediction_results[
    "prediction_error_hours"
] = residuals


prediction_results.to_csv(

    "Delivery_Logistics_predictions.csv",

    index=False

)


print(
    "\nPrediction results saved successfully."
)


# ============================================================
# 20. LOGISTICS OPTIMIZATION RECOMMENDATIONS
# ============================================================

print("\n===================================")
print("LOGISTICS OPTIMIZATION STRATEGIES")
print("===================================")


print(
    "1. Prioritize shipments with high predicted delivery times."
)


print(
    "2. Add additional delivery-time buffers for long-distance routes."
)


print(
    "3. Use weather information to identify shipments requiring proactive intervention."
)


print(
    "4. Allocate vehicles and delivery staff according to predicted workload."
)


print(
    "5. Monitor differences between predicted and actual delivery time."
)


print(
    "6. Compare delivery partner performance using prediction errors."
)


print(
    "7. Consider both predicted delivery time and transportation cost "
    "when selecting operational alternatives."
)


print(
    "8. Retrain the model periodically as new logistics data becomes available."
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n===================================")
print("WEEK 4 PROJECT SUMMARY")
print("===================================")

print(
    "Prediction Target:",
    target
)

print(
    "Selected Model:",
    best_model_name
)

print(
    "MAE:",
    final_mae
)

print(
    "RMSE:",
    final_rmse
)

print(
    "R2:",
    final_r2
)

print(
    "Mean CV RMSE:",
    cv_rmse.mean()
)

print(
    "\nWeek 4 predictive modeling completed successfully."
)
