# ============================================================
# STUDENT PERFORMANCE PREDICTION - MODEL TRAINING
# MLOps Project
# ============================================================

import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

DATA_PATH = "data/processed/student_processed.csv"

MODEL_DIR = "models"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

COMPARISON_PATH = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)

EXPERIMENT_NAME = "Student-Performance-Prediction"

RANDOM_STATE = 42

TARGET = "final_score"


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

# SQLite works on both:
# - Windows
# - GitHub Actions Linux

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# START TRAINING
# ============================================================

print("\n" + "=" * 65)
print("STUDENT PERFORMANCE PREDICTION - MODEL TRAINING")
print("=" * 65)


# ============================================================
# CHECK DATASET
# ============================================================

print("\nLoading processed dataset...")

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"\nProcessed dataset not found:\n"
        f"{DATA_PATH}\n\n"
        f"Run the preprocessing step first:\n"
        f"python src/preprocess.py"
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset shape: {df.shape}"
)

print(
    f"Rows: {df.shape[0]}"
)

print(
    f"Columns: {df.shape[1]}"
)


# ============================================================
# CHECK TARGET COLUMN
# ============================================================

if TARGET not in df.columns:

    raise ValueError(
        f"\nTarget column '{TARGET}' "
        f"was not found in the dataset.\n"
        f"Available columns:\n"
        f"{list(df.columns)}"
    )


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]


print("\n" + "-" * 65)
print("FEATURE INFORMATION")
print("-" * 65)

print(
    f"Number of features: {X.shape[1]}"
)

print(
    f"Features: {list(X.columns)}"
)

print(
    f"\nTarget: {TARGET}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)


print("\n" + "-" * 65)
print("TRAIN / TEST SPLIT")
print("-" * 65)

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples:  {len(X_test)}"
)


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE
        )
}


# ============================================================
# TRAIN MODELS
# ============================================================

results = []

best_model = None

best_model_name = None

best_rmse = float("inf")


print("\n" + "=" * 65)
print("MODEL TRAINING")
print("=" * 65)


for model_name, model in models.items():

    print("\n" + "-" * 65)

    print(
        f"TRAINING MODEL: {model_name}"
    )

    print("-" * 65)

    try:

        # ----------------------------------------------------
        # START MLFLOW RUN
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name=model_name
        ):

            # ------------------------------------------------
            # TRAIN
            # ------------------------------------------------

            model.fit(
                X_train,
                y_train
            )

            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            predictions = model.predict(
                X_test
            )

            # ------------------------------------------------
            # CALCULATE METRICS
            # ------------------------------------------------

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            mse = mean_squared_error(
                y_test,
                predictions
            )

            rmse = mse ** 0.5

            r2 = r2_score(
                y_test,
                predictions
            )

            # ------------------------------------------------
            # DISPLAY METRICS
            # ------------------------------------------------

            print(
                f"MAE  : {mae:.4f}"
            )

            print(
                f"MSE  : {mse:.4f}"
            )

            print(
                f"RMSE : {rmse:.4f}"
            )

            print(
                f"R²   : {r2:.4f}"
            )

            # ------------------------------------------------
            # LOG BASIC PARAMETERS
            # ------------------------------------------------

            mlflow.log_param(
                "model_name",
                model_name
            )

            mlflow.log_param(
                "random_state",
                RANDOM_STATE
            )

            mlflow.log_param(
                "test_size",
                0.20
            )

            # ------------------------------------------------
            # LOG MODEL PARAMETERS
            # ------------------------------------------------

            try:

                model_params = model.get_params()

                for parameter, value in model_params.items():

                    try:

                        mlflow.log_param(
                            parameter,
                            value
                        )

                    except Exception:
                        pass

            except Exception:
                pass


            # ------------------------------------------------
            # LOG METRICS
            # ------------------------------------------------

            mlflow.log_metric(
                "MAE",
                float(mae)
            )

            mlflow.log_metric(
                "MSE",
                float(mse)
            )

            mlflow.log_metric(
                "RMSE",
                float(rmse)
            )

            mlflow.log_metric(
                "R2",
                float(r2)
            )


            # ------------------------------------------------
            # LOG MODEL TO MLFLOW
            # ------------------------------------------------

            try:

                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    skops_trusted_types=[
                        "sklearn.tree._tree.Tree"
                    ]
                )

            except Exception as model_error:

                warnings.warn(
                    f"MLflow model logging failed "
                    f"for {model_name}: "
                    f"{model_error}"
                )


            # ------------------------------------------------
            # STORE RESULTS
            # ------------------------------------------------

            results.append(
                {
                    "model": model_name,
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2": r2
                }
            )


            # ------------------------------------------------
            # SELECT BEST MODEL
            # ------------------------------------------------

            if rmse < best_rmse:

                best_rmse = rmse

                best_model = model

                best_model_name = model_name


    except Exception as error:

        print(
            f"\nWARNING: {model_name} failed."
        )

        print(
            f"Error: {error}"
        )

        print(
            "Continuing with the next model..."
        )

        continue


# ============================================================
# CHECK WHETHER TRAINING WAS SUCCESSFUL
# ============================================================

if len(results) == 0:

    raise RuntimeError(
        "\nAll models failed during training."
    )


if best_model is None:

    raise RuntimeError(
        "\nNo best model could be selected."
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


# Lower RMSE = better prediction error.
# Sort only for displaying the comparison.

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)


print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

results_df.to_csv(
    COMPARISON_PATH,
    index=False
)


print(
    f"\nModel comparison saved to:"
)

print(
    COMPARISON_PATH
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    BEST_MODEL_PATH
)


print(
    f"\nBest model: "
    f"{best_model_name}"
)

print(
    f"Best RMSE: "
    f"{best_rmse:.4f}"
)

print(
    f"Best model saved to:"
)

print(
    BEST_MODEL_PATH
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 65)

print(
    f"""
Best Model       : {best_model_name}
Best RMSE        : {best_rmse:.4f}
Model File       : {BEST_MODEL_PATH}
Comparison File  : {COMPARISON_PATH}
MLflow Database  : mlflow.db
Experiment       : {EXPERIMENT_NAME}
"""
)

print("=" * 65)