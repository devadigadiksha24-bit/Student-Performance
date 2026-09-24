import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


DATA_PATH = "data/processed/student_processed.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop("final_score", axis=1)
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

mlflow.set_experiment("Student-Performance-Prediction")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = []

best_model = None
best_r2 = float("-inf")
best_name = None

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = mean_squared_error(
            y_test,
            predictions
        ) ** 0.5

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        mlflow.log_param(
            "model",
            name
        )

        mlflow.log_metric(
            "RMSE",
            rmse
        )

        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "R2",
            r2
        )

        mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=[
        "sklearn.tree._tree.Tree"
    ]
)

        results.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        print("\nModel:", name)
        print("RMSE:", round(rmse, 3))
        print("MAE:", round(mae, 3))
        print("R2:", round(r2, 3))

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name


joblib.dump(
    best_model,
    f"{MODEL_DIR}/best_model.pkl"
)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print(results_df)

print("\nBest Model:", best_name)
print("Best R2:", round(best_r2, 3))

print("\nBest model saved to:")
print("models/best_model.pkl")