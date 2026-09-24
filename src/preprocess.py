import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/raw/student_performance.csv"
OUTPUT = "data/processed/student_processed.csv"


def preprocess_data():

    df = pd.read_csv(INPUT)

    print("Dataset shape:", df.shape)

    # Remove duplicate records
    df = df.drop_duplicates()

    # Handle missing values
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())

    categorical_columns = df.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    # Convert categorical variables
    df = pd.get_dummies(
        df,
        columns=[
            "gender",
            "extracurricular",
            "internet_access",
            "parental_support"
        ],
        drop_first=True
    )

    df.to_csv(OUTPUT, index=False)

    print("Processed dataset saved to:", OUTPUT)
    print("Processed shape:", df.shape)


if __name__ == "__main__":
    preprocess_data()