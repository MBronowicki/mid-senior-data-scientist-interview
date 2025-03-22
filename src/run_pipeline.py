from src.data_ingestion import load_data
from src.vectorize_data import save_sparse_matrix
from src.feature_engineering import extract_features, feature_engineering_datetime
from src.data_preprocessing import (
    clean_data,
    convert_columns_dtypes,
    count_and_remove_non_english_rows,
    preprocess_text
)

def main():
    print("Starting data pipeline...")

    filepath = "./data/2025_data_to_explore.csv"

    # Data ingestion and cleaning
    df = load_data(filepath=filepath)
    df = clean_data(df)

    # Feature Engineering
    print("Feature engineering process...")
    df = extract_features(df, column="company_description")
    df = feature_engineering_datetime(df, datetime_column="created_at")

    # Preprocessing data
    print("Preprocess text started...")
    df = convert_columns_dtypes(df)
    df_cleaned = count_and_remove_non_english_rows(df, column="company_description")
    df_cleaned["company_description"] = df_cleaned["company_description"].apply(preprocess_text)

    # Save preprcessed data to csv
    df_cleaned.to_csv("./data/preprocessed_data.csv", index=False)

    # Create Vectorize text using TF-IDF and save as sparse matrix
    print("Tf-Idf vectorising and saving sparse matrix...")
    save_sparse_matrix(df_cleaned)
    print("Data pipeline finished!")

if __name__ == "__main__":
    main()