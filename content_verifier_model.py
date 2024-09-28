import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump
import time

# File paths
data_folder = r"C:\Users\sd876\OneDrive\Desktop\AI-Content-Verifier\Ai-Content-Verifier\data\processed"
model_path = r"C:\Users\sd876\OneDrive\Desktop\AI-Content-Verifier\Ai-Content-Verifier\src\saved_models\best_model.pkl"
vectorizer_path = r"C:\Users\sd876\OneDrive\Desktop\AI-Content-Verifier\Ai-Content-Verifier\src\saved_models\tfidf_vectorizer.pkl"

def read_csv_files(folder_path):
    dataframes = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                dataframes.append(df)
                print(f"Successfully read {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
    return pd.concat(dataframes, ignore_index=True) if dataframes else None

def clean_data(df):
    # Remove rows with NaN values
    df_cleaned = df.dropna(subset=['text', 'label']).copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Convert 'text' column to string type
    df_cleaned.loc[:, 'text'] = df_cleaned['text'].astype(str)
    
    # Remove empty strings
    df_cleaned = df_cleaned[df_cleaned['text'].str.strip() != '']
    
    # Ensure 'label' is numeric
    df_cleaned['label'] = pd.to_numeric(df_cleaned['label'], errors='coerce')
    df_cleaned = df_cleaned.dropna(subset=['label'])
    
    return df_cleaned

try:
    # Check if the data folder exists
    if not os.path.isdir(data_folder):
        print(f"Error: The folder {data_folder} does not exist.")
        exit(1)

    # Read and combine all CSV files
    df = read_csv_files(data_folder)
    
    if df is None or df.empty:
        print("No data was loaded. Please check your CSV files.")
        exit(1)

    print(f"Total rows in combined dataset: {len(df)}")

    # Clean the data
    df_cleaned = clean_data(df)
    print(f"Rows after cleaning: {len(df_cleaned)}")

    # Check if 'text' and 'label' columns exist
    if 'text' not in df_cleaned.columns or 'label' not in df_cleaned.columns:
        print("Error: 'text' or 'label' column is missing in the CSV files.")
        exit(1)

    X = df_cleaned['text']
    y = df_cleaned['label']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorization
    print("Starting vectorization...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_transformed = vectorizer.fit_transform(X_train)
    print("Vectorization completed.")

    # Model Training with Hyperparameter Tuning
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=3)

    print("Starting model training...")
    start_time = time.time()
    grid_search.fit(X_train_transformed, y_train)
    print(f"GridSearchCV fitting completed in {time.time() - start_time:.2f} seconds.")

    # Save the best model and vectorizer
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))  # Create directory if it doesn't exist

    best_model = grid_search.best_estimator_
    dump(best_model, model_path)
    dump(vectorizer, vectorizer_path)

    print("Best model and vectorizer saved successfully.")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
