import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Function to preprocess data
def preprocess_data(data):
    logging.info("NaN counts in required columns before dropping:")
    
    # Drop rows where either 'text' or 'label' is NaN
    nan_counts = data[['text', 'label']].isna().sum()
    logging.info(nan_counts)
    data = data.dropna(subset=['text', 'label'])

    # Combine the text fields that might be relevant for classification
    # Adjust the column names to match your data, e.g., combining 'text', 'title', 'abstract'
    data['combined_text'] = (
        data['text'].fillna('') + ' ' + 
        data['title'].fillna('') + ' ' + 
        data['abstract'].fillna('')
    )
    
    # Filter out rows where 'combined_text' is empty
    data = data[data['combined_text'].str.strip().astype(bool)]
    
    logging.info(f"Shape after filtering: {data.shape}")
    return data

# Main function to load data, preprocess, train the model, and save the model
def main():
    logging.basicConfig(level=logging.INFO)
    
    # File paths to the datasets
    file_paths = [
        r"C:\Users\sd876\OneDrive\Desktop\AI-Content-Verifier\Ai-Content-Verifier\data\raw\data_set.csv",
        r"C:\Users\sd876\OneDrive\Desktop\AI-Content-Verifier\Ai-Content-Verifier\data\raw\LLM.csv",
        r"C:\Users\sd876\OneDrive\Desktop\AI-Content-Verifier\Ai-Content-Verifier\data\raw\train_essays_v1.csv",
        r"C:\Users\sd876\OneDrive\Desktop\AI-Content-Verifier\Ai-Content-Verifier\data\raw\train_from_LLM-Detect_AI-GT_1MNB-3SGD.csv"
    ]
    
    # Load and concatenate all CSVs
    logging.info("Loading data from files...")
    data_frames = [pd.read_csv(file) for file in file_paths]
    combined_data = pd.concat(data_frames, ignore_index=True)
    
    logging.info(f"Combined data shape: {combined_data.shape}")
    
    # Preprocess the data
    processed_data = preprocess_data(combined_data)
    
    if processed_data.empty:
        raise ValueError("Processed data is empty after preprocessing. Check your input data for issues.")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data['combined_text'], 
        processed_data['label'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Define a pipeline with TF-IDF and Logistic Regression
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)),
        ('logreg', LogisticRegression(max_iter=1000))
    ])
    
    # Train the model
    logging.info("Training the model...")
    model.fit(X_train, y_train)
    logging.info("Model training completed.")
    
    # Evaluate the model
    logging.info("Model evaluation:")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logging.info("\n" + report)
    
    # Save the trained model using joblib
    model_filename = 'model_logistic_regression.pkl'
    joblib.dump(model, model_filename)
    logging.info(f"Model saved to {model_filename}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
