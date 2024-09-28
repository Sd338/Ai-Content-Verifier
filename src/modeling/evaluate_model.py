import os
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample data (replace this with your actual data)
X_train = ["This is a sample text.", "Another example sentence."]
y_train = [0, 1]

# Define paths
vectorizer_path = r"C:\Users\sd876\OneDrive\Desktop\AI-Content-Verifier\Ai-Content-Verifier\src\saved_models\tfidf_vectorizer.pkl"
model_path = r"C:\Users\sd876\OneDrive\Desktop\AI-Content-Verifier\Ai-Content-Verifier\src\saved_models\best_model.pkl"

# Ensure the directory exists and has write permissions
def ensure_directory_exists_and_writable(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)  # Create the directory if it doesn't exist
            print(f"Directory created: {directory}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return False
    if not os.access(directory, os.W_OK):  # Check if the directory is writable
        print(f"Permission denied for writing to directory: {directory}")
        return False
    return True

# Train and save the model and vectorizer
def save_model_and_vectorizer(X_train, y_train, vectorizer_path, model_path):
    # Check if the directory paths are valid and writable
    if not ensure_directory_exists_and_writable(vectorizer_path) or not ensure_directory_exists_and_writable(model_path):
        print("Saving process aborted due to directory issues.")
        return

    # Initialize vectorizer and model
    vectorizer = TfidfVectorizer()
    model = LogisticRegression()

    # Fit vectorizer and model
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model.fit(X_train_tfidf, y_train)

    try:
        # Save the vectorizer
        dump(vectorizer, vectorizer_path)
        print(f"Vectorizer saved successfully at: {vectorizer_path}")

        # Save the model
        dump(model, model_path)
        print(f"Model saved successfully at: {model_path}")

    except PermissionError as pe:
        print(f"Permission error: {pe}")
    except Exception as e:
        print(f"Error saving files: {e}")

# Call the function to save the model and vectorizer
save_model_and_vectorizer(X_train, y_train, vectorizer_path, model_path)
