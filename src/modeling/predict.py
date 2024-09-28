from joblib import load

# Define the paths for the vectorizer and model
vectorizer_path = r"C:\Users\sd876\OneDrive\Desktop\AI-Content-Verifier\Ai-Content-Verifier\src\saved_models\tfidf_vectorizer.pkl"
model_path = r"C:\Users\sd876\OneDrive\Desktop\AI-Content-Verifier\Ai-Content-Verifier\src\saved_models\best_model.pkl"

def load_model_and_vectorizer():
    try:
        vectorizer = load(vectorizer_path)
        model = load(model_path)
        return vectorizer, model
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please check if the model and vectorizer files exist at the specified paths.")
        exit(1)

def predict_text(text, vectorizer, model):
    text_transformed = vectorizer.transform([text])
    prediction = model.predict(text_transformed)
    confidence = model.predict_proba(text_transformed)[0]
    
    if prediction[0] == 1:
        result = "AI-generated"
        confidence_score = confidence[1]
    else:
        result = "Human-generated"
        confidence_score = confidence[0]
    
    return result, confidence_score

def get_multiline_input():
    print("Enter your text (press Ctrl+D on Unix or Ctrl+Z on Windows followed by Enter to finish):")
    lines = []
    while True:
        try:
            line = input()
            lines.append(line)
        except EOFError:
            break
    return '\n'.join(lines)

def main():
    vectorizer, model = load_model_and_vectorizer()
    
    print("Welcome to the AI Content Verifier!")
    print("This tool will help you classify text as either AI-generated or Human-generated.")
    
    while True:
        sample_text = get_multiline_input()
        
        if not sample_text.strip():
            print("No text was entered. Exiting the program.")
            break
        
        result, confidence = predict_text(sample_text, vectorizer, model)
        print(f"\nPrediction: {result}")
        print(f"Confidence: {confidence:.2f}\n")
        
        continue_choice = input("Do you want to verify another text? (y/n): ").strip().lower()
        if continue_choice != 'y':
            break
    
    print("Thank you for using the AI Content Verifier!")

if __name__ == "__main__":
    main()