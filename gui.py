import sys
import os
import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Define the paths for your vectorizer and model
vectorizer_path = r"C:\Users\sd876\OneDrive\Desktop\AI-Content-Verifier\Ai-Content-Verifier\src\saved_models\tfidf_vectorizer.pkl"
model_path = r"C:\Users\sd876\OneDrive\Desktop\AI-Content-Verifier\Ai-Content-Verifier\src\saved_models\best_model.pkl"

# Load the vectorizer and model
vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)

# AI Content Detector Application (similar to uploaded design)
class AIContentDetectorApp:
    def __init__(self, parent):
        self.frame = tk.Frame(parent, bg="#F2F2F2")
        self.frame.pack(fill='both', expand=True)

        # Title bar
        title_label = tk.Label(self.frame, text="AI Content Verifier", font=("Helvetica", 24, "bold"), bg="#007AFF", fg="#FFFFFF")
        title_label.pack(fill="x", pady=20)

        # Instructions
        instructions_label = tk.Label(self.frame, text="Paste the text content below:", font=("Helvetica", 14), bg="#F2F2F2", fg="#000000")
        instructions_label.pack(pady=10)

        # Text input area
        self.text_area = tk.Text(self.frame, height=8, width=70, font=("Helvetica", 14), bg="#FFFFFF", fg="#000000", borderwidth=1, relief="solid")
        self.text_area.pack(padx=20, pady=10)

        # Buttons
        button_frame = tk.Frame(self.frame, bg="#F2F2F2")
        button_frame.pack(pady=10)

        self.check_button = tk.Button(button_frame, text="Check Content", command=self.predict_content, bg="#28A745", fg="#FFFFFF", font=("Helvetica", 12, "bold"), padx=20, pady=10, relief="flat")
        self.check_button.grid(row=0, column=0, padx=10)

        self.reset_button = tk.Button(button_frame, text="Reset", command=self.reset_text_area, bg="#FF3B30", fg="#FFFFFF", font=("Helvetica", 12, "bold"), padx=20, pady=10, relief="flat")
        self.reset_button.grid(row=0, column=1, padx=10)

        # Status label
        self.result_label = tk.Label(self.frame, text="Prediction: -\nConfidence: -", font=("Helvetica", 16), bg="#F2F2F2", fg="#000000")
        self.result_label.pack(pady=20)

        # Footer with Email ID
        footer_label = tk.Label(self.frame, text="© 2024 AI Content Detector | Contact: support@contentdetector.com", font=("Helvetica", 10), bg="#007AFF", fg="#FFFFFF")
        footer_label.pack(fill="x", side=tk.BOTTOM, pady=10)

    def predict_content(self):
        input_text = self.text_area.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showwarning("Input Error", "Please enter some text.")
            return

        # Preprocess the text (lowercase as an example)
        preprocessed_text = input_text.lower()

        # Vectorize the input text
        vectorized_text = vectorizer.transform([preprocessed_text])

        # Predict using the loaded model
        prediction = model.predict(vectorized_text)[0]
        confidence = np.max(model.predict_proba(vectorized_text)) * 100

        result_text = "AI-Generated" if prediction == 1 else "Human-Generated"

        # Update result display
        self.result_label.config(text=f"Prediction: {result_text}\nConfidence: {confidence:.2f}%")

    def reset_text_area(self):
        self.text_area.delete("1.0", tk.END)
        self.result_label.config(text="Prediction: -\nConfidence: -")

# Main Application
class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Content Detector")
        self.root.geometry("800x600")
        self.root.configure(bg="#F2F2F2")

        # Create the AI Content Detector tab
        self.ai_content_Verifier_tab = AIContentDetectorApp(self.root)

# Main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()
