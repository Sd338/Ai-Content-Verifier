import os
import pandas as pd

# Define paths for raw and processed data directories
raw_data_dir = r'C:\Users\sd876\OneDrive\Desktop\AI-Content-Verifier\Ai-Content-Verifier\data\raw'
processed_data_dir = r'C:\Users\sd876\OneDrive\Desktop\AI-Content-Verifier\Ai-Content-Verifier\data\processed'

# Ensure the processed directory exists
os.makedirs(processed_data_dir, exist_ok=True)

# Function to process a CSV file
def process_csv(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # If the dataset contains AI and Student labels (like "AI" and "Student")
    if 'AI' in df.columns or 'Student' in df.columns:
        df['label'] = df['label'].replace({'AI': 0, 'Student': 1})

    # If the dataset uses numerical labels (like 0 for AI and 1 for Human)
    elif '0' in df.columns or '1' in df.columns:
        df['label'] = df['label'].replace({0: 0, 1: 1})  # No change needed but can be added for consistency

    # Optional: Perform additional cleaning (text cleaning, removing duplicates, etc.)
    # You can add functions here for text cleaning if needed
    
    # Save processed file to the processed directory
    output_file_name = os.path.join(processed_data_dir, os.path.basename(file_path))
    df.to_csv(output_file_name, index=False)
    print(f'Processed and saved: {output_file_name}')

# Process all CSV files in the raw data directory
for file_name in os.listdir(raw_data_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(raw_data_dir, file_name)
        process_csv(file_path)

print("Data processing complete. All files saved to the processed directory.")
