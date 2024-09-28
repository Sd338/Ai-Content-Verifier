import os
from src.utils.helpers import load_all_data, preprocess_data

def test_load_all_data():
    # Path to the processed data directory
    data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')

    # Load all CSV data
    all_dataframes = load_all_data(data_directory)

    # Check if dataframes are loaded
    assert len(all_dataframes) > 0, "No dataframes were loaded."

    # Print the first few rows of each DataFrame to verify loading
    for i, df in enumerate(all_dataframes):
        print(f"DataFrame {i+1}:")
        print(df.head())

def test_preprocess_data():
    # Use a sample DataFrame to test the preprocessing function
    data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')
    all_dataframes = load_all_data(data_directory)
    
    # Assuming there's at least one dataframe loaded
    if all_dataframes:
        df = all_dataframes[0]

        # Apply preprocessing
        preprocessed_df = preprocess_data(df)

        # Print the first few rows of the preprocessed DataFrame
        print("Preprocessed DataFrame:")
        print(preprocessed_df.head())
    else:
        print("No data to preprocess.")

if __name__ == "__main__":
    test_load_all_data()
    test_preprocess_data()
