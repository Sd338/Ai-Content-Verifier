import os
import pandas as pd
from typing import List

def load_all_data(data_directory: str) -> List[pd.DataFrame]:
    """Load all CSV data files from a specified directory.

    Args:
        data_directory (str): Path to the directory containing CSV files.

    Returns:
        List[pd.DataFrame]: List of DataFrames loaded from the CSV files.
    """
    dataframes = []
    for file_name in os.listdir(data_directory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_directory, file_name)
            df = pd.read_csv(file_path)
            dataframes.append(df)
            print(f"Loaded data from {file_name} with {df.shape[0]} rows and {df.shape[1]} columns.")
    return dataframes

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic preprocessing on the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Example preprocessing steps
    # Drop rows with missing values (you can adjust this based on your needs)
    df = df.dropna()

    # Convert any necessary columns to the appropriate data type
    # Example: df['some_column'] = df['some_column'].astype(str)

    # Return the preprocessed DataFrame
    return df
