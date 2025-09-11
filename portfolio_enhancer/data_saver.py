# data_saver.py
# This module is responsible for saving data to a file.

import pandas as pd

def save_data_to_csv(data_df: pd.DataFrame, filename: str):
    """
    Saves a pandas DataFrame to a CSV file.

    Args:
        data_df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the file to create.
    """
    try:
        data_df.to_csv(filename)
        print(f"\nHistorical data successfully saved to '{filename}'")
    except Exception as e:
        print(f"An error occurred while saving data: {e}")