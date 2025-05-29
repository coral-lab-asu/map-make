import numpy as np
import pandas as pd
import os
import json
from utils.table_utils import *

def load_output_from_file(file_name):
    output_loaded = []
    with open(file_name, 'r') as text_file:
        output_loaded = text_file.readlines()  # readlines() reads all lines and stores them in a list
    return output_loaded


def generate_output_lines(output_path="", start_line=None, end_line=None):
    """
    Fetch output lines from the output file within the specified line range.

    Args:
        output_path (str): The path to the output file.
        start_line (int, optional): The starting line number (1-based index).
            Defaults to the first line if not provided.
        end_line (int, optional): The ending line number (inclusive, 1-based index).
            Defaults to the last line of the file if not provided.

    Returns:
        list: A list of output lines within the specified range.
    """
    selected_output_lines = []
    total_lines = 0  # To keep track of the total number of lines

    # Determine the total number of lines if end_line is not provided
    if start_line is None or end_line is None:
        with open(output_path, 'r') as output_file:
            for total_lines, line in enumerate(output_file, 1):
                pass  # Just counting the lines
        # Set default values if start_line or end_line is not provided
        if start_line is None:
            start_line = 0
        if end_line is None:
            end_line = total_lines
    else:
        # Validate line numbers
        if start_line < 0 or start_line > end_line:
            return []  # Invalid line numbers, return empty list

    # Read the file and collect the specified lines
    current_line = 0
    with open(output_path, 'r') as output_file:
        for line in output_file:
            current_line += 1
            if current_line >= start_line:
                if current_line > end_line:
                    break  # We've read all the required lines
                selected_output_lines.append(line.strip())

    return selected_output_lines



def read_file(file_path):
    """
    Reads a file and returns a dictionary if it's a JSON file,
    or a list of lines if it's a text file.
    :param file_path: Path to the file.
    :return: Dictionary (for JSON) or list of strings (for text).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    if file_path.endswith('.json'):
        # Read and return JSON data as a dictionary
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        # Read and return text data as a list of lines
        with open(file_path, 'r') as file:
            return file.readlines()
    
def write_file(file_path, data):
    """
    Writes data to a file. If it's a JSON file, writes data as JSON. 
    If it's a text file, writes data as lines.
    :param file_path: Path to the file.
    :param data: Dictionary for JSON, or list of strings for text files.
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.json':
        # Write data as JSON
        if not isinstance(data, (dict, list)):
            raise ValueError("For JSON files, data must be a dictionary or list.")
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    elif file_extension == '.txt':
        # Write data as text lines
        if not isinstance(data, list):
            raise ValueError("For text files, data must be a list of strings.")
        with open(file_path, 'w') as file:
            file.writelines(data)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


import pandas as pd

def custom_compare_tables(str1: str, str2: str):
    """
    Compare two tables represented as strings (str1: model output, str2: gold table)
    and return matching columns, model-only columns, gold-only columns, any empty columns,
    and the count of columns in the gold table that are not present in the model output.
    
    Args:
    - str1 (str): The model output table as a string.
    - str2 (str): The gold standard table as a string.
    
    Returns:
    - dict: A dictionary with matching columns, model-only columns, gold-only columns, 
            empty columns, and the count of missing columns in the model compared to the gold table.
    """
    # Check if either string is empty
    if not str1 or not str2:
        return f"One string is empty: {str1, str2}"
    
    # Convert the strings to dataframes
    d1 = convert_to_df(str1)
    d2 = convert_to_df(str2)
    
    matching_columns = []
    d1_only_columns = []
    d2_only_columns = []
    empty_columns_d1 = []
    empty_columns_d2 = []
    
    # Iterate over d1 columns and check if they exist in d2
    for col in d1.columns:
        if col in d2.columns:
            matching_columns.append(col)
        else:
            d1_only_columns.append(col)
        #Check if the column in d1 is empty or has all null values

        y = d1[col].isnull().sum()
        #print(d1[col].isnull().sum(),len(d1))
        try:
            # This will check if all entries in the column are null
            if (d1[col] == '').all():  # If all values in the column are null
                empty_columns_d1.append(col)
        except Exception as e:
            print("Error:", e)

    
    # Iterate over d2 columns and check if they exist in d1

    for col in d2.columns:
        if col not in d1.columns:
            d2_only_columns.append(col)

    # Calculate the number of columns in the gold table that are not present in the model output
    missing_columns_in_model = len(d2_only_columns)
    
    # Return results as a dictionary
    return {
        "MISSING COLUMN COUNT IN MODEL": missing_columns_in_model,
        "matching_columns": matching_columns,
        "model_only_columns": d1_only_columns,
        "gold_only_columns": d2_only_columns,
        "empty_columns_in_model": empty_columns_d1
        #"empty_columns_in_gold": empty_columns_d2
    }

def compare_values(str1: str, str2: str):

    if not str1 or not str2:
      return f"One string is empty: {str1, str2}"
    # Assuming convert_to_df converts the string into a dataframe
    df1 = convert_to_df(str1)  # Model output
    df2 = convert_to_df(str2)  # Gold table
    
    # Set the column with empty string as index
    df1.index = df1['']  # Index from the column named ''
    df2.index = df2['']  # Index from the column named ''
    
    # Drop the empty string column after setting it as the index
    df1 = df1.drop(columns=[''])
    df2 = df2.drop(columns=[''])

    # Initialize a list to store the mismatched tuples
    mismatches = []
    
    # Find the common columns between the two dataframes
    common_columns = df1.columns.intersection(df2.columns)
    
    # Find the common index values between the two dataframes
    common_indices = df1.index.intersection(df2.index)
    
    # Iterate over the common rows (indices) and common columns
    for row in common_indices:
        for col in common_columns:
            value1 = df1.at[row, col]  # Value from model output
            value2 = df2.at[row, col]  # Value from gold table
            
            # If the values differ, add to mismatches
            ## value 1: model output, value 2 gold table
            if value1 != value2:
                mismatches.append((row, col, value1, value2))
    
    return mismatches