import numpy as np
import pandas as pd
import re
SEP = "|"


def separate_tables(output_text): ## Function to separate player table and team table in the rotowire dataset.
    """
    Separates the input text into multiple tables based on headers.
    Returns a dictionary where keys are table names and values are table contents.
    """
    import re

    # Replace '<NEWLINE>' with '\n' to standardize line breaks
    output_text = re.sub(r'\s*<NEWLINE>\s*', '\n', output_text).strip()

    # Initialize variables
    tables = {}
    current_table_name = None
    current_table_content = []

    # Split the text into lines
    lines = output_text.splitlines()

    # Process each line
    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        # Check if the line is a table header (e.g., 'Team:' or 'Player:')
        header_match = re.match(r'^(\w+):\s*$', line)
        if header_match:
            # If we were collecting a previous table, save it
            if current_table_name:
                tables[current_table_name] = '\n'.join(current_table_content)
                current_table_content = []

            # Start a new table
            current_table_name = header_match.group(1)
        else:
            # Collect the line into the current table content
            if current_table_name:
                current_table_content.append(line)

    # After the loop, save the last table
    if current_table_name and current_table_content:
        tables[current_table_name] = '\n'.join(current_table_content)

    return tables

def extract_tables_from_responses(ai_responses):
    """
    Extracts named tables from AI responses based on '<ENDTABLE>' or '<TABLE END>' markers.

    Parameters:
    - ai_responses: List of dictionaries, each containing at least a 'response' key.

    Returns:
    - A list of dictionaries, each with 'data_point' and extracted tables.
    """
    extracted_tables = []
    cnt = 0
    
    for response_dict in ai_responses:
        data_point = cnt
        cnt += 1
        response_text = response_dict['response']

        # Replace all variations of '<NEWLINE>' with actual newline characters
        response = re.sub(r'\s*<NEWLINE>\s*', '\n', response_text).strip()
        
        # Uncomment the following line to debug the processed response
        # print(f"Data Point {data_point} Response:\n{response}\n{'-'*50}")
        
        # Dictionary to hold all extracted tables for this response
        tables = {}
        
        # Updated pattern to match table names with spaces and handle both end markers
        table_pattern = r'([\w\s]+):\s*\n((?:\|.*\n?)+?)(?:<ENDTABLE>|<TABLE END>)'

        # Find all tables in the response
        matches = list(re.finditer(table_pattern, response, re.MULTILINE))
        
        if not matches:
            print(f"No matches found for data_point {data_point}. Check response format.")
        
        for match in matches:
            table_name = match.group(1).strip()
            table_content = match.group(2).strip()
            tables[table_name] = table_content
            print(f"Table '{table_name}' found for data_point {data_point}")

        # Append extracted tables along with data_point to results
        extracted_tables.append({
            'data_point': data_point,
            **tables
        })

    return extracted_tables


def parse_text_to_table(text, strict=False):  ### UPDATED BY NA.
    """
    Parse the table text and return a numpy array of the table's content.
    """
    import numpy as np
    import re
    SEP = '|'  # Define the separator

    # Replace all instances of "<NEWLINE>" (with or without surrounding spaces) with actual newline characters
    text = re.sub(r'\s*<NEWLINE>\s*', '\n', text).strip()
    lines = text.splitlines()
    
    data = []
    
    # Remove any empty or whitespace-only lines at the beginning
    while lines and not lines[0].strip():
        lines.pop(0)

    # Check if the first line is a title (does not contain the separator)
    if lines and SEP not in lines[0]:
        title_line = lines.pop(0).strip()
        # Modify the title to create a header element
        title = title_line.rstrip(':').strip()
    else:
        title = ''
    
    # Process each line of the table
    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        
        # Ensure each line starts and ends with the separator
        if not line.startswith(SEP):
            line = SEP + line
        if not line.endswith(SEP):
            line = line + SEP
        
        # Split the line into cells and strip extra spaces
        cells = [x.strip() for x in line[1:-1].split(SEP)]
        data.append(cells)
    
    # Remove empty rows
    data = [row for row in data if any(cell.strip() for cell in row)]
    
    if not strict and data:
        # Determine the maximum number of columns
        n_col = max(len(row) for row in data)
        # Pad rows with missing columns
        data = [row + [""] * (n_col - len(row)) for row in data]
    
    # Insert the title as the first element of the header row
    if title and data:
        if data[0][0] == '':
            data[0][0] = title + ' Name'
        else:
            data[0] = [title + ' Name'] + data[0][1:]
    
    # Convert to a numpy array
    try:
        data = np.array(data, dtype=str)
    except Exception as e:
        if strict:
            raise ValueError("Failed to convert the data to a numpy array.") from e
        data = None
    
    return data



import numpy as np

def convert_markdown_table_to_numpy_array(table_string): 
    """
    Converts a markdown-formatted table string into a NumPy array.
    
    If the string is empty or only contains whitespace, returns an empty array.
    
    Args:
        table_string (str): The markdown table as a string.
    
    Returns:
        numpy.ndarray: A 2D NumPy array representing the table, or an empty array if input is empty.
    """
    # Check if the input string is empty or contains only whitespace
    if not table_string.strip():
        return np.array([])  # Return an empty NumPy array
    
    # Split the string into lines
    lines = table_string.strip().split('\n')
    
    # Initialize a list to hold the rows of the table
    data = []

    for line in lines:
        # Remove leading and trailing whitespace and '|'
        line = line.strip().strip('|')

        # Split the line into cells based on '|'
        cells = line.split('|')

        # Strip whitespace from each cell
        cells = [cell.strip() for cell in cells]

        # Ignore separator lines (e.g., '---|---|---')
        if not any(cells) or set(cells) == {'---'}:
            continue

        # Append the row to the data list
        data.append(cells)

    # Find the maximum number of columns
    max_cols = max(len(row) for row in data)

    # Pad rows with empty strings to ensure consistent column count
    for row in data:
        row.extend([''] * (max_cols - len(row)))

    # Convert the list of lists into a NumPy array
    array = np.array(data, dtype=object)

    return array


def convert_to_df(satring):
    # Assuming 'array' is your NumPy array    
    # The first row is assumed to be the header
    array = convert_markdown_table_to_numpy_array(satring)
    if(len(array) == 0):
        return None
    headers = array[0]
    data = array[1:]
    df = pd.DataFrame(data, columns=headers)
    return df


def extract_tables(data_list): ## For wikibio
    """
    Extracts table lines from the 'response' field in a list of dictionaries.
    The table lines start with "Final Answer:" followed by the table.

    Args:
        data_list (list): List of dictionaries containing 'response' keys.

    Returns:
        list: A list of extracted table strings.
    """
    tables = []
    pattern = r"\**Final Answer:\**\s*((?:\|.*(?:<NEWLINE>\s*\|.*)*)*)"# Regex to match "Final Answer:" followed by the table line

    for entry in data_list:
        response =entry['response']
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            table_line = match.group(1).strip()
            tables.append(table_line)
        else:
            print(f"No table found in data_point {entry.get('data_point')}.")

    return tables

