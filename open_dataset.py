import pandas as pd


file_path = '/Users/yixinshen/Agent/multi_admissions_notes.json'
output_csv_path = '/Users/yixinshen/Agent/first_record.csv'

try:
    # Read the JSON file into a pandas DataFrame.
    df = pd.read_json(file_path)
    
    # Get the first record using head(1)
    first_record = df.head(1)
    
    # Save the first record to a CSV file without the DataFrame index
    first_record.to_csv(output_csv_path, index=False)
    
    print(f"Successfully saved the first record to '{output_csv_path}'")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")
