import pandas as pd

file_path = '/Users/yixinshen/Agent/mimic_clip_dataset.csv'

try:
    # Read the dataset from the CSV file
    df = pd.read_csv(file_path)

    # 1. Rewrite SUBJECT_ID to be the same as the original ROW_ID
    df['SUBJECT_ID'] = df['ROW_ID']

    # 2. Re-index the ROW_ID to be a new sequential index (0, 1, 2, ...)
    df['ROW_ID'] = range(len(df))

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

    print(f"Successfully updated IDs in {file_path}")

except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
