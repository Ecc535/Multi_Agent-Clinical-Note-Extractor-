import pandas as pd

# Load the dataset and ground truth
data = pd.read_csv('/Users/yixinshen/Agent/mimic_clip_dataset.csv')

# Load the truth JSON and transpose
truth_df = pd.read_json('/Users/yixinshen/Agent/mimic_clip_truth_enriched.json').T

# Ensure index is string, since JSON keys are strings like "45901"
truth_df.index = truth_df.index.astype(str)

record_index = 0
single_record_df = data.iloc[[record_index]]

# SUBJECT_ID from CSV (convert to string to match truth JSON)
note_id_key = str(single_record_df['SUBJECT_ID'].iloc[0])
print("Looking for key:", note_id_key)

# Now this will work
corresponding_truth_df = truth_df.loc[[note_id_key]]

# Save outputs
single_record_df.to_csv('/Users/yixinshen/Agent/single_record.csv', index=False)
corresponding_truth_df.to_json('/Users/yixinshen/Agent/single_truth.json', orient='index', indent=4, force_ascii=False)

print(f"Successfully split record at index {record_index}.")
print("Saved data to: /Users/yixinshen/Agent/single_record.csv")
print("Saved truth to: /Users/yixinshen/Agent/single_truth.json")