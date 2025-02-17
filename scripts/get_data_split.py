import os
import pandas as pd

def create_test_splits():
    os.makedirs("./data/v2_splits/test", exist_ok=True)
    
    # Loop over all files in the validation splits directory
    for filename in os.listdir("./data/v2_splits/val/"):
        print('\nProcessing ' + filename)
        
        # Path to the current validation split file
        val_file_path = os.path.join("./data/v2_splits/val/", filename)
        
        # Read the validation split data
        df_val = pd.read_json(val_file_path, lines=True, encoding='utf-8')
        val_ids = set(df_val['id'])  # Create a set of 'id' values present in the validation split

        # Path to the corresponding original dataset file
        original_file_path = os.path.join("./data/val_setv2/", filename)
        
        # Read the original dataset data
        df_original = pd.read_json(original_file_path, lines=True, encoding='utf-8')

        # Filter to create the test split by excluding rows whose 'id' is in the validation split
        df_test = df_original[~df_original['id'].isin(val_ids)]

        # Save the test split data
        test_file_path = os.path.join("./data/v2_splits/test/", filename)
        df_test.to_json(test_file_path, orient='records', lines=True, force_ascii=False)

        print(f"Generated test split for {filename}: {len(df_test)} rows.")
        
if __name__ == "__main__":
    create_test_splits()
