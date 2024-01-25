import os
import pandas as pd
import shutil
from tqdm import tqdm
import re



def process_files_in_subfolder(subfolder_path):
    clean_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.csv')]

    for clean_file in clean_files:
        recording_id = clean_file.split('.')[0]
        info_list = recording_id.split('_')
        cleaned_data_filepath = os.path.join(subfolder_path, clean_file)
        df = pd.read_csv(cleaned_data_filepath)

        print(f"Adding columns to {clean_file}")
        # Add additional columns for metadata
        df['subject_id'] = info_list[0]
        df['sensor_type'] = info_list[1]
        df['activity_type'] = info_list[2]
        df['activity_subtype'] = info_list[3]
        df['recording_id'] = recording_id
        df.to_csv(cleaned_data_filepath, index=False)

def process_all_subfolders(data_folder):
    subfolders = [os.path.join(data_folder, d) for d in os.listdir(data_folder)
                  if os.path.isdir(os.path.join(data_folder, d))]
    for subfolder in subfolders:
        print(f"Processing {subfolder}...")
        process_files_in_subfolder(subfolder)


def sort_files(src_folder):
    pattern = re.compile(r's\d{7}', re.IGNORECASE)
    for file in os.listdir(src_folder):
        if os.path.isfile(os.path.join(src_folder, file)):
            match = pattern.search(file)
            if match:
                substring = match.group()
                new_folder = os.path.join(src_folder, substring)
                if not os.path.exists(new_folder):
                    os.mkdir(new_folder)
                shutil.move(os.path.join(src_folder, file), os.path.join(new_folder, file))


def merge_all_into_dataframe(clean_data_folder):
    dataframe = pd.DataFrame()
    # Getting all files recursively from all subfolders
    all_files = []
    for subdir, _, files in os.walk(clean_data_folder):
        for file in files:
            all_files.append(os.path.join(subdir, file))

    for file_path in tqdm(all_files, desc="Merging all cleaned data into one DataFrame", unit="file"):
        try:
            # Load data into a DataFrame
            new_df = pd.read_csv(file_path)

            # Merge into the base DataFrame
            dataframe = pd.concat([dataframe, new_df], ignore_index=True)
        except Exception as e:
            print(f"Error with file {file_path}: {str(e)}")

    return dataframe

if __name__ == '__main__':
    process_all_subfolders('./anonymized_dataset_2023/Respeck/')
    process_all_subfolders('./anonymized_dataset_2023/Thingy/')
    respeck_df = merge_all_into_dataframe('./anonymized_dataset_2023/Respeck/')
    thingy_df = merge_all_into_dataframe('./anonymized_dataset_2023/Respeck/')

    # Save the DataFrame as a CSV file
    respeck_df.to_csv('respeck_dataset.csv', index=False)
    thingy_df.to_csv('thingy_dataset.csv', index=False)
