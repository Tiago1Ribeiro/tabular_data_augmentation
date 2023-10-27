# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Authors: Francisco Melícias e Tiago F. R. Ribeiro
# Creation date (file creation): 24/10/2023
# Description: This file contains utility functions used in the project.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import os

def save_results_to_csv(results, csv_file_path):
    """
    Saves the results to a CSV file.

    Parameters:
        results : dict
            Dictionary containing the results.
        csv_file_path : str
            Path to the CSV file.
    """

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Check if the file exists
    file_exists = False
    try:
        pd.read_csv(csv_file_path)
        file_exists = True
    except FileNotFoundError:
        pass

    # Append results to the CSV file
    df.to_csv(csv_file_path, mode='a', header=not file_exists, index=False)


def load_dataset(data_directory, augmentation='None', ignore_columns=None):
    """
    Load and validate training and test data based on the augmentation option.

    Parameters:
    - data_directory: str, path to the directory containing data files.
    - augmentation: str, augmentation option ('None', 'SMOTE', 'SMOTE-NC', 'RealTabFormer', 'GReaT').
    - ignore_columns: list, optional, list of column names to ignore during validation.

    Returns:
    - df_train: pd.DataFrame, training dataset.
    - df_test: pd.DataFrame, test dataset.
    """

    # Define file paths based on augmentation option
    file_paths = {
        'None': {'train': 'EdgeIIot_train_100k.csv', 'test': 'EdgeIIot_test.csv'},
        'SMOTE': {'train': 'EdgeIIot_train_100k_SMOTE.csv', 'test': 'EdgeIIot_encoded_test.csv'},
        'SMOTE-NC': {'train': 'EdgeIIot_train_100k_SMOTE_NC.csv', 'test': 'EdgeIIot_test.csv'},
        'RealTabFormer': {'train': 'EdgeIIot_train_100k_RealTabFormer.csv', 'test': 'EdgeIIot_test.csv'},
        'GReaT': {'train': 'EdgeIIot_train_100k_GReaT.csv', 'test': 'EdgeIIot_test.csv'},
    }

    # Validate augmentation option
    if augmentation not in file_paths:
        raise ValueError("AUGMENTATION option not recognized.\n \t     Please choose between 'None', 'SMOTE', 'SMOTE-NC', 'RealTabFormer', or 'GReaT'.")

    # Load training data
    df_train_path = os.path.join(data_directory, file_paths[augmentation]['train'])
    df_train = pd.read_csv(df_train_path, low_memory=False)

    # Load test data
    df_test_path = os.path.join(data_directory, file_paths[augmentation]['test'])
    df_test = pd.read_csv(df_test_path, low_memory=False)

    # Ignore specified columns during validation
    if ignore_columns:
        df_train = df_train.drop(columns=ignore_columns, errors='ignore')
        df_test = df_test.drop(columns=ignore_columns, errors='ignore')

    # Validate if test data has the same columns as training data
    if set(df_train.columns) != set(df_test.columns):
        different_columns = set(df_train.columns) ^ set(df_test.columns)
        print(f"Warning: Test data has different columns than training data.\nColumns: {different_columns}")

    print(f"Loading complete.\nTraining data: {df_train.shape[0]} lines, {df_train.shape[1]} columns. Test data: {df_test.shape[0]} lines, {df_test.shape[1]} columns.")

    return df_train, df_test


# def save_results_to_csv(results, file_path, columns=None, append=True):
#     """
#     Saves the results of experiments to a CSV file. Creates the file if it
#     doesn't exist. If columns are provided, uses them as column names;
#     otherwise, uses keys from the results dictionary.

#     Parameters:
#     - results (dict or list): A dictionary or list containing the results to be saved.
#     - file_path (str): The path of the CSV file.
#     - columns (list, optional): A list of column names. If None, uses keys from the results dictionary.
#     - append (bool, optional): If True, appends the results to the existing file. If False, overwrites the file.

#     Returns:
#     - None
#     """

#     # If results is a list, convert it to a dictionary with default column names
#     if isinstance(results, list):
#         results = {'Column{}'.format(
#             i + 1): results[i] for i in range(len(results))}

#     # If columns are not provided, use keys from the results dictionary
#     if columns is None:
#         columns = list(results.keys())

#     # Check if the file already exists
#     file_exists = os.path.exists(file_path)

#     # Append or create a new CSV file
#     if append and file_exists:
#         existing_data = pd.read_csv(file_path)
#         new_data = pd.DataFrame(results, columns=columns, index=[0])
#         combined_data = pd.concat([existing_data, new_data], ignore_index=True)
#         combined_data.to_csv(file_path, index=False)
#         print(f"Results appended to the existing file: {file_path}")
#     else:
#         new_data = pd.DataFrame(results, columns=columns, index=[0])
#         new_data.to_csv(file_path, index=False)
#         if file_exists and not append:
#             print(f"Existing file overwritten with new results: {file_path}")
#         elif not file_exists:
#             print(f"New CSV file created: {file_path}")


# def save_results_to_csv(results, file_path, columns=None, append=True):
#     """
#     Saves the results of experiments to a CSV file. Creates the file if it
#     doesn't exist. If columns are provided, uses them as column names;
#     otherwise, uses keys from the results dictionary.

#     Parameters:
#     - results (dict or list): A dictionary or list containing the results to be saved.
#     - file_path (str): The path of the CSV file.
#     - columns (list, optional): A list of column names. If None, uses keys from the results dictionary.
#     - append (bool, optional): If True, appends the results to the existing file. If False, overwrites the file.

#     Returns:
#     - None
#     """

#     # Create the directory if it doesn't exist
#     directory = os.path.dirname(file_path)
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     # If results is a list, convert it to a dictionary with default column names
#     if isinstance(results, list):
#         results = {'Column{}'.format(
#             i + 1): results[i] for i in range(len(results))}

#     # If columns are not provided, use keys from the results dictionary
#     if columns is None:
#         columns = list(results.keys())

#     # Check if the file already exists
#     file_exists = os.path.exists(file_path)

#     # Append or create a new CSV file
#     if append and file_exists:
#         existing_data = pd.read_csv(file_path)
#         new_data = pd.DataFrame(results, columns=columns, index=[0])
#         combined_data = pd.concat([existing_data, new_data], ignore_index=True)
#         combined_data.to_csv(file_path, index=False)
#         print(f"Results appended to the existing file: {file_path}")
#     else:
#         new_data = pd.DataFrame(results, columns=columns, index=[0])
#         new_data.to_csv(file_path, index=False)
#         if file_exists and not append:
#             print(f"Existing file overwritten with new results: {file_path}")
#         elif not file_exists:
#             print(f"New CSV file created: {file_path}")

