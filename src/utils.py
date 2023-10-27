# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Authors: Francisco Melícias e Tiago F. R. Ribeiro
# Creation date (file creation): 24/10/2023
# Description: This file contains utility functions used in the project.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
import numpy as np


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
        'SMOTE': {'train': 'train_smote.csv', 'test': 'encoded_testData.csv'},
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

    print(f"Loading complete.\nTraining data: {df_train.shape[0]} rows, {df_train.shape[1]} columns. \nTest data: {df_test.shape[0]} rows, {df_test.shape[1]} columns.")

    return df_train, df_test



def one_hot_encode_categorical(X_train, X_test, random_state=None):
    """
    One-hot encode categorical features in X_train and X_test.

    Parameters:
    - X_train: pd.DataFrame, training dataset.
    - X_test: pd.DataFrame, test dataset.
    - random_state: int or None, random state for train_test_split.

    Returns:
    - X_train_enc: pd.DataFrame, one-hot encoded training dataset.
    - X_test_enc: pd.DataFrame, one-hot encoded test dataset.
    """

    # Extract categorical features
    cat_features_train = X_train.select_dtypes(include="object").columns
    cat_features_test = X_test.select_dtypes(include="object").columns

    # Check if there are categorical features
    if cat_features_train.empty and cat_features_test.empty:
        print("No categorical features found. Returning original datasets.")
        return X_train, X_test

    # Concatenate X_train and X_test
    X_comb = pd.concat([X_train[cat_features_train], X_test[cat_features_test]], axis=0)

    # Apply one-hot encoding (get_dummies)
    X_comb_enc = pd.get_dummies(X_comb, dtype='int8')

    # Split back into X_train and X_test
    rows_train = len(X_train)
    X_train_enc = X_comb_enc.iloc[:rows_train, :]
    X_test_enc = X_comb_enc.iloc[rows_train:, :]
    
    print("Encoding complete.")
    print(f"No of features before encoding: {X_train.shape[1]}" + "\n" + f"No of features after encoding: {X_train_enc.shape[1]}")

    return X_train_enc, X_test_enc

def encode_labels(y_train, y_test):
    """
    Encode labels using LabelEncoder, print the correspondence between original and encoded labels,
    and return the label encoder for potential inverse transformations.

    Parameters:
    - y_train: pd.Series or array-like, training labels.
    - y_test: pd.Series or array-like, test labels.

    Returns:
    - y_train_enc: pd.Series, encoded training labels.
    - y_test_enc: pd.Series, encoded test labels.
    - le: LabelEncoder, label encoder instance.
    """

    # Instantiate the label encoder
    le = LabelEncoder()

    # Fit and encode the training labels
    y_train_enc = le.fit_transform(y_train)

    # Encode the test labels
    y_test_enc = le.transform(y_test)

    # Print the correspondence between original and encoded labels
    print('Attack_type and encoded labels:\n')
    for i, label in enumerate(le.classes_):
        print(f'{label:23s} {i:d}')

    return y_train_enc, y_test_enc, le




def scale_data(X_train, X_test, scaler_type='standard'):
    """
    Scale the input data using the specified scaler.

    Parameters:
    X_train (np.array): The training data to be scaled.
    X_test (np.array): The test data to be scaled.
    scaler_type (str): The type of scaler to use. Options are 'standard', 'minmax', and 'robust'. Default is 'standard'.

    Returns:
    X_train_scaled (np.array): The scaled training data.
    X_test_scaled (np.array): The scaled test data.

    Raises:
    ValueError: If the scaler_type is not 'standard', 'minmax', or 'robust'.
    Exception: If there was an error during scaling.
    """
    
    if not isinstance(X_train, np.ndarray) or not isinstance(X_test, np.ndarray):
        raise ValueError("Input data should be numpy array")
    if scaler_type not in ['standard', 'minmax', 'robust']:
        raise ValueError(f'Unknown scaler: {scaler_type}')

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()

    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info(f'Scaling successful with {scaler_type} scaler.')
    except Exception as e:
        logging.error(f'Error during scaling: {str(e)}')
        raise
    
    pretty_print_stats(X_train_scaled, X_test_scaled)
    
    return X_train_scaled, X_test_scaled


def pretty_print_stats(X_train, X_test):
    """
    Pretty print the mean and standard deviation of the input data.

    Parameters:
    X_train (np.array): The training data.
    X_test (np.array): The test data.
    """

    # Calculate mean and standard deviation
    train_mean, train_std = X_train.mean(), X_train.std()
    test_mean, test_std = X_test.mean(), X_test.std()

    # Create a dictionary of the stats
    stats = {
        'Train Data': {
            'Mean': round(train_mean, 2),
            'Standard Deviation': round(train_std, 2)
        },
        'Test Data': {
            'Mean': round(test_mean, 2),
            'Standard Deviation': round(test_std, 2)
        }
    }

    # Pretty print the stats
    from pprint import pprint
    pprint(stats, underscore_numbers=True)




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

