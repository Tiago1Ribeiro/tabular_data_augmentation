# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Authors: Francisco Melícias e Tiago F. R. Ribeiro
# Creation date (file creation): 24/10/2023
# Description: This file contains utility functions used in the project.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import logging

import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
)


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
    df.to_csv(csv_file_path, mode="a", header=not file_exists, index=False)


def load_dataset(data_directory, augmentation="None", ignore_columns=None):
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
        "None": {"train": "EdgeIIot_train_100k.csv", "test": "EdgeIIot_test.csv"},
        "SMOTE": {
            "train": "EdgeIIot_train_smote_v2.csv",
            "test": "EdgeIIot_test_enc_v2.csv",
        },
        "SMOTE-NC": {"train": "train_smotenc_v2.csv", "test": "EdgeIIot_test.csv"},
        "RealTabFormer": {
            "train": "EdgeIIot_train_100k_RealTabFormer.csv",
            "test": "EdgeIIot_test.csv",
        },
        "GReaT": {
            "train": "EdgeIIot_train_100k_GReaT.csv",
            "test": "EdgeIIot_test.csv",
        },
    }

    # Validate augmentation option
    if augmentation not in file_paths:
        raise ValueError(
            "AUGMENTATION option not recognized.\n \t     Please choose between 'None', 'SMOTE', 'SMOTE-NC', 'RealTabFormer', or 'GReaT'."
        )

    # Load training data
    df_train_path = os.path.join(data_directory, file_paths[augmentation]["train"])
    df_train = pd.read_csv(df_train_path, low_memory=False)

    # Load test data
    df_test_path = os.path.join(data_directory, file_paths[augmentation]["test"])
    df_test = pd.read_csv(df_test_path, low_memory=False)

    # Ignore specified columns during validation
    if ignore_columns:
        df_train = df_train.drop(columns=ignore_columns, errors="ignore")
        df_test = df_test.drop(columns=ignore_columns, errors="ignore")

    # Validate if test data has the same columns as training data
    if set(df_train.columns) != set(df_test.columns):
        different_columns = set(df_train.columns) ^ set(df_test.columns)
        print(
            f"Warning: Test data has different columns than training data.\nColumns: {different_columns}"
        )

    print(
        f"Loading complete.\nTrain data: {df_train.shape[0]} rows, {df_train.shape[1]} columns. \nTest data: {df_test.shape[0]} rows, {df_test.shape[1]} columns."
    )

    return df_train, df_test


def encode_categorical(X_train, X_test, encoding="onehot"):
    """
    Encode categorical features in X_train and X_test.

    Parameters:
    - X_train: pd.DataFrame, training dataset.
    - X_test: pd.DataFrame, test dataset.
    - encoding: str, optional, type of encoder to use. Options are 'onehot' and 'label'. Default is 'onehot'.

    Returns:
    - X_train_enc: pd.DataFrame, encoded training dataset.
    - X_test_enc: pd.DataFrame, encoded test dataset.
    - additional_info: dict, additional information depending on the encoding type.
    """

    # Extract categorical features
    cat_features_train = X_train.select_dtypes(include=["object", "category"]).columns
    cat_features_test = X_test.select_dtypes(include=["object", "category"]).columns

    X_comb = pd.concat([X_train, X_test], axis=0)

    # Check if there are categorical features
    if cat_features_train.empty and cat_features_test.empty:
        print("No categorical features found. Returning original datasets.")
        return X_train, X_test, None

    cat_features = list(set(cat_features_train) | set(cat_features_test))

    print(f"Categorical features to be encoded:\n")
    print("\n".join(cat_features))

    if encoding == "label":
        # LabelEncoder (Encode target labels with value between 0 and n_classes-1)
        categorical_columns = []
        categorical_dims = []

        # Encode categorical features
        for feature in cat_features:
            le = LabelEncoder()
            X_comb[feature] = le.fit_transform(X_comb[feature])
            categorical_columns.append(feature)
            # Number of unique values in the encoded this feature
            categorical_dims.append(len(le.classes_))

        # Split back into X_train and X_test
        rows_train = len(X_train)
        X_train_enc = X_comb.iloc[:rows_train, :]
        X_test_enc = X_comb.iloc[rows_train:, :]

        print("\nEncoding complete.")
        print(
            f"No of features before encoding: {X_train.shape[1]}"
            + "\n"
            + f"No of features after encoding: {X_train_enc.shape[1]}"
        )

        additional_info = {
            "categorical_columns": categorical_columns,
            "categorical_dims": categorical_dims,
        }

        return X_train_enc, X_test_enc, additional_info

    else:
        # Apply one-hot encoding (get_dummies) only to categorical features
        X_comb_enc = pd.get_dummies(
            X_comb, columns=cat_features_train, drop_first=True, dtype="int8"
        )

        # Drop original categorical columns
        X_comb_enc = X_comb_enc.drop(columns=cat_features, errors="ignore")

        # Split back into X_train and X_test
        rows_train = len(X_train)
        X_train_enc = X_comb_enc.iloc[:rows_train, :]
        X_test_enc = X_comb_enc.iloc[rows_train:, :]

        print("\nEncoding complete.")
        print(
            f"No of features before encoding: {X_train.shape[1]}"
            + "\n"
            + f"No of features after encoding: {X_train_enc.shape[1]}"
        )

        additional_info = {"encoded_columns": list(X_comb_enc.columns)}

        return X_train_enc, X_test_enc, additional_info


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
    print("Attack_type and encoded labels:\n")
    for i, label in enumerate(le.classes_):
        print(f"{label:23s} {i:d}")

    return y_train_enc, y_test_enc, le


def scale_data(X_train, X_test, scaler_type="standard"):
    """
    Scale the input data using the specified scaler.

    Parameters:
    - X_train (np.array): The training data to be scaled.
    - X_test (np.array): The test data to be scaled.
    - scaler_type (str): The type of scaler to use. Options are 'standard', 'minmax', and 'robust'. Default is 'standard'.

    Returns:
    - X_train_scaled (np.array): The scaled training data.
    - X_test_scaled (np.array): The scaled test data.

    Raises:
    ValueError: If the scaler_type is not 'standard', 'minmax', or 'robust'.
    Exception: If there was an error during scaling.
    """

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    if scaler_type not in ["standard", "minmax", "robust"]:
        raise ValueError(f"Unknown scaler: {scaler_type}")

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()

    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info(f"Scaling successful with {scaler_type} scaler.")
    except Exception as e:
        logging.error(f"Error during scaling: {str(e)}")
        raise

    pretty_print_stats(X_train_scaled, X_test_scaled)

    return X_train_scaled, X_test_scaled


def pretty_print_stats(X_train, X_test):
    """
    Pretty print the mean and standard deviation of the input data.

    Parameters:
    - X_train (np.array): The training data.
    - X_test (np.array): The test data.
    """

    # Calculate mean and standard deviation
    train_mean, train_std = X_train.mean(), X_train.std()
    test_mean, test_std = X_test.mean(), X_test.std()

    # print stats in a nice table wirh regular spacing, left aligned with separator
    print(f"{'':<10}{'mean':<10}{'std':<10}")
    print(f"{'Train:':<10}{train_mean:<10.3f}{train_std:<10.3f}")
    print(f"{'Test:':<10}{test_mean:<10.3f}{test_std:<10.3f}")


def format_value(value):
    return "{:.2f}%".format(value * 100)


def print_results_table(results):
    """
    Print the results with tabulate.

    Parameters:
    - results (dict): The results dictionary.
    """

    formatted_results = {
        key: format_value(value)
        for key, value in results.items()
        if key not in ["model", "augmentations", "timestamp"]
    }

    print(
        tabulate(
            [
                ["Accuracy", formatted_results["accuracy"]],
                ["Precision (macro)", formatted_results["precision_macro"]],
                ["Recall (macro)", formatted_results["recall_macro"]],
                ["F1 (macro)", formatted_results["f1_macro"]],
                ["Precision (weighted)", formatted_results["precision_weighted"]],
                ["Recall (weighted)", formatted_results["recall_weighted"]],
                ["F1 (weighted)", formatted_results["f1_weighted"]],
            ],
            headers=["Metric", "Value"],
            tablefmt="fancy_grid",
        )
    )
