"""Some helper functions for project 1."""

import csv
import numpy as np
import os

def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the corresponding numpy arrays,
    as well as the column headers (features).

    Args:
        data_path (str): Path to the data folder
        sub_sample (bool, optional): If True, the data will be subsampled. Defaults to False.

    Returns:
        headers (list): List of column names for the features
        x_train (np.array): Training data
        x_test (np.array): Test data
        y_train (np.array): Labels for training data (-1, 1)
        train_ids (np.array): IDs for training data
        test_ids (np.array): IDs for test data
    """
    # Load y_train
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )

    # Load x_train and x_test with headers
    with open(os.path.join(data_path, "x_train.csv"), 'r') as f:
        headers = f.readline().strip().split(',')

    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    # Separate the IDs (column 0)
    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)

    # Remove the ID column from x_train and x_test
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # Sub-sampling
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    # Remove the ID from the headers
    headers = headers[1:]

    return headers, x_train, x_test, y_train, train_ids, test_ids

def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
