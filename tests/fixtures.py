"""Pytest fixtures."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytest import fixture

from twitter_analyzer import DATA_DIR, TESTS_DIR
from twitter_analyzer.data_preprocessing import LABEL_DICT, get_dataset_splits


def get_data():
    """Load the training data."""
    data_file = Path(DATA_DIR, "Corona_NLP_train.csv")
    data = pd.read_csv(data_file, encoding="ISO-8859-1")

    data["Sentiment"] = data["Sentiment"].replace(LABEL_DICT)

    return data


@fixture
def get_test_data():
    """Retrieve a subset of training data."""
    data = get_data()
    new_data = []

    for i in range(5):
        subsample = data[data["Sentiment"] == i].iloc[:10, :]
        new_data.append(subsample)

    new_df = pd.concat(new_data, axis=0)

    return new_df


@fixture
def get_data_subsample():
    """Get a splitted subsample."""
    data = get_data()
    train, val, test = get_dataset_splits(data)

    return train[0], val[0], test[0]


@fixture
def get_subsample_tokenized():
    """Get a tokenized subsample."""
    X_train = {}
    X_train["input_ids"] = torch.randint(1, 30, (32, 10))
    X_train["attention_mask"] = torch.randint(1, 30, (32, 10))
    y_train = np.random.randint(0, 5, 32)

    X_val = {}
    X_val["input_ids"] = torch.randint(1, 30, (8, 10))
    X_val["attention_mask"] = torch.randint(1, 30, (8, 10))
    y_val = np.random.randint(0, 5, 8)

    X_test = {}
    X_test["input_ids"] = torch.randint(1, 30, (12, 10))
    X_test["attention_mask"] = torch.randint(1, 30, (12, 10))
    y_test = np.random.randint(0, 5, 12)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


@fixture
def load_ready_training_data():
    """Load a subsample of data ready for training."""
    with open(Path(TESTS_DIR, "data/testing_data.pkl"), "rb") as file:
        data = pickle.load(file)

    return data["train"], data["val"]
