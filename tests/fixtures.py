"""Pytest fixtures."""
from pathlib import Path

import pandas as pd
from pytest import fixture

from twitter_analyzer import DATA_DIR
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
