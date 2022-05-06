"""
Data preprocessing utility functions.

This module contains several function used to preprocess the
data related to the project.
"""
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from transformers.tokenization_utils import PreTrainedTokenizer

MAX_TWEETS_LENGTH = 280


def load_raw_dataset(train_file: str, test_file: str) -> pd.DataFrame:
    """
    Load the raw dataset.

    The dataset is stored in kaggle and is divided into
    train and test.

    Args:
        train_file (str): Training set filepath.
        test_file (str): Test set filepath.

    Returns:
        pd.DataFrame: Total raw dataset.
    """
    train = pd.read_csv(train_file, encoding="ISO-8859-1")
    test = pd.read_csv(test_file, encoding="ISO-8859-1")

    df = pd.concat([train, test], axis=0)

    return df


def sentiment_to_integer(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the sentiment categories into integers.

    Args:
        dataset (pd.DataFrame): Original dataset.

    Returns:
        pd.DataFrame: Transformed dataset with the tweets'
            sentiment represented as integers.
    """
    label_dict = {
        "Extremely Negative": 0,
        "Negative": 1,
        "Neutral": 2,
        "Positive": 3,
        "Extremely Positive": 4,
    }

    dataset["Sentiment"] = dataset["Sentiment"].replace(label_dict)

    return dataset


def get_dataset_splits(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Split the dataset into three sets: training, validation and test.

    Args:
        dataset (pd.DataFrame): Original dataset.
        test_size (float): Percentage of data for the test set.
        validation_size (float): Percentage of data for the validation size.
        random_state (int): Control the shuffling applied to the data before
            the split

    Returns:
        Tuple: A tuple containing:
            Tuple[np.ndarray, np.ndarray]: Training tweets and labels.
            Tuple[np.ndarray, np.ndarray]: Validation tweets and labels.
            Tuple[np.ndarray, np.ndarray]: Testing tweets and labels.
    """
    # Get train and test split
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["OriginalTweet"].values,
        dataset["Sentiment"].values,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset["Sentiment"],
    )

    # Get the validation split from training split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train,
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def tokenize_splits(
    tokenizer: PreTrainedTokenizer,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    X_test: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenize the training, validation and test data.

    Args:
        tokenizer (transformers.tokenization_utils.PreTrainedTokenizer):
            Model tokenizer.
        X_train (torch.Tensor): Training data.
        X_val (torch.Tensor): Validation data.
        X_test (torch.Tensor): Test data.

    Returns:
        Tuple: A tuple containing:
            torch.Tensor: Encoded training data.
            torch.Tensor: Encoded validation data.
            torch.Tensor: Encoded test data.
    """
    X_train = tokenizer.batch_encode_plus(
        X_train,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=MAX_TWEETS_LENGTH,
        return_tensors="pt",
    )

    X_val = tokenizer.batch_encode_plus(
        X_val,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=MAX_TWEETS_LENGTH,
        return_tensors="pt",
    )

    X_test = tokenizer.batch_encode_plus(
        X_test,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=MAX_TWEETS_LENGTH,
        return_tensors="pt",
    )

    return X_train, X_val, X_test


def convert_to_datasets(
    training: Tuple[torch.Tensor, np.ndarray],
    validation: Tuple[torch.Tensor, np.ndarray],
    testing: Tuple[torch.Tensor, np.ndarray],
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Convert the splits into datasets.

    Args:
        training (Tuple[torch.Tensor, np.ndarray]): Tuple
            containing the training embeddings on first
            position and labels on the second position.
        validation (Tuple[torch.Tensor, np.ndarray]): Tuple
            containing the validation embeddings on first
            position and labels on the second position.
        testing (Tuple[torch.Tensor, np.ndarray]): Tuple
            containing the testing embeddings on first
            position and labels on the second position.

    Returns:
        Tuple: A tuple containing:
            TensorDataset: Training dataset.
            TensorDataset: Validation dataset.
            TensorDataset: Test dataset.
    """
    dataset_train = TensorDataset(
        training[0]["inputs_ids"],
        training[0]["attention_mask"],
        torch.tensor(training[1]),
    )

    dataset_val = TensorDataset(
        validation[0]["inputs_ids"],
        validation[0]["attention_mask"],
        torch.tensor(validation[1]),
    )

    dataset_test = TensorDataset(
        testing[0]["inputs_ids"], testing[0]["attention_mask"], torch.tensor(testing[1])
    )

    return dataset_train, dataset_val, dataset_test
