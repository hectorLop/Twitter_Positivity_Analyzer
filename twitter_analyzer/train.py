"""
Training script.

This module allows to train a BERT model to
classify tweets according their positivity
"""
from typing import Tuple

from torch.utils.data import TensorDataset
from transformers.tokenization_utils import PreTrainedTokenizer
from twitter_analizer.data_preprocessing import (
    convert_to_datasets,
    get_dataset_splits,
    load_raw_dataset,
    sentiment_to_integer,
    tokenize_splits,
)


def get_data(
    train_file: str, test_file: str, tokenizer: PreTrainedTokenizer
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Get the datasets ready for training.

    Args:
        train_file (str): Train dataset filepath.
        test_file (str): Test dataset filepath.
        tokenizer (PreTrainedTokenizer): Tokenizer to
            crete embeddings.

    Returns:
        Tuple: A Tuple containing:
            TensorDataset: Train dataset.
            TensorDataset: Validation dataset.
            TensorDataset: Test dataset.
    """
    raw_dataset = load_raw_dataset(train_file, test_file)
    raw_dataset = sentiment_to_integer(raw_dataset)

    X_train, y_train, X_val, y_val, X_test, y_test = get_dataset_splits(raw_dataset)

    X_train, X_val, X_test = tokenize_splits(tokenizer, X_train, X_val, X_test)

    train, val, test = convert_to_datasets(
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    )

    return train, val, test
