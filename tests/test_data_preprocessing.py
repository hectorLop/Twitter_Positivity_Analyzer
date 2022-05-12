"""
Data preprocesses unitary testing.

This module contains the unitary testing for
the different data preprocesses.
"""
from typing import Callable

from transformers import BertTokenizer

from tests.fixtures import get_data_subsample, get_test_data
from twitter_analyzer.data_preprocessing import get_dataset_splits, tokenize_splits


def test_get_dataset_splits(get_test_data: Callable):
    """Test the get_dataset_splits function."""
    data = get_test_data

    train, val, test = get_dataset_splits(data)

    y_train = train[1]
    y_val = val[1]
    y_test = test[1]

    assert len(y_test) == 10
    assert len(y_train) == 32
    assert len(y_val) == 8


def test_tokenize_splits(get_data_subsample: Callable):
    """Test the tokenize_splits function."""
    X_train, X_val, X_test = get_data_subsample

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    train, val, test = tokenize_splits(tokenizer, X_train, X_val, X_test)

    assert train
    assert list(train.keys()) == ["input_ids", "token_type_ids", "attention_mask"]
    assert val
    assert list(val.keys()) == ["input_ids", "token_type_ids", "attention_mask"]
    assert test
    assert list(test.keys()) == ["input_ids", "token_type_ids", "attention_mask"]
