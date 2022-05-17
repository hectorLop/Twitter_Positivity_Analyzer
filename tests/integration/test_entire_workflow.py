"""
Integration tests.

This module contains the tests to integrate the
data preprocessing and the training modules.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import BertForSequenceClassification

from twitter_analyzer import TESTS_DIR
from twitter_analyzer.train import main


class Hyperparameters:
    """Experiment testing hyperparameters."""

    train: str = str(Path(TESTS_DIR, "data/sub_train_sample.csv"))
    test: str = str(Path(TESTS_DIR, "data/sub_test_sample.csv"))
    batch_size: int = 2
    lr: float = 0.00001
    epochs: int = 1
    warmup_steps: int = 0
    checkpoint: str = ""
    device: str = "cpu"


def test_entire_workflow():
    """Test the entire experimentation workflow."""
    base_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=5,
        output_attentions=False,
        output_hidden_states=False,
    )

    initial_parameters = base_model.parameters()
    for param in initial_parameters:
        init = param.data.detach().clone()
        break

    trained_model = main(Hyperparameters())

    print("AAAAAAAAAAA")
    final_parameters = trained_model.parameters()
    for param in final_parameters:
        final = param.data.detach().clone()
        break

    print("FFFFFFFF")
    assert not torch.equal(init, final)
