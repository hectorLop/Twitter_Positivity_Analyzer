"""
Training module unitary testing.

This module tests if the training function works.
"""
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.tokenization_utils import PreTrainedTokenizer

from tests.fixtures import load_ready_training_data
from twitter_analyzer.utils import train


def test_train(load_ready_training_data):
    """Test the training function for an epoch."""
    train_data, val_data = load_ready_training_data

    dataloader_train = DataLoader(
        train_data,
        sampler=RandomSampler(train_data),
        batch_size=1,
        num_workers=4,
    )
    dataloader_val = DataLoader(
        val_data,
        sampler=RandomSampler(val_data),
        batch_size=1,
        num_workers=4,
    )

    print("Creating model")
    # Create the model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=5,
        output_attentions=False,
        output_hidden_states=False,
    )

    # Create the optimizer and the scheduler
    optimizer = AdamW(model.parameters(), lr=0.00001, eps=1e-8)

    print("Start training")
    # Train the model
    model = train(
        model,
        train_loader=dataloader_train,
        val_loader=dataloader_val,
        optimizer=optimizer,
        scheduler=None,
        epochs=1,
        checkpoint="",
        device="cpu",
    )
