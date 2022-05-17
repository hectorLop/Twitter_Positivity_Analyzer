"""
Training script.

This module allows to train a BERT model to
classify tweets according their positivity
"""
from argparse import ArgumentParser, Namespace
from typing import Tuple

from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.tokenization_utils import PreTrainedTokenizer

from twitter_analyzer.data_preprocessing import (
    LABEL_DICT,
    convert_to_datasets,
    get_dataset_splits,
    load_raw_dataset,
    tokenize_splits,
)
from twitter_analyzer.utils import train


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
    raw_dataset["Sentiment"] = raw_dataset["Sentiment"].replace(LABEL_DICT)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_dataset_splits(
        raw_dataset
    )

    X_train, X_val, X_test = tokenize_splits(tokenizer, X_train, X_val, X_test)

    train, val, test = convert_to_datasets(
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    )

    return train, val, test


def main(args: Namespace) -> None:
    """
    Perform a training experiment.

    Args:
        args (Nampespace): Training hyperparameters.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    print("Getting data")
    train_set, val_set, _ = get_data(
        train_file=args.train, test_file=args.test, tokenizer=tokenizer
    )

    print("Creating dataloaders")
    # Create dataloaders
    dataloader_train = DataLoader(
        train_set,
        sampler=RandomSampler(train_set),
        batch_size=args.batch_size,
        num_workers=4,
    )
    dataloader_val = DataLoader(
        val_set,
        sampler=RandomSampler(val_set),
        batch_size=args.batch_size,
        num_workers=4,
    )

    print("Creating model")
    # Create the model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(LABEL_DICT),
        output_attentions=False,
        output_hidden_states=False,
    )

    # Create the optimizer and the scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(dataloader_train) * args.epochs,
    )
    print("Start training")
    # Train the model
    model = train(
        model,
        train_loader=dataloader_train,
        val_loader=dataloader_val,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        checkpoint=args.checkpoint,
        device=args.device,
    )

    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", required=True, help="Training set filepath")
    parser.add_argument("--test", required=True, help="Test set filepath")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument(
        "--warmup_steps", type=int, default=0, help="Number of warmup steps"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/BERT_model_epoch",
        help="Checkpoint path",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Training device")

    args = parser.parse_args()

    main(args)
