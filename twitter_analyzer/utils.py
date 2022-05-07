"""
Twitter Analyzer utils.

This module contains training utility functions.
"""
import gc
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification


def evaluate(
    dataloader: DataLoader, model: BertForSequenceClassification
) -> Tuple[float, List[int], List[int]]:
    """
    Evaluate a model's performance.

    Args:
        dataloader (DataLoader): Data to evaluate the model.
        model (BertForSequenceClassification): BERT model.

    Returns:
        Tuple: A tuple containing:
            float: Average loss.
            List[int]: List the predictions.
            List[int]: List containing the ground truth
    """
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in tqdm(dataloader):
        # Put the data into the GPU
        batch = tuple(b.to("cuda") for b in batch)

        # Create the input
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        # Get the output without gradients
        with torch.no_grad():
            outputs = model(**inputs)

        # Compute loss
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        # Get the predictions and true values
        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    # compute average loss
    loss_val_avg = loss_val_total / len(dataloader)

    predictions = np.concatenate(predictions, axis=0).argmax(1)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


def train(
    model: BertForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epochs: int,
) -> BertForSequenceClassification:
    """
    Train a BERT model.

    Args:
        model (BertForSequenceClassification): BERT model.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (torch.optim.Optimizer): Optimizer algorithm.
        scheduler (LambdaLR): Learning rate scheduler.
        epochs (int): Number of epochs.

    Returns:
        BertForSequenceClassification: Trained BERT model.
    """
    for epoch in range(1, epochs + 1):
        model.train()
        model.to("cuda")

        best_val_loss = np.inf
        loss_train_total = 0
        train_preds, train_true_vals = [], []

        for batch in tqdm(train_loader, desc="Epoch {:1d}".format(epoch)):
            # Set gradients to zero
            model.zero_grad()
            optimizer.zero_grad()

            # Put the data into the GPU
            batch = tuple(b.to("cuda") for b in batch)
            # Define the input
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
            }

            outputs = model(**inputs)

            # Get the loss and compute the backward gradients
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            # Get the predictions and true values
            logits = outputs[1].detach().cpu().numpy()
            label_ids = inputs["labels"].cpu().numpy()
            train_preds.append(logits)
            train_true_vals.append(label_ids)

            # Clip the gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters
            optimizer.step()
            scheduler.step()

        # Compute training accuracy
        train_preds = np.concatenate(train_preds, axis=0).argmax(1)
        train_true_vals = np.concatenate(train_true_vals, axis=0)
        train_acc = accuracy_score(train_true_vals, train_preds)

        tqdm.write("\n Epoch {epoch}")

        # Training metrics
        loss_train_ave = loss_train_total / len(train_loader)
        tqdm.write(f"Training loss: {loss_train_ave}")
        tqdm.write(f"Training accuracy: {train_acc}")

        # Validation metrics
        val_loss, val_preds, val_true_vals = evaluate(val_loader, model)
        val_acc = accuracy_score(val_true_vals, val_preds)
        tqdm.write(f"Validation loss: {val_loss}")
        tqdm.write(f"Val accuracy: {val_acc}")

        # Save checkpoint if the validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"models/BERT_ft_epoch{epoch}.model")

        gc.collect()

    return model
