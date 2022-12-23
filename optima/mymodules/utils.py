from datetime import datetime
import torch


def accuracy_fct(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    predictions = predictions.cpu().tolist()
    acc = sum([int(i == j) for i, j in zip(predictions, labels.cpu().tolist())]) / len(predictions)
    return acc


def format_time():
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m-%d-%H:%M:%S")
    return date_time




