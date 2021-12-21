#! python3
"""The training script."""

import argparse
import os

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from network import CNN
from dataset import train as get_train_data


HERE = os.path.dirname(os.path.abspath(__file__))


def train(net: nn.Module, epochs: int, lr_decay, pbar: bool = True):
    """Run the training loop."""
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    schedule = optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=lr_decay)
    train_data = get_train_data()

    net.train()

    epoch_it = trange(epochs, desc="epochs", disable=not pbar)
    batch_it = tqdm(train_data, desc="batches", disable=not pbar)
    for epoch in epoch_it:
        for images, labels in batch_it:
            batch_input = Variable(images)
            batch_labels = Variable(labels)

            batch_output = net(batch_input)
            loss = loss_func(batch_output, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()
        epoch_it.set_description(f"Epoch {epoch + 1} loss: {loss.item()}")
        batch_it.refresh()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e", type=int, help="Number of epochs to run.", required=True
    )

    default_path = os.path.join(HERE, "data", "cnn.pth")
    parser.add_argument("--output-path", default=default_path)

    parser.add_argument("--lr-decay", type=float, default=.9)

    parser.add_argument("-p", type=bool, help="Turn off progress bars.")

    args = parser.parse_args()

    net = CNN()

    train(net, epochs=args.e, lr_decay=args.lr_decay)

    torch.save(net.state_dict(), args.output_path)
