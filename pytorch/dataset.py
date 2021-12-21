"""Module providing the dataset ready to use.

In the MNIST example, this is a little overkill, since the majority of this
funcionality already is part of the torchvision dataset accessor.

"""

import functools
import os
from typing import List, Optional, Callable

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data")


def _chain(funcs):
    """Chain a list of functions, left-to-right.

    Credit goes to https://mathieularose.com/function-composition-in-python/

    """
    if not funcs:
        raise ValueError("Can't chain zero functions.")
    return functools.reduce(
        lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs
    )


def download():
    """Download the MNIST dataset."""
    MNIST(root=DATA_PATH, download=True)


def train(
    batch_size: int = 32,
    shuffle: bool = True,
    augmentations: Optional[List[Callable]] = None,
    **kwargs: dict
) -> DataLoader:
    """Get the training dataset ready for use after applying augmentations.

    In this simple example, this just defaults to calling the dataset object.

    Parameters
    ----------
    batch_size
        Number of samples per batch
    shuffle
        Whether to shuffle the data after each epoch
    augmentations
        Augmentations to apply to the PIL images in the dataset.
        Functions are applied left-to-right.
        If torchvision.transforms.ToTensor is not in the list, it will be
        appended.
    kwargs
        kwargs are forwarded to the DataLoader constructor.

    Returns
    -------
    torch.utils.DataLoader
        The pre-processed dataset as a DataLoader

    """
    augmentations = augmentations or []
    if ToTensor not in augmentations:
        augmentations.append(ToTensor())

    transform = _chain(augmentations)
    train_data = MNIST(root=DATA_PATH, train=True, transform=transform)
    data_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
    }
    data_loader_kwargs.update(kwargs)
    return DataLoader(train_data, **data_loader_kwargs)


def test(
    batch_size: int = 32, augmentations: Optional[List[Callable]] = None, **kwargs
) -> MNIST:
    """Get the test dataset ready for use after applying augmentations.

    In this simple example, this just defaults to calling the dataset object.

    Parameters
    ----------
    batch_size
        Number of samples ber batch
    augmentations
        Augmentations to apply to the PIL images in the dataset.
        Functions are applied left-to-right.
        If torchvision.transforms.ToTensor is not in the list, it will be
        appended.
    kwargs
        kwargs are forwarded to the DataLoader constructor.

    Returns
    -------
    torchvision.datasets.MNIST
        The pre-processed dataset as CWH tensors.

    """
    test_data = MNIST(root=DATA_PATH, train=False, transform=ToTensor())
    data_loader_kwargs = {"batch_size": batch_size}
    data_loader_kwargs.update(kwargs)
    return DataLoader(test_data, **data_loader_kwargs)


if __name__ == "__main__":
    download()
