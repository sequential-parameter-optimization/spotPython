from torchvision import datasets
import torchvision.transforms as transforms
from typing import Tuple


def load_data_cifar10(data_dir: str = "./data") -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Load the CIFAR-10 dataset.
        This function loads the CIFAR-10 dataset using the torchvision library.
        The data is split into a training set and a test set.

    Args:
        data_dir (str):
            The directory where the data is stored. Defaults to "./data".

    Returns:
        Tuple[datasets.CIFAR10, datasets.CIFAR10]:
            A tuple containing the training set and the test set.

    Examples:
        >>> trainset, testset = load_data_cifar10()
        >>> print(f"Training set size: {len(trainset)}")
        Training set size: 50000
        >>> print(f"Test set size: {len(testset)}")
        Test set size: 10000

    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset


# Example usage
if __name__ == "__main__":
    # Load the CIFAR-10 dataset
    trainset, testset = load_data_cifar10()

    # Print the size of the training set and the test set
    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")
