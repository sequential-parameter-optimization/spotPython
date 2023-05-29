from torchvision import datasets
import torchvision.transforms as transforms


def load_data_cifar10(data_dir="./data"):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset
