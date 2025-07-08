import torchvision
import torchvision.transforms as tf
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

class MNISTDataset(Dataset):
    def __init__(self, train: bool = True, transform=None):
        self.dataset = torchvision.datasets.MNIST(
            root='./data',
            train = train,
            download=True
        )
        self.transform = tf.ToTensor()

    def __getitem__(self, index):
        return self.transform(self.dataset[index][0]), self.dataset[index][1]
    
    def __len__(self):
        return len(self.dataset)
    
class CIFARDataset(Dataset):
    def __init__(self, train: bool = True):
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train = train,
            download=True
        )
        self.transform = tf.ToTensor()

    def __getitem__(self, index):
        return self.transform(self.dataset[index][0]), self.dataset[index][1]
    
    def __len__(self):
        return len(self.dataset)
    
def get_mnist_dataloaders(batch_size: int = 32):
    transform = tf.Compose([
        tf.ToTensor(),
        tf.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNISTDataset(train=True, transform=transform)
    test_dataset = MNISTDataset(train=False, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def get_cifar_dataloaders(batch_size: int = 128):
    transform = tf.Compose([
        tf.ToTensor(),
        tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = MNISTDataset(train=True, transform=transform)
    test_dataset = MNISTDataset(train=False, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
