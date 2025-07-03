import torchvision
import torchvision.transforms as tf
import plotly as ptl

from torch.utils.data import DataLoader, Dataset

class MNISTDataset(Dataset):
    def __init__(self, train: bool = True):
        self.dataset = torchvision.datasets.MNIST(
            root='./data',
            train=train,
            download=True
        )
        self.transform = tf.ToTensor()

    def __getitem__(self, index):
        return self.transform(self.dataset[index][0]), self.dataset[index][1]
    
    def __len__(self):
        return len(self.dataset)

ds = CIFARDataset()
image = ds[0][0].permute(1, 2, 0).numpy()
plt.imshow(image, cmap='gray')
plt.show()
print(ds[0][1])

