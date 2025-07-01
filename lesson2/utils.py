import torch
from torch.utils.data import Dataset, DataLoader

# __len__(self), __getitem__(self, idx)


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

def make_classification_data():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = torch.tensor(data['data'], dtype=torch.float32)
    y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
    return X, y

def make_regression_data(n=100, noize=0.2, source='random'):
    if source == 'random':
        X = torch.randn(n,1)
        w, b = -5, 10
        y = w * X + b + noize * torch.randn(n, 1)
        return X, y
    elif source == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unk source')
    
def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return ((y_pred - y_true) ** 2).mean()

def log_epoch(epoch, avg_loss, **metrics):
    message = f'Epoch: {epoch}\tloss:{avg_loss:.4f}'
    for k, v in metrics.items():
        message += f'\t{k}: {v:.4f}'

def accuracy(y_pred, y_true):
    y_pred_bin = (y_pred > 0.5).float()
    return (y_pred_bin == y_true).float().mean().item()