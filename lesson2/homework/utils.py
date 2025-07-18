import torch
import numpy as np
from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_regression_data(n=100, noise=0.1, source='random'):
    if source == 'random':
        X = torch.rand(n, 1)
        w, b = 2.0, -1.0
        y = w * X + b + noise * torch.randn(n, 1)
        return X, y
    elif source == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32) # type: ignore
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1) # type: ignore
        return X, y
    else:
        raise ValueError('Unknown source')

def make_classification_data(n=100, source='random'):
    if source == 'random':
        X = torch.rand(n, 2)
        w = torch.tensor([2.0, -3.0])
        b = 0.5
        logits = X @ w + b
        y = (logits > 0).float().unsqueeze(1)
        return X, y
    elif source == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = torch.tensor(data['data'], dtype=torch.float32) # type: ignore
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1) # type: ignore
        return X, y
    else:
        raise ValueError('Unknown source')

def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean().item()

def accuracy(y_pred, y_true):
    y_pred_bin = (y_pred > 0.5).float()
    return (y_pred_bin == y_true).sum().item()

def log_epoch(epoch, loss, **metrics):
    msg = f"Epoch {epoch}: loss={loss:.4f}"
    for k, v in metrics.items():
        msg += f", {k}={v:.4f}"
    print(msg)

# import torch
# from torch.utils.data import Dataset, DataLoader

# # __len__(self), __getitem__(self, idx)


# class CustomDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, index):
#         return self.X[index], self.y[index]

# def make_classification_data():
#     from sklearn.datasets import load_breast_cancer
#     data = load_breast_cancer()
#     X = torch.tensor(data['data'], dtype=torch.float32)
#     y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
#     return X, y

# def make_regression_data(n=100, noize=0.2, source='random'):
#     if source == 'random':
#         X = torch.randn(n,1)
#         w, b = -5, 10
#         y = w * X + b + noize * torch.randn(n, 1)
#         return X, y
#     elif source == 'diabetes':
#         from sklearn.datasets import load_diabetes
#         data = load_diabetes()
#         X = torch.tensor(data['data'], dtype=torch.float32)
#         y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
#         return X, y
#     else:
#         raise ValueError('Unk source')
    
# def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#     return ((y_pred - y_true) ** 2).mean()

# def log_epoch(epoch, avg_loss, **metrics):
#     message = f'Epoch: {epoch}\tloss:{avg_loss:.4f}'
#     for k, v in metrics.items():
#         message += f'\t{k}: {v:.4f}'

# def accuracy(y_pred, y_true):
#     y_pred_bin = (y_pred > 0.5).float()
#     return (y_pred_bin == y_true).float().mean().item()