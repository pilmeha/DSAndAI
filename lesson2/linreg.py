# y = w1 * x1 + w2 * x2 + ... + b

# Y = X_{n x m} @ w_{m c 1} + b

# 

# L: f: (w, X, Y') -> R

# L = (Y - Y')^2 / n

# dL/dw = -2 * X^T @ (Y' - Y) / n

# dL/db = -2 / n * (Y - Y)

import torch
from utils import CustomDataset, make_regression_data, mse, log_epoch
from torch.utils.data import DataLoader
from tqdm import tqdm


class LinearRegression(): 
    def __init__(self, in_features):
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)

    def __call__(self, X):
        return X @ self.w + self.b
    
    def forward(self, X):
        return self.__call__(X)

    def backward(self, X: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor) -> None:
        n = X.size(0)
        self.dw = -1 / n * X.T @ (y - y_pred)
        self.db = -(y - y_pred).mean()

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return ((y_pred - y_true) ** 2).mean()

if __name__ == '__main__':
    EPOCHS = 100

    X, y =  make_regression_data(10000)
    dataset = CustomDataset(X, y)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
    )

    model = LinearRegression(1)
    lr = 0.1
    epochs = 100

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0

        for i, (batch_x, batch_y) in enumerate(dataloader):
            y_pred = model.forward(batch_x)
            loss = mse(y_pred, batch_y)
            epoch_loss += loss.item()

            model.backward(batch_x, batch_y, y_pred)
            model.step(lr)
        avg_loss = epoch_loss / len(dataloader)
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)

    print(model.w, model.b)

