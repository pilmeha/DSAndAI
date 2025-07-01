import torch
from utils import CustomDataset, make_classification_data, mse, log_epoch, accuracy
from torch.utils.data import DataLoader
from tqdm import tqdm

# (0; 1)
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (torch.exp(-x) + 1)

class LogisticRegression(): 
    def __init__(self, in_features):
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)

    def __call__(self, X):
        return sigmoid(X @ self.w + self.b)
    
    def forward(self, X):
        return self.__call__(X)

    def backward(self, X: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor) -> None:
        n = X.size(0)
        error = y - y_pred
        self.dw = -1 / n * X.T @ error
        self.db = -error.mean()

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

if __name__ == '__main__':
    EPOCHS = 100

    X, y =  make_classification_data()
    dataset = CustomDataset(X, y)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
    )

    model = LogisticRegression(X.shape[1])
    lr = 0.1
    epochs = 100

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0

        for i, (batch_x, batch_y) in enumerate(dataloader):
            y_pred = model.forward(batch_x)
            loss = -(batch_y * torch.log(y_pred + 1e-8) + (1 - batch_y) * torch.log((1 - y_pred) + 1e-8)).mean()
            acc = accuracy(y_pred, batch_y)
            epoch_loss += loss.item()

            model.backward(batch_x, batch_y, y_pred)
            model.step(lr)
        avg_loss = epoch_loss / len(dataloader)
        avg_acc = epoch_acc / len(dataloader)
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)

    print(model.w, model.b)

