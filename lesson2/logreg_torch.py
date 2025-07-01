import torch
from torch.utils.data import DataLoader
from utils import make_regression_data, CustomDataset, accuracy, log_epoch

class LogisticRegressionTorch(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear= torch.nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear.forward(x)
    
if __name__ == '__main__':
    EPOCHS = 100

    X, y =  make_regression_data(10000)
    dataset = CustomDataset(X, y)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
    )

    lr = 0.1

    model = LogisticRegressionTorch(X.shape[1])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        total_acc = 0
        for i, (batch_x, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model.forward(batch_y)
            loss = loss_fn.forward(y_pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy(torch.sigmoid(y_pred), batch_y)

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss, accuracy=avg_acc)

    print(model.linear.weight.data, model.linear.bias.data)
