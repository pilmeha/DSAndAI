#Задание 1: Модификация существующих моделей (30 баллов)
#1.1 Расширение линейной регрессии (15 баллов)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_regression_data, mse, log_epoch, RegressionDataset

class LinearRegression(nn.Module):
    def __init__(self, in_features, lambda_):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.lambda_ = lambda_

    def forward(self, x):
        return self.linear(x)
    
    def l1_reg(self):
        return self.lambda_ * torch.sum(torch.abs(self.linear.weight.data))

    def l2_reg(self):
        return self.lambda_ * torch.sum(torch.pow(self.linear.weight.data, 2))

if __name__ == '__main__':
    # Генерируем данные
    X, y = make_regression_data(n=200)
    
    # Создаём датасет и даталоадер
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    
    # Параметры
    reg = 'l2'
    lambda_ = 0.01
    learning_rate = 0.1
    patience = 10
    min_delta = 1e-4
    in_features = dataset[0][0].shape[0]

    # Создаём модель, функцию потерь и оптимизатор
    model = LinearRegression(in_features=in_features, lambda_=lambda_)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Early Stopping
    best_loss = float('inf')
    epochs_no_improve = 0

    # Обучаем модель
    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)

            if reg == 'l1':
                loss += model.l1_reg()

            elif reg == 'l2':
                loss += model.l2_reg()

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)

    # # Сохраняем модель
    # torch.save(model.state_dict(), 'linreg_torch_l2.pth')

        if avg_loss + min_delta < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            # Сохраняем лучшую модель
            torch.save(model.state_dict(), 'linreg_torch_es.pth')
        else:
            epochs_no_improve +=1 
            if epochs_no_improve >= patience:
                print(f"Early stoppin на эпохе {epoch}")
                break
    
    
    # Загружаем модель
    new_model = LinearRegression(in_features=1, lambda_=lambda_)
    new_model.load_state_dict(torch.load('linreg_torch_es.pth'))
    new_model.eval() 

    print(model.linear.weight.data, model.linear.bias.data)