from torch.utils.data import DataLoader
from homework_datasets import CSVDataset, clean_car_data
import torch
import torch.nn as nn
import torch.optim as optim
from utils import make_regression_data, mse, log_epoch, RegressionDataset
from homework_model_modification_linreg import LinearRegression

# class LinearRegression(nn.Module):
#     def __init__(self, in_features, lambda_):
#         super().__init__()
#         self.linear = nn.Linear(in_features, 1)
#         self.lambda_ = lambda_

#     def forward(self, x):
#         return self.linear(x)
    
#     def l1_reg(self):
#         return self.lambda_ * torch.sum(torch.abs(self.linear.weight.data))

#     def l2_reg(self):
#         return self.lambda_ * torch.sum(torch.pow(self.linear.weight.data, 2))

if __name__ == '__main__':
    df = clean_car_data("C:\\Users\\garma\\Downloads\\archive\\CarPrice_Assignment.csv")

    dataset = CSVDataset(
        dataframe=df,
        target_column='price',
        # cat_columns=['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'fuelsystem'],
        cat_columns=['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem'],
        num_columns=['symboling', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg'],
        normalize=True,
        encode='label'  # или 'label'
    )

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    for X_batch, y_batch in dataloader:
        print("X:", X_batch.shape)
        print("y:", y_batch.shape)

        print("NaN в X:", torch.isnan(X_batch).any())
        print("NaN в y:", torch.isnan(y_batch).any())

        break

    # Параметры
    reg = 'l1'
    lambda_ = 0.01
    learning_rate = 0.1
    patience = 10
    min_delta = 1e-4
    in_features = dataset[0][0].shape[0]

    # Создаём модель, функцию потерь и оптимизатор
    model = LinearRegression(in_features=in_features, lambda_=lambda_)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # # Early Stopping
    # best_loss = float('inf')
    # epochs_no_improve = 0

    # Обучаем модель
    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in dataloader:
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

    # Сохраняем модель
    torch.save(model.state_dict(), 'linreg_torch_cars_prices.pth')

        # if avg_loss + min_delta < best_loss:
        #     best_loss = avg_loss
        #     epochs_no_improve = 0
        #     # Сохраняем лучшую модель
        #     torch.save(model.state_dict(), 'linreg_torch_es.pth')
        # else:
        #     epochs_no_improve +=1 
        #     if epochs_no_improve >= patience:
        #         print(f"Early stoppin на эпохе {epoch}")
        #         break
    
    # Загружаем модель
    new_model = LinearRegression(in_features=in_features, lambda_=lambda_)
    new_model.load_state_dict(torch.load('linreg_torch_cars_prices.pth'))
    new_model.eval() 

    print(model.linear.weight.data, model.linear.bias.data)
