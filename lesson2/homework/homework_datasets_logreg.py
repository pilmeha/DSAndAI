from torch.utils.data import DataLoader
from homework_datasets import CSVDataset, clean_weather_data
import torch
import torch.nn as nn
import torch.optim as optim
from utils import make_regression_data, mse, log_epoch, RegressionDataset, accuracy
from homework_model_modification_logreg import LogisticRegression


if __name__ == '__main__':

    df = clean_weather_data("C:\\Users\\garma\\Downloads\\archive (1)\\weatherAUS.csv")
    dataset = CSVDataset(
        dataframe=df,
        target_column='RainTomorrow',
        # cat_columns=['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'fuelsystem'],
        cat_columns=['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'],
        num_columns=['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'],
        normalize=True,
        encode='onehot'  # или 'label'
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')

    # Параметры
    num_classes = 2
    learning_rate = 0.1
    in_features = dataset[0][0].shape[0]
    epochs = 100

    # Создаём модель, функцию потерь и оптимизатор
    model = LogisticRegression(in_features=in_features, num_classes=num_classes)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_acc = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_x)
            y_pred = model.forward(batch_y)
            loss = criterion.forward(y_pred, batch_y)
            loss.backward()
            optimizer.step()

            # Вычисляем accuracy
            y_pred = torch.argmax(logits, dim=1)
            total_acc += accuracy(y_pred, batch_y)
            total_loss += loss.item()
        

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss, accuracy=avg_acc)

    print(model.linear.weight.data, model.linear.bias.data)