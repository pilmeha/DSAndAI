import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_classification_data, accuracy, log_epoch, ClassificationDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class LogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    # Генерация многоклассовых данных
    from sklearn.datasets import make_classification
    X_np, y_np = make_classification(
        n_samples=500, 
        n_features=2, 
        n_classes=3, 
        n_informative=2, 
        n_redundant=0, 
        n_clusters_per_class=1
    )
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    
    # Создаём датасет и даталоадер
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    
    # Параметры
    num_classes = 3
    learning_rate = 0.1

    # Создаём модель, функцию потерь и оптимизатор
    model = LogisticRegression(in_features=2, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Обучаем модель
    epochs = 100
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_acc = 0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            # Вычисляем accuracy
            y_pred = torch.argmax(logits, dim=1)
            total_acc += accuracy(y_pred, batch_y)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss, acc=avg_acc)
    
    # Сохраняем модель
    torch.save(model.state_dict(), 'logreg_torch.pth')
    
    # Загружаем модель
    new_model = LogisticRegression(in_features=2, num_classes=num_classes)
    new_model.load_state_dict(torch.load('logreg_torch.pth'))
    new_model.eval() 
    with torch.no_grad():
        logits = new_model(X)
        y_pred = torch.argmax(logits, dim=1)

    y_true_np = y_np
    y_pred_np = y_pred.numpy()

    # Метрики
    print("Classification report:")
    print(classification_report(y_true_np, y_pred_np))

    # ROC-AUC (для многоклассового случая нужен one-hot)
    y_true_1hot = np.eye(num_classes)[y_true_np]
    y_score = torch.softmax(logits, dim=1).numpy()
    auc = roc_auc_score(y_true_1hot, y_score, multi_class='ovr')
    print(f"ROC-AUC score: {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    