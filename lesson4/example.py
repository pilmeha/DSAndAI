import torch
from datasets import get_mnist_loaders
from models import SimpleCNN, CNNwithResidual
from trainer import train_model
from utils import plot_training_history, count_parameters, compare_models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

train_loader, test_loader = get_mnist_loaders(batch_size=64)

simple_cnn = SimpleCNN(input_channels=1, num_classes=10).to(device)
residual_cnn = CNNwithResidual(input_channels=1, num_classes=10).to(device)

print(f'Simple CNN parameters: {count_parameters(simple_cnn)}')
print(f'Residual CNN parameters: {count_parameters(residual_cnn)}')