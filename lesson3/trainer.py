import torch
import tqdm import tqdm

def run_epoch(model, data_loader, loss_fn, optimizer=None, device='cude', is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    total_correct = 0
    total = 0

    for i, (image, target) in enumerate(tqdm(data_loader)):
        image, target = image.to(device), target.to(device)

        if not is_test and optimizer is not None:
            optimizer.zero_grad()

        

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, devices='cude'):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(epochs):
