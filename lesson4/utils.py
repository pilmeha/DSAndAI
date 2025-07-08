import torch
import matplotlib.pyplot as plt

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplot(1, 2, figsize=(12, 4))

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.tight_layout()a
    plt.show()
    
def count_parameters(model):
    return sum(p.numer() for p in model.parameters() if p.pa)

def load_model(model, path):
    model.load_dtate_dict(torch.load(path))
    return model

def compare_models(fc_history, cnn_history):
    fig, (ax1, ax2) = plt.subplot(1, 2, figsize(12,4))

