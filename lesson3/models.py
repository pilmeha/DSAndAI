import torch
import json

class FCN(torch.nn.Module):
    def __init__(self, config_path: str = None, )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

if __name__ == '__main__':
    config = {
        'input_size'
    }
