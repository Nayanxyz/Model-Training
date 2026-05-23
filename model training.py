import torch
import torch.nn as nn
import torch.optim as optim


# 1. THE ARCHITECTURE (The Brain)
class SimpleImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # The Magnifying Glass: Scans a black-and-white image (1 channel) using a 3x3 box
        self.scanner = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)

        # The Decision Maker: Takes the scanned patterns and picks a number 0-9 (10 options)
        # Note: 26x26 comes from shrinking the 28x28 image with the 3x3 scanner
        self.decision = nn.Linear(in_features=16 * 26 * 26, out_features=10)

