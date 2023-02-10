
import torch
import torch.nn as nn
import pytorch_lightning as pl

class KeypointModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        num_filters = 32
        kernel_size = 3
        padding = 1
        stride = 1

        self.cnn = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(32, num_filters * 2, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(64, num_filters * 4, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(128, num_filters * 8, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 30),
            nn.Tanh()
        )

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)

        x = self.cnn(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.fc(x)

        return x


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
