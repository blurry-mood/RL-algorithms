from torch import nn

class QNetwork(nn.Module):

    def __init__(self, input_channels, num_actions) -> None:
        super().__init__()

        self.model = nn.Sequential(
                    nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, stride=2), 
                    nn.MaxPool2d(3, stride=3, padding=1),
                    nn.PReLU(),
                    nn.BatchNorm2d(64),
                    nn.Dropout(0.3),
                    
                    nn.MaxPool2d(3, stride=3, padding=1),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), 
                    nn.PReLU(),
                    nn.BatchNorm2d(128),
                    nn.Dropout(0.3),

                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(128, 64),
                    nn.GELU(),
                    nn.Linear(64, 1024),
                    nn.ELU(),
                    nn.Dropout(0.2),
                    nn.Linear(1024, num_actions)
                )

    def forward(self, x):
        # print(x.min(), x.max(), x.mean(), x.std())
        prob = self.model(x)
        return prob