import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Player(nn.Module):
    def __init__(self, rows=8, gpu=False):
        super(Player, self).__init__()

        num_tiles = (rows // 2) * 3 # 3 rows of tiles (3 rows for each player)
        output_size = num_tiles * 4 # 4 possible moves for each tile

        # Fully connected layers
        self.MLP = nn.Sequential(
            nn.Linear(rows**2, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Output layer
        self.output = nn.Linear(128, output_size)

        # Initialize weights
        # self.__init_weights()

        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.to(self.device)

    # def __init_weights(self):
        # initialize weights randomly for each call to the constructor
        # for layer in self.MLP:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.random_uniform_(layer.weight, -0.1, 0.1)
        #         nn.init.zeros_(layer.bias)
        

    def forward(self, x: np.ndarray):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Forward pass
        x = self.MLP(x)

        # Output layer
        x = self.output(x)

        # Softmax activation
        o = F.softmax(x, dim=0)
        
        # Reshape the output
        o = o.view(x.size(0)//4, -1, 4)

        return o
    
    def get_weights(self):
        return [param.data.cpu().numpy() for param in self.parameters()]

"""
[[0.0002, 0.0003, 0.0004, 0.0005],
[0.0006, 0.0007, 0.0008, 0.0009],
[0.0010, 0.0011, 0.0012, 0.0013],
[0.0014, 0.0015, 0.0016, 0.0017],
[0.0018, 0.0019, 0.0020, 0.0021],
[0.0022, 0.0023, 0.0024, 0.0025],
[0.0026, 0.0027, 0.0028, 0.0029],
[0.0030, 0.0031, 0.0032, 0.0033],
[0.0034, 0.0035, 0.0036, 0.0037],
[0.0038, 0.0039, 0.0040, 0.0041],
[0.0042, 0.0043, 0.0044, 0.0045],
[0.0046, 0.0047, 0.0048, 0.0049],
[0.0050, 0.0051, 0.0052, 0.0053],
[0.0054, 0.0055, 0.0056, 0.0057],
[0.0058, 0.0059, 0.0060, 0.0061],
[0.0062, 0.0063, 0.0064, 0.0065],
[0.0066, 0.0067, 0.0068, 0.0069],
[0.0070, 0.0071, 0.0072, 0.0073],
[0.0074, 0.0075, 0.0076, 0.0077],
[0.0078, 0.0079, 0.0080, 0.0081],
[0.0082, 0.0083, 0.0084, 0.0085],
[0.0086, 0.0087, 0.0088, 0.0089],
[0.0090, 0.0091, 0.0092, 0.0093],
[0.0094, 0.0095, 0.0096, 0.0097],
[0.0098, 0.0099, 0.0100, 0.0101],
[0.0102, 0.0103, 0.0104, 0.0105],
[0.0106, 0.0107, 0.0108, 0.0109],
[0.0110, 0.0111, 0.0112, 0.0113],
[0.0114, 0.0115, 0.0116, 0.0117]
]
"""