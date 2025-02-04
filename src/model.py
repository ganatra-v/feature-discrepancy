import numpy as np
import torch
import torch.nn as nn

class neural_network(nn.Module):
  def __init__(self, num_features, hidden_layers, hidden_size, num_classes):
    super(neural_network, self).__init__()

    activation = nn.Sigmoid()

    layers = [
        nn.Linear(num_features, hidden_size),
        activation,
    ]

    for hidden_ in range(hidden_layers):
      layers.extend([
          nn.Linear(hidden_size, hidden_size),
          activation,
      ])

    layers.append(nn.Linear(hidden_size, num_classes))
    self.layers = nn.Sequential(*layers)

    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    return self.layers(x)

  def infer(self, x):
    self.eval()
    with torch.no_grad():
      if type(x) is np.ndarray: # only pass np.ndarray if model is on cpu
        x = torch.from_numpy(x.astype(np.float32)).cuda()
      x = self.forward(x)
      x = self.sigmoid(x).cpu().detach()
      return x