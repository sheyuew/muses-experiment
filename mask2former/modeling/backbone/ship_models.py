import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


@BACKBONE_REGISTRY.register()
class DenseAutoEncoder(nn.Module):
    def __init__(self, input_shape, code_dim=32, hidden_dim=128):
        super().__init__()
        
        input_dim = torch.prod(torch.tensor(input_shape))
        
        self.encoder = nn.Sequential(
            ### BEGIN SOLUTION
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=784, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32, bias=True)
            ### END SOLUTION
            )
        
        self.decoder = nn.Sequential(
            ### BEGIN SOLUTION
            nn.Linear(in_features=32, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=784, bias=True),
            nn.Unflatten(dim=-1, unflattened_size=torch.Size([1, 28, 28]))
            ### END SOLUTION
            )

    def forward(self, ims):
        ### BEGIN SOLUTION
        # Compute the codes using the encoder
        codes = self.encoder(ims)

        # Compute the estimated images using the decoder
        ims_est = self.decoder(codes)

        ### END SOLUTION
        return ims_est, codes
    
