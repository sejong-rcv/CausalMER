import torch
from torch import nn
import torch.nn.functional as F
import torchvision

class VisionModel(nn.Module): 
    def __init__(self, hyp_params):
        super(VisionModel, self).__init__()
        self.activation = nn.ReLU()
        output_dim = hyp_params.output_dim
        self.v_branch = torchvision.models.resnet18()
        self.v_branch.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.v_branch.fc = nn.Linear(512, output_dim, bias=True)

    def forward(self, x_v):
        x_v = x_v.reshape(x_v.shape[0], 1, x_v.shape[1], x_v.shape[2])
        v_pred = self.v_branch(x_v)
        return v_pred