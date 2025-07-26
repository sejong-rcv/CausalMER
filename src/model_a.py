import torch
from torch import nn
import torch.nn.functional as F
import torchvision

eps = 1e-12

class AudioModel(nn.Module): 
    def __init__(self, hyp_params):
        super(AudioModel, self).__init__()
        self.activation = nn.ReLU()
        output_dim = hyp_params.output_dim
        self.a_branch = torchvision.models.resnet18()
        self.a_branch.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.a_branch.fc = nn.Linear(512, output_dim, bias=True)

    def forward(self, x_a):
        x_a = x_a.reshape(x_a.shape[0], 1, x_a.shape[1], x_a.shape[2])
        a_pred = self.a_branch(x_a)
        return a_pred