import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from transformers import AlbertModel
eps = 1e-12

class AlbertTextModel(nn.Module): # AlbertTextModel
    def __init__(self, hyp_params):
        super(AlbertTextModel, self).__init__()
        self.activation = nn.ReLU()
        output_dim = hyp_params.output_dim
        self.linear_transformation = nn.Linear(300, 128)
        self.model = AlbertModel.from_pretrained('albert-base-v2')
        self.text_mlp = nn.Linear(768, output_dim)

    def forward(self, x_l):
        padded_tensor = self.linear_transformation(x_l)
        outputs = self.model(inputs_embeds=padded_tensor)
        last_hidden_states = outputs.last_hidden_state
        pooled_out = torch.mean(last_hidden_states, dim=1)
        t_pred = self.text_mlp(pooled_out)
        
        return t_pred