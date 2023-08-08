import torch
import torch.nn as nn
import torch.nn.functional as F


class neural_network(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation):
        super().__init__()
        self.activation = activation

        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False)]+1*[nn.Linear(hidden_dim, hidden_dim, bias=False)])
        output_layer = nn.Linear(hidden_dim, 1, bias=False)
        # output_layer.requires_grad_(False)
        self.layers.append(output_layer)  # output layer
        # for layer in self.layers:
        #     layer.weight = torch.nn.Parameter(layer.weight / 1)
        if activation == 'relu':
            self.activation = F.relu

    def forward(self, data):
        feats = data
        for layer in self.layers[:-1]:
            # print(layer)
            feats = layer(feats)
            feats = self.activation(feats)
        feats = self.layers[-1](feats)
        return feats

def create_model(args):
    model = neural_network(input_dim=args.input_dim, hidden_dim=args.model_hidden_dim, activation=args.model_activation)
    return model.cuda()