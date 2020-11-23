import torch
import torch.nn as nn
import torch.nn.functional as F

ERMhidden_dim = 256

class LayerBase(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super(LayerBase, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        lin1 = nn.Linear(input_dim, hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        lin3 = nn.Linear(hidden_dim, output_dim)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True),
                                   lin2, nn.ReLU(True),
                                   lin3)

    def forward(self, x):
        out = self._main(x)
        out = out.reshape(-1, self.output_dim)
        m = nn.Softmax(dim=1)       # require?
        return m(out)


class PhiLayer(LayerBase):
    def __init__(self, hidden_dim: int):
        super(PhiLayer, self).__init__(28*28*2, 10, hidden_dim)


class FLayer(LayerBase):
    def __init__(self, hidden_dim: int):
        super(FLayer, self).__init__(10, 10, hidden_dim)


class GLayer(LayerBase):
    def __init__(self, hidden_dim: int):
        super(GLayer, self).__init__(10, 10, hidden_dim)


class Model(nn.Module):
    def __init__(self, hidden_dim: int):
        super(Model, self).__init__()
        self.phi = PhiLayer(hidden_dim)
        self.f = FLayer(hidden_dim)
        self.g = GLayer(hidden_dim)
        
    def forward(self, x):
        x = self.phi.forward(x)
        y = self.f.forward(x)
        z = self.g.forward(x)
        return torch.cat([y,z], axis=0)

model = Model(ERMhidden_dim)
print (model)


sample_input = torch.randn(1,28*28*2)
outputs = model(sample_input)
print (outputs)
