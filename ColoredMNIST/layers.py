import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import numpy as np

import torch.optim as optim

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
        return out


class PhiLayer(LayerBase):
    def __init__(self, hidden_dim: int):
        super(PhiLayer, self).__init__(28*28, 256, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(super().forward(x))


class FLayer(LayerBase):
    def __init__(self, hidden_dim: int):
        super(FLayer, self).__init__(256, 10, hidden_dim)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        return self.activation(super().forward(x))


class GLayer(LayerBase):
    def __init__(self, hidden_dim: int):
        super(GLayer, self).__init__(256, 10, hidden_dim)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        return self.activation(super().forward(x))


class Model(nn.Module):
    def __init__(self, hidden_dim: int):
        super(Model, self).__init__()
        self.phi = PhiLayer(hidden_dim)
        self.f = FLayer(hidden_dim)
        self.g = GLayer(hidden_dim)

    def forward(self, x0):
        x = self.phi.forward(x0)
        y = self.f.forward(x)
        z = self.g.forward(x)
        return torch.cat([y, z], axis=0)


# obtain MNIST
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data',
                                      train=True,
                                      download=True,
                                      transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=2)
testset = torchvision.datasets.MNIST(root='./data',
                                     train=False,
                                     download=True,
                                     transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=100,
                                         shuffle=False,
                                         num_workers=2)
classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

model = Model(ERMhidden_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 20

for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, torch.cat([labels, labels], axis=0))
        # backward
        loss.backward()
        # optimize
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            print('[{:d}, {:5d}] loss: {:.3f}'
                  .format(epoch+1, i+1, running_loss/100))
            running_loss = 0.0

print('Finished Training')
