import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torchvision
import torchvision.transforms as transforms
import numpy as np

import torch.optim as optim

from typing import Optional

ERMhidden_dim = 64
epochs = 20
learning_rate = 0.001
momentum = 0.9
batch_size = 100


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
        super(PhiLayer, self).__init__(28*28, 32, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(super().forward(x))


class FLayer(LayerBase):
    def __init__(self, hidden_dim: int):
        super(FLayer, self).__init__(32, 10, hidden_dim)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        return self.activation(super().forward(x))


class GLayer(LayerBase):
    def __init__(self, hidden_dim: int):
        super(GLayer, self).__init__(32, 10, hidden_dim)
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
        return torch.cat([y, z], axis=1)


class TwoHotCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, 
                 reduce=None, reduction: str = 'mean') -> None:
        super(TwoHotCrossEntropyLoss, self).__init__(weight, size_average,
                                                     reduce, reduction)
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        e1 = F.cross_entropy(input[:, :10], target, weight=self.weight, 
                             ignore_index=-100, reduction=self.reduction)
        e2 = F.cross_entropy(input[:, 10:], target, weight=self.weight, 
                             ignore_index=-100, reduction=self.reduction)
        return e1 + e2


# Refered by https://qiita.com/fukuit/items/215ef75113d97560e599
# obtain MNIST
def downlaod_MNIST():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data',
                                          train=True,
                                          download=True,
                                          transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data',
                                         train=False,
                                         download=True,
                                         transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=2)
    classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

    return trainloader, testloader


def learning(model, trainloader):
    criterion = TwoHotCrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs.reshape((batch_size, 28*28)))
            loss = criterion(outputs, labels)
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


def test(model, testloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for (images, labels) in testloader:
            outputs = model(images.reshape(-1,28*28))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted%10 == labels).sum().item()

    print('Accuracy: {:2f} %%'.format(100*float(correct/total)))


def main():
    model = Model (ERMhidden_dim)
    trainloader, testloader = downlaod_MNIST()
    learning(model, trainloader)

    test(model, testloader)


if __name__ == '__main__':
    main()
