import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)

batch_size = 50

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./batches', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

validation_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./batches', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
 			transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=batch_size, shuffle=False, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc1_drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        return F.log_softmax(self.fc2(x))

model = Net()

if cuda:
    model.cuda()

def main():
    epochs = 5

    #learning_rate = .1#float(sys.argv[1])
    moment = .1
    opt = optim.SGD(model.parameters(), lr=.1, momentum=moment, weight_decay=.1)

    lossv, accv = [], []
    for epoch in range(1, epochs + 1):
        train(epoch, opt)
        validate(lossv, opt, accv)
    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1,epochs+1), lossv)
    plt.title('validation loss')
    plt.savefig("decayloss" + str(0.1) + ".png")

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1,epochs+1), accv)
    plt.title('validation accuracy')
    plt.savefig("decay" + str(0.1) + ".png")


def train(epoch, optimizer, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def validate(loss_vector, optimizer, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        val_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))

if __name__ == "__main__":
    main()
