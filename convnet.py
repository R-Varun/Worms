import torch
from torch.autograd import Variable
import os
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch.optim as optim
import shutil

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage import  color

from PIL import Image


import torchvision.models as models

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def resize(image, width = 30, height = 30):
	r = image.resize((width,height))
	r = np.asarray(r)

	# r = r.flatten()
	return r


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

dir_path = os.path.dirname(os.path.realpath(__file__))
neg = os.path.join(dir_path, "neg_characters", "img")
pos = os.path.join(dir_path, "pos_characters", "img")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(7744, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = torch.nn.functional.dropout2d(x, p=.5)
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model = Net()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), .0001)

transform = transforms.Compose([transforms.Resize((100, 100)),
                                transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.ImageFolder(root="C:/Users/Varun/Worms/train", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root="./test", transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=True, num_workers=2)


criterion = nn.CrossEntropyLoss()

cuda = torch.cuda.is_available()

# cuda = False
if cuda:
    model = model.cuda()


def train(epoch, save=False):
    model.train()
    total_correct = 0
    train_loader = trainloader
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        total_correct += correct
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Acc: {:.2f}%/{:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0],
                       100. * correct / train_loader.batch_size,
                       100. * total_correct / ((batch_idx + 1) * train_loader.batch_size)))

    save_checkpoint({
        'epoch': epoch + 1,
        'arch': "ye",
        'state_dict': model.state_dict(),
        'best_prec1': 0,
        'optimizer': optimizer.state_dict(),
    }, True)
    print("SAVED")


global global_model
global_model = None
def loadModel():
    global global_model
    if global_model == None:
        if os.path.isfile('model_best.pth.tar'):
            checkpoint = torch.load('model_best.pth.tar')
            nm = Net()
            nm.load_state_dict(checkpoint['state_dict'])
            op = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
            op.load_state_dict(checkpoint['optimizer'])
            global_model = nm, op
            return nm, op
    else:
        return global_model


def test():
    model.eval()
    test_loss = 0
    correct = 0
    test_loader = testloader

    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += criterion(output, target).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    return test_loss, test_acc


imsize = 256
loader = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor(),  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def image_loader(image):
    """load image, returns cuda tensor"""
    # image = image.convert("RGB")
    image = color.gray2rgb(image)
    image = Image.fromarray(np.uint8(image*255))

    # plt.imshow(image)
    # plt.show()
    image = loader(image).float()

    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    # image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image  #assumes that you're using GPU


def predict(image):
    n, o = loadModel()
    test = image_loader(image)
    outputs = n(test)
    _, predicted = torch.max(outputs.data, 1)
    # print(_)
    return int(predicted[0])

def main():
    for i in range(200):
        train(i)

    # global model
    # global optimizer
    #
    # model, optimizer = loadModel()
    #
    test()

    pass



if __name__ == "__main__":
    main()