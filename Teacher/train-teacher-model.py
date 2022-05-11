# teacher model

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Grayscale(),
     transforms.Normalize((0.5,), (0.5,))])

batch_size = 4

# load data

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

# define neural network architecture

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'), 
              #nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

def add_conv_blocks(num_layers):
  layers = []
  for i in range(num_layers):
        layers.append(conv_block(64, 64))
  return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self, in_channels, num_classes, num_non_pool_layers): # 1x32x32 input
        super().__init__()
        self.conv1 = conv_block(in_channels, 64) # layer 1
        num_first_half_layers = (num_non_pool_layers+5) // 2
        self.res1 = add_conv_blocks(num_first_half_layers-1) # layers 2, 3, ... 10
        self.conv2 = nn.Sequential(conv_block(64, 1)) # layer 11
        self.conv3 = nn.Sequential(conv_block(1, 64)) # layer 12
        self.res2 = add_conv_blocks(num_non_pool_layers - num_first_half_layers - 2) # layers 13, 14, 15, 16
        self.conv4 = conv_block(64, 128, pool=True) # layer 17
        self.conv5 = conv_block(128, 256, pool=True) # layer 18
        self.conv6 = conv_block(256, 512, pool=True) # layer 19
        self.conv7 = conv_block(512, 512) # layer 20
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(512, num_classes)) # layer 21

    def forward(self, x):
        outputs = {}
        out = self.conv1(x)
        out = self.res1(out) + out
        mid = self.conv2(out)
        outputs['middle'] = mid # middle layer output
        out = self.conv3(mid)
        out = self.res2(out) + out
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.classifier(out)
        outputs['final'] = out # final layer output
        return outputs

net = Net(1, 10, 16)
net.to(device)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# train
num_epochs = 8
loss_curve = []
acc_curve = []
for epoch in range(num_epochs):  # loop over the dataset multiple times
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
      # get the inputs; data is a list of [inputs, labels]
      #inputs, labels = data
      inputs, labels = data[0].to(device), data[1].to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)['final']
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      if i % 2000 == 1999:    # print every 2000 mini-batches
          loss_curve.append( running_loss / 2000)
          print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
          running_loss = 0.0
          # evaluate accuracy on test set
          net.eval()
          
          correct = 0
          total = 0
          with torch.no_grad():
            for data in testloader:
              images, labels = data[0].to(device), data[1].to(device)
              outputs = net(images)['final']
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()

          acc = (100 * correct / total)
          acc_curve.append(acc)
          print('Accuracy of the network on the 10000 test images: %d %%' % acc)

          net.train()

LOSS_PLOT_PATH = F"./results/loss-plot.png"
ACC_PLOT_PATH = F"./results/accuracy-plot.png"

loss_curve = np.array(loss_curve)
acc_curve = np.array(acc_curve)

# plot and save loss curve while training
plt.figure()
plt.plot(np.arange(len(loss_curve)), loss_curve)
plt.xlabel('time')
plt.ylabel('loss')
plt.savefig(LOSS_PLOT_PATH)
plt.close()

# plot and save accuracy curve while training
plt.figure()
plt.plot(np.arange(len(acc_curve)), acc_curve)
plt.xlabel('time')
plt.ylabel('accuracy')
plt.savefig(ACC_PLOT_PATH)
plt.close()

print('Finished training model:')

# evaluate
net.eval()
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
  for data in testloader:
      images, labels = data[0].to(device), data[1].to(device)

        
      # calculate outputs by running images through the network 
      outputs = net(images)['final']
      # the class with the highest energy is what we choose as prediction
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

acc = (100 * correct / total)
print('Accuracy of the network on the 10000 test images: %d %%' % acc)
print('Finished evaluating model')




# saving model
#LOCAL_PATH = './cifar_net.pth'
#torch.save(net.state_dict(), LOCAL_PATH)
PATH = F"./models/teacher_net.pth"
torch.save(net.state_dict(), PATH)
ACC_PATH = F"./results/accuracy.txt"
acc_file = open(ACC_PATH, "a")
acc_file.write("accuracy for model with 21 layers: %d %%" % acc)
acc_file.write("\n")
acc_file.close()
