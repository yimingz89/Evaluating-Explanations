# collect and save teacher output data set
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pickle
import scipy
from scipy import ndimage
PATH = F"./models/teacher_net.pth"
TEST_PATH = F"../Student/data/teacher-21-layers-test.pkl"
TRAIN_PATH = F"../Student/data/teacher-21-layers-train.pkl"

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

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Grayscale(),
     transforms.Normalize((0.5,), (0.5,))])
batch_size = 4

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net(1, 10, 16)
net.to(device)
net.load_state_dict(torch.load(PATH))

# collect teacher prediction test set
correct = 0
total = 0
teacher_testset = []
teacher_trainset = []

# softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# visualize
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))#, cmap='gray', vmin=0, vmax=255)
    #plt.colorbar()
    plt.show()

# edge detector (sobel operator)
def edge_detector(img):
    im = img.detach().cpu().numpy()
    edges = np.zeros(im.shape)
    b, _, W, H = im.shape
    for i in range(b):
        dx = ndimage.sobel(np.squeeze(im[i]), 0) # horizontal change
        dy = ndimage.sobel(np.squeeze(im[i]), 1) # vertical change
        m = np.hypot(dx, dy) # magnitude
        m = m.reshape((1,W,H))
        edges[i] = m
        #m *= 255.0 / np.max(m) # normalize
    return edges

# vanilla gradient
def saliency_grad(net, img):
    with torch.enable_grad():
        img = img.to(device)
        img.requires_grad = True
        logits = net(img)['final']
        labels = np.argmax(logits.detach().cpu().numpy(), axis=1)
        label_logits = logits[np.arange(len(logits)), labels]
        label_logits.sum().backward()
        #for i in range(len(label_logits)):
            #label_logits[i].backward(retain_graph=True)
        grads = img.grad.detach().cpu().numpy()
        return grads, labels

# saliency smooth grad
def saliency_smooth_grad(net, img, N_repeat=50, noise_level=0.15):
    B, _, H, W = img.shape
    with torch.no_grad():
        img = img.to(device)
        logits = net(img)['final']
        labels = np.argmax(logits.detach().cpu().numpy(), axis=1)
        label_logits = logits[np.arange(len(logits)), labels]
    with torch.enable_grad():
        sigma = noise_level * (img.max() - img.min())
        noises = torch.randn(B * N_repeat, 1, H, W).to(device) * sigma #50x1x32x32
        imgs = (torch.cat([img] * N_repeat, dim=0) + noises).to(device) # 200x1x32x32
        imgs.requires_grad = True
        logits = net(imgs)['final'] #200x10
        label_logits = logits[:, labels] #200x4
        label_logits.sum().backward()
        grads = imgs.grad.detach().cpu().numpy() # 200x1x32x32
        mean_grads = np.zeros((B, 1, H, W))
        for i in range(B):
            grad = grads[np.arange(grads.shape[0] // B) * B + i].mean(axis=0)
            mean_grads[i] = grad
        return mean_grads, labels

# Grad-CAM
def saliency_gradcam(net, img):
    B, _, H, W = img.shape
    img = img.to(device)
    with torch.no_grad():
        logits = net(img)['final']
        labels = np.argmax(logits.detach().cpu().numpy(), axis=1)
        conv_maps = net.maxpool(net.relu(net.bn1(net.conv1(img))))
    return img, labels
# LIME
def saliency_lime(net, img):
    pass

# SHAP
def saliency_shap(net, img):
    pass

# get full teacher train set for student
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        images, labels = data[0].to(device), data[1].to(device) # images will be used for gradient computation for saliency_grad
        gradcam, labels = saliency_gradcam(net, images)
        edges = edge_detector(images)
        images_copy = torch.clone(data[0]).detach().to(device) # copy images before sending to device for gradient computations for saliency_smooth_grad 
        grads, labels = saliency_grad(net, images)
        smooth_grads, labels = saliency_smooth_grad(net, images_copy)
        outputs = net(images)['final']
        probabilities = np.zeros((batch_size, len(classes)))
        for j in range(batch_size):
          probabilities[j] = softmax(outputs.cpu().numpy()[j])
        _, predicted = torch.max(outputs.data, 1)

        if i < 0.75 * len(testloader):    
          middle = net(images)['middle']
          for i in range(batch_size):
            teacher_trainset.append((images[i].cpu().numpy(), middle[i].cpu().numpy(), grads[i], smooth_grads[i], edges[i], probabilities[i]))
        else:
          for i in range(batch_size):
            teacher_testset.append((images[i].cpu().numpy(), probabilities[i])) # no explanation at test time

with open(TRAIN_PATH, 'wb') as f:
   pickle.dump(teacher_trainset, f)

with open(TEST_PATH, 'wb') as f:
   pickle.dump(teacher_testset, f)

