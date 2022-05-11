# Define datasets and data loaders for teacher output dataset (will be useful for training student)
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
from torch.utils.data import Dataset, DataLoader

TEST_PATH = F"./teacher-data/teacher-21-layers-test.pkl"
TRAIN_PATH = F"./teacher-data/teacher-21-layers-train.pkl"

class TeacherTrainset(Dataset):

    def __init__(self, train_path):
      # data loading
      self.train_path = train_path
      with open(train_path, 'rb') as f:
        teacher_set = pickle.load(f)
        images = np.array([d[0] for d in teacher_set])
        middle_layer =  np.array([d[1] for d in teacher_set])
        gradient = np.array([d[2] for d in teacher_set])
        smoothgrad = np.array([d[3] for d in teacher_set])
        edge = np.array([d[4] for d in teacher_set])
        self.x = np.array([images, middle_layer, gradient, smoothgrad, edge])
        self.y = np.array([d[5] for d in teacher_set])
        self.n_samples = len(teacher_set)

    def __getitem__(self, index):
        return self.x[:,index], self.y[index]

    def __len__(self):
        return self.n_samples
    
class TeacherTestset(Dataset):

    def __init__(self):
      # data loading
      with open(TEST_PATH, 'rb') as f:
        teacher_set = pickle.load(f)
        self.x = np.array([d[0] for d in teacher_set])
        self.y = np.array([d[1] for d in teacher_set])
        self.n_samples = len(teacher_set)

    def __getitem__(self, index):
      return self.x[index], self.y[index]

    def __len__(self):
      return self.n_samples

trainset = TeacherTrainset(TRAIN_PATH)
trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True, num_workers=2)

testset = TeacherTestset()
testloader = DataLoader(dataset=testset, batch_size=4, shuffle=True, num_workers=2)

# Middle layer visualization
def imshow(img, path):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))#, cmap='gray', vmin=0, vmax=255)
    #plt.colorbar()
    #plt.show()
    plt.savefig(path)

NUM_EXAMPLES = 1
batch_size = 4
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(len(trainset))
print(len(trainloader))
print(len(testset))
print(len(testloader))
for _ in range(NUM_EXAMPLES):
  batch, labels = next(iter(trainloader))
  images = batch[:,0]
  #images = np.squeeze(images, axis=1)
  middle_layers = batch[:,1]
  gradients = batch[:,2]
  smoothgrads = batch[:,3]
  edges = batch[:,4]
  #explanations = np.squeeze(explanations, axis=1)
  print(labels)
  #preds = np.argmax(labels.cpu().numpy(), axis=1) <- this is for soft labels (probabilities) in the teacher set
  imshow(torchvision.utils.make_grid(images), 'images.pdf')
  imshow(torchvision.utils.make_grid(middle_layers), 'middle-layers.pdf')
  imshow(torchvision.utils.make_grid(gradients), 'gradients.pdf')
  imshow(torchvision.utils.make_grid(smoothgrads), 'smoothgrads.pdf')
  imshow(torchvision.utils.make_grid(edges), 'edges.pdf')
  #print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
  
