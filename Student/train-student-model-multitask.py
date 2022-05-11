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
import argparse
from torch.utils.data import Dataset, DataLoader

TEST_PATH = F"./data/teacher-21-layers-test.pkl"
TRAIN_PATH = F"./data/teacher-21-layers-train.pkl"

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
        edge_detector = np.array([d[4] for d in teacher_set])
        self.x = np.array([images, middle_layer, gradient, smoothgrad, edge_detector])
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

#trainset = TeacherTrainset(TRAIN_PATH)
#trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True, num_workers=2)

testset = TeacherTestset()
testloader = DataLoader(dataset=testset, batch_size=4, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
    def __init__(self, in_channels=1, num_classes=10, num_non_pool_layers=16, num_multitask_layers=5): # 1x32x32 input
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
        self.resm = add_conv_blocks(num_multitask_layers)
        self.convm = nn.Sequential(conv_block(64, 1))
        #self.convm1 = conv_block(512, 256) # multitask layer 1
        #self.convm2 = conv_block(256, 128) # multitask layer 2
        #self.convm3 = conv_block(128, 64) # multitask layer 3
        #self.resm1 = add_conv_blocks(num_multitask_layers-4) # multitask layer 4,5,6,7
        #self.convm4 = conv_block(64, 1) # multitask layer 8

    def forward(self, x):
        outputs = {}
        if len(x.shape) == 5: # training phase (4x4x1x32x32), i.e. an extra explanation (i.e. teacher middle layer) is passed in with the input.
            input = x[:,0]
            outputs['middle-layer'] = x[:,1]
            outputs['gradient'] = x[:,2]
            outputs['smoothgrad'] = x[:,3]
            outputs['edge-detector'] = x[:,4]
            # ... add other explanation methods as we expand dataset
        elif len(x) == 4: # test phase (4x1x32x32)
            input = x

        input = input.type(torch.cuda.FloatTensor) # use float32 tensor
        # predict class
        out = self.conv1(input)
        out = self.res1(out) + out
        mid = self.conv2(out)
        outputs['middle'] = mid # middle layer output
        out = self.conv3(mid)
        last_same_dim_layer = self.res2(out) + out
        out = self.conv4(last_same_dim_layer)
        out = self.conv5(out)
        out = self.conv6(out)
        last_layer = self.conv7(out)
        out = self.classifier(last_layer)
        outputs['final'] = out # final layer output

        # predict explanation
        outm = self.resm(last_same_dim_layer)
        outm = self.convm(outm)
        #outm = self.convm1(last_layer)
        #outm = self.convm2(outm)
        #outm = self.convm3(outm)
        #outm = self.resm1(outm) + outm
        #outm = self.convm4(outm)
        outputs['predicted explanation'] = outm # predicted explanation

        return outputs
  
# define loss function and optimizer

# lam = 0.001 # hyperparameter for penalizing middle layer being far from explanation, want to do a search for a good value for this (i.e. one which maximizes accuracy)
# criterion = nn.CrossEntropyLoss()

def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def student_loss(output, target, explanation_type='middle-layer', explanation_use='multitask', lam=1e-4):
    final = output['final']
    explanation = output[explanation_type]
    if explanation_use == 'direct-supervision': # only works for middle layer explanation
         middle = output['middle']
         loss = cross_entropy(final, target)
         loss = loss + lam *  torch.dist(explanation, middle).pow(2)
         return loss
    elif explanation_use == 'multitask': # all other multi-task based explanations, e.g. smoothgrad
         predicted_explanation = output['predicted explanation']
         loss = cross_entropy(final, target)
         #print(str(loss) + ', ' + str(torch.dist(explanation, predicted_explanation)))
         loss = loss + lam * torch.dist(explanation, predicted_explanation)
         return loss
    else:
         raise ValueError('Invalid explanation type given')

#criterion = nn.CrossEntropyLoss()

TOTAL_TRAIN_SIZE = 7500
NUM_BLOCKS = 5

def train_model(explanation):
    for sz in range(1,NUM_BLOCKS+1,1):
        if sz != 3:
            continue
        size = (TOTAL_TRAIN_SIZE // NUM_BLOCKS) * sz
        print("train size: " + str(size))
        train_path = "./data/teacher-21-layers-train-" + str(size) + ".pkl"
        print(train_path)
        trainset = TeacherTrainset(train_path)
        trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True, num_workers=2)
        for j in range(7,10,1):
          for i in range(-1,3,2): #(-10,3,2)
            net = Net(in_channels=1, num_classes=10, num_non_pool_layers=16)
            optimizer = optim.Adam(net.parameters(), lr=0.0001)
            net.to(device)
          
            # train

            if i == -10: # baseline
              lam = 0
            else:
              lam = 0.01 * (10 ** i)
            print('lam: ' + str(lam))

            num_epochs = 10
            loss_curve = []
            acc_curve = []

            print('trainloader len: ' + str(len(trainloader)))
            for epoch in range(num_epochs):  # loop over the dataset multiple times
              running_loss = 0.0    
              for i, data in enumerate(trainloader):
                  raise Error('stop here')
                  # get the inputs; data is a list of [inputs, labels]
                  inputs, teacher_predictions = data[0].to(device), data[1].to(device)
                  # zero the parameter gradients
                  optimizer.zero_grad()
                  # forward + backward + optimize
                  student_outputs = net(inputs)
                  loss = student_loss(student_outputs, teacher_predictions, explanation_type=explanation, explanation_use='multitask', lam=lam)
                  loss.backward()
                  optimizer.step()

                  # print statistics
                  running_loss += loss.item()
                  if i % 375 == 374:    # print every 375 mini-batches
                      loss_curve.append( running_loss / 375)
                      print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, i + 1, running_loss / 375))
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
                          labels = np.argmax(labels.cpu().numpy(), axis=1)
                          correct += (predicted.cpu().numpy() == labels).sum().item()

                      acc = (100 * correct / total)
                      acc_curve.append(acc)
                      print('Accuracy of the network on the 2500 test images: %d %%' % acc)

                      net.train()

            LOSS_DATA_PATH = F"./results/" + explanation + "/" + str(size) + "/run-" + str(j) + "-loss-curve-" + str(lam) + ".npy"
            ACC_DATA_PATH = F"./results/" + explanation + "/" + str(size) + "/run-" + str(j) + "-accuracy-curve-" + str(lam) + ".npy"
            loss_curve = np.array(loss_curve)
            acc_curve = np.array(acc_curve)
            np.save(LOSS_DATA_PATH, loss_curve)
            np.save(ACC_DATA_PATH, acc_curve)

            print('Finished training model with lambda: ' + str(lam))

            # since we're not training, we don't need to calculate the gradients for our outputs
            net.eval() # set to evaluation mode
            correct = 0
            total = 0
            with torch.no_grad():
              for data in testloader:
                  images, labels = data[0].to(device), data[1].to(device)
                
                  # calculate outputs by running images through the network 
                  outputs = net(images)['final']
                  # the class with the highest energy is what we choose as prediction
                  _, predicted = torch.max(outputs.data, 1)
                  total += labels.size(0)
                  labels = np.argmax(labels.cpu().numpy(), axis=1)
                  correct += (predicted.cpu().numpy() == labels).sum().item()

            acc = (100 * correct / total)
            print('Accuracy of the network on the test images: %d %%' % acc)
            print('Finished evaluating model')

            # saving model
            #PATH = F"./models/" + str(explanation) + "/" + str(size) + "/run-" + str(j) + "-student-21-layers-" + str(lam)
            #torch.save(net.state_dict(), PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse explanation type")
    parser.add_argument('explanation_type', type=str, help='A required string argument for the explanation type used to supervise the student training')
    args = parser.parse_args()
    explanation = args.explanation_type
    train_model(explanation)
