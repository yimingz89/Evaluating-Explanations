import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
import pickle
TEST_PATH = F"./data/teacher-21-layers-test.pkl"
TRAIN_PATH = F"./data/teacher-21-layers-train.pkl"

random.seed(0)

train_set = []
with open(TRAIN_PATH, 'rb') as f:
    train_set = pickle.load(f)
for i in range(1,6,1):
    size = i * 1500
    SAMPLE_PATH = F"./data/teacher-21-layers-train-" + str(size) + ".pkl"
    sample_train_set = random.sample(train_set, i * 1500)
    print(len(sample_train_set))
    with open(SAMPLE_PATH, 'wb') as f:
        pickle.dump(sample_train_set, f)
    
