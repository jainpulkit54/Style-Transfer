import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

dataset_path = '' # Specify the path to the test dataset here
transformations = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

complete_data = datasets.ImageFolder(dataset_path,transformations)
complete_data_loader = DataLoader(complete_data, batch_size = 1, shuffle = True)

x_test = np.empty([400,3,224,224])
y_test = np.empty([400,1])

for index, (xx, yy) in enumerate(complete_data_loader):
    x_test[index,:,:,:] = xx[0].numpy()
    y_test[index,0] = yy.numpy()

# Code for converting the numpy array into Tensors
y_test = torch.from_numpy(y_test)
y_test = y_test.long()
y_test = y_test.squeeze_()

print('The Shape of the Test Set is:',x_test.shape)
test_examples = x_test.shape[0]

net = models.resnet18(pretrained = False)
for param in net.parameters():
    param.requires_grad = False
num_ftrs = net.fc.in_features
num_classes = 4
net.fc = nn.Linear(num_ftrs, num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('resnet18parameters.pth') # loading the saved parameters
parameters = checkpoint['state_dict'] # contains the learned parameters
epoch = checkpoint['Epochs'] # contains the information regarding the number of epochs
arch = checkpoint['Architecture'] # contains the information regarding the architecture
net.load_state_dict(parameters)
net.to(device)

# Code for evaluating the test accuracy
test_correct = 0
inputs_test = torch.from_numpy(x_test).type(torch.FloatTensor)
labels_test = y_test
inputs_test = inputs_test.to(device)
labels_test = labels_test.to(device)
net.eval()
with torch.no_grad():
    predictions_test = net(inputs_test)
    predictions_test_class = torch.argmax(predictions_test,dim = 1)
    for j in range(test_examples):
        if(predictions_test_class[j] == labels_test[j]):
            test_correct = test_correct + 1

test_accuracy = (test_correct/test_examples)*100
print('The Accuracy over the Test Set is:', test_accuracy)
