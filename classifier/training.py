import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

dataset_path = '' # Specify the dataset path here and since we are using the Image Folder, the images should be present in folder containing subfolders belonging to each
transformations = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

complete_data = datasets.ImageFolder(dataset_path,transformations)
complete_data_loader = DataLoader(complete_data, batch_size = 1, shuffle = True)

# The below code is assuming that the dataset contains a total of 24000 training images

x_train = np.empty([20000,3,224,224])
y_train = np.empty([20000,1])
x_val = np.empty([4000,3,224,224])
y_val = np.empty([4000,1])

for index, (xx, yy) in enumerate(complete_data_loader):
    if(index < 20000):
        x_train[index,:,:,:] = xx[0].numpy()
        y_train[index,0] = yy.numpy()
    elif(index >= 20000 and index < 24000):
        x_val[(index-20000),:,:,:] = xx[0].numpy()
        y_val[(index-20000),0] = yy.numpy()

# Code for converting the numpy array into Tensors
y_train = torch.from_numpy(y_train)
y_train = y_train.long()
y_train = y_train.squeeze_()

y_val = torch.from_numpy(y_val)
y_val = y_val.long()
y_val = y_val.squeeze_()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('The number of target labels available for Training Set are:',y_train.shape)
print('The number of target labels available for Validation Set are:',y_val.shape)
print('The device to be used for training the Network is:',device)

print('The Shape of the Training Dataset is:',x_train.shape)
print('The Shape of the Validation Dataset is:',x_val.shape)

net = models.resnet18(pretrained = True)
for param in net.parameters():
    param.requires_grad = False
num_ftrs = net.fc.in_features
num_classes = 4
net.fc = nn.Linear(num_ftrs, num_classes)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.fc.parameters(), betas = (0.9,0.999), eps = 1e-8, weight_decay = 1e-3)

epochs = 25
batch_size = 50
epoch_loss = np.empty([epochs,1])
train_accuracy = np.empty([epochs,1])
validation_accuracy = np.empty([epochs,1])
training_examples = x_train.shape[0]
val_examples = x_val.shape[0]
no_of_batches = training_examples/batch_size

for epoch in range(epochs):
    net.train() # Setting up the network in the training mode
    running_loss = 0.0
    for i in range(0,training_examples,batch_size):
        temp = x_train[i:(i + batch_size),:,:,:].reshape(batch_size,3,224,224)
        inputs_train = torch.from_numpy(temp).type(torch.FloatTensor)
        labels_train = y_train[i:(i + batch_size)]
        inputs_train = inputs_train.to(device)
        labels_train = labels_train.to(device)

        optimizer.zero_grad()
        predictions_train = net(inputs_train)
        loss = criterion(predictions_train,labels_train)
        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.item()
        
    epoch_loss[epoch,0] = (running_loss/no_of_batches)
    
    net.eval() # Setting up the network in the evaluation mode    
    
    # Code for evaluating the train accuracy
    train_correct = 0
    for i in range(0,training_examples,batch_size):
        temp = x_train[i:(i + batch_size),:,:,:].reshape(batch_size,3,224,224)
        inputs_train = torch.from_numpy(temp).type(torch.FloatTensor)
        labels_train = y_train[i:(i + batch_size)]
        inputs_train = inputs_train.to(device)
        labels_train = labels_train.to(device)
            
        with torch.no_grad():
            predictions_train = net(inputs_train)
            predictions_train_class = torch.argmax(predictions_train,dim = 1)
            for j in range(batch_size):
                if(predictions_train_class[j] == labels_train[j]):
                    train_correct = train_correct + 1
    
    train_accuracy[epoch,0] = (train_correct/training_examples)*100
    
    # Code for evaluating the validation accuracy
    validation_correct = 0
    for i in range(0,val_examples,batch_size):
        temp = x_val[i:(i + batch_size),:,:,:].reshape(batch_size,3,224,224)
        inputs_val = torch.from_numpy(temp).type(torch.FloatTensor)
        labels_val = y_val[i:(i + batch_size)]
        inputs_val = inputs_val.to(device)
        labels_val = labels_val.to(device)
        
        with torch.no_grad():
            predictions_val = net(inputs_val)
            predictions_val_class = torch.argmax(predictions_val,dim = 1)
            for j in range(batch_size):
                if(predictions_val_class[j] == labels_val[j]):
                    validation_correct = validation_correct + 1
        
    validation_accuracy[epoch,0] = (validation_correct/val_examples)*100
            
    print("Loss in",(epoch + 1),"/",epochs,"epochs is:",epoch_loss[epoch,0],",Train Accuracy:",train_accuracy[epoch,0],",Validation accuracy:",validation_accuracy[epoch,0])

print("Finished Training")

plt.plot(epoch_loss)
plt.title('Loss Function')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(train_accuracy, label = 'Train Accuracy')
plt.plot(validation_accuracy, label = 'Validation Accuracy')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.show()

# Code for saving the learned model parameters
net.cpu()
torch.save({'Architecture': 'resnet18',
            'Epochs': epochs,
           'state_dict': net.state_dict()},'resnet18parameters.pth')
