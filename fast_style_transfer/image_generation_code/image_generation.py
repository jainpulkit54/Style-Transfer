import time
import re
import numpy as np
import csv
import torch
import cv2
import os
from torchvision import transforms
from transformer_net import TransformerNet
from matplotlib import pyplot as plt

model_path_1 = 'custom_saved_models/hot-spring.pth' # Specify the model path using which the style transfer needs to be carried out
stylized_image_path = '' # Specify the path where you need to store the stylized images

x_val = np.empty([1000,256,256,3], dtype = np.uint8()) # assuming that we style transfering 1000 images
for i in range(1000):
    img = cv2.imread('path to the image file')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # returns numpy image of shape 256 * 256 * 3
    x_val[i,:,:,:] = img

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
content_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
style_model_1 = TransformerNet()
state_dict_1 = torch.load(model_path_1)

# remove saved deprecated running_* keys in InstanceNorm from the checkpoint
for k in list(state_dict_1.keys()):
    if re.search(r'in\d+\.running_(mean|var)$', k):
        del state_dict_1[k]

folder_name1 = 'hot-spring-generated'

path1 = stylized_image_path + folder_name1
if(not os.path.exists(path1)):
    os.makedirs(path1)

style_model_1.load_state_dict(state_dict_1)
style_model_1.to(device)
    
with torch.no_grad():
    for i in range(val_examples):
        content_image = x_val[i,:,:,:]
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)
        
        if(i>=0 and i<=999):
            stylized_image_path = stylized_image_path + folder_name1 + '/Image' + str(i+1) + '.jpg'
            output = style_model_1(content_image).cpu()
            img = np.transpose(output[0],(1,2,0)).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        cv2.imwrite(stylized_image_path,img)
        stylized_image_path = ''
        


