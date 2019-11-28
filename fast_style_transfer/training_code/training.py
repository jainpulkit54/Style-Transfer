import time
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
from vgg import Vgg16
from matplotlib import pyplot as plt

dataset_path = '' # Specify the dataset path here and since we are using the image folder so the images should be in folder with subfolder representing the various classes
epochs = 2
batch_size = 4

# Specify the style image to be used for training. We have basically used five different images for style and a separate model needs to be trained for each of the style

#style_image = 'candy.jpg'
style_image = 'skyscraper.jpg'
#style_image = 'ocean.jpg'
#style_image = 'hot_spring.jpg'
#style_image = 'desert_sand.jpg'
save_model_dir = 'custom_saved_models/' # The directory for storing the trained models
content_weight = 1e5 
style_weight = 1e10
learning_rate = 1e-3
log_interval = 250 # interval after which log the total loss obtained i.e.,
# after 2000 samples of dataset when batch size is 4
image_size = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
torch.manual_seed(42)
transform = transforms.Compose([transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x.mul(255))])

train_dataset = datasets.ImageFolder(dataset_path,transform)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
transformer = TransformerNet().to(device) # this is the network that will require training
optimizer = Adam(transformer.parameters(), learning_rate)
mse_loss = torch.nn.MSELoss()

vgg = Vgg16(requires_grad = False).to(device) # this network is used to obtain the feature maps
style_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Lambda(lambda x: x.mul(255))])

style = utils.load_image(style_image, size = None)
# The size for 'mosaic.jpg' style image is 1024 x 1024 x 3
style = style_transform(style) # the shape will be [3, style_image_size, style_image_size]
style = style.repeat(batch_size, 1, 1, 1).to(device) # the size here will be [batch_size, 3, style_image_size, style_image_size]

features_style = vgg(utils.normalize_batch(style))
gram_style = [utils.gram_matrix(y) for y in features_style]\
# The 'gram_style' will be a list containing gram matrices formed at different layers i.e., from 1st four layers:
# gram_style[0] will be formed using feature maps from 'relu1_2' and will be of dimension [batch_size, 64, 64]
# gram_style[1] will be formed using feature maps from 'relu2_2' and will be of dimension [batch_size, 128, 128]
# gram_style[2] will be formed using feature maps from 'relu3_2' and will be of dimension [batch_size, 256, 256]
# gram_style[3] will be formed using feature maps from 'relu4_3' and will be of dimension [batch_size, 512, 512]

# The code for training the Transformer Network goes as follows:

for epoch in range(epochs):
    transformer.train()
    agg_content_loss = 0
    agg_style_loss = 0
    count = 0
    # Now in the following loop, we will iterate over each batch of the given dataset
    # 'batch_id' variable contains the batch number
    # 'x' variable contains the data sample i.e. tensor in this case of size (batch_size, 3, image_size, image_size)
    # 'label' variable contains the labels corresponding to the data samples fetched 
    for batch_id, (x,label) in enumerate(train_loader):
        n_batch = len(x) # this will evaluate to batch_size value
        count = count + n_batch
        optimizer.zero_grad()
        
        x = x.to(device)
        y = transformer(x)
        y = utils.normalize_batch(y)
        x = utils.normalize_batch(x)
        
        features_y = vgg(y)
        features_x = vgg(x)
        
        # Can use features from some different layers:
        # The original paper mentions to take features from the "relu3_3" layer whereas we are taking it from
        # "relu2_2" layer
        content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)
        
        style_loss = 0
        
        for ft_y, gm_s in zip(features_y, gram_style):
            gm_y = utils.gram_matrix(ft_y)
            style_loss = style_loss + mse_loss(gm_y, gm_s[:n_batch,:,:])
        
        style_loss = style_loss * style_weight
        
        total_loss = content_loss + style_loss
        total_loss.backward()
        optimizer.step()
        
        agg_content_loss = agg_content_loss + content_loss.item()
        agg_style_loss = agg_style_loss + style_loss.item()
        
        if((batch_id + 1) % log_interval == 0):
            message = '{}\tEpoch {}:\t[{}/{}]\tContent Loss: {:.2f}\tStyle Loss: {:.2f}\tTotal Loss: {:.2f}'.format(
                time.ctime(), epoch + 1, count, len(train_dataset), agg_content_loss / (batch_id + 1),
                agg_style_loss / (batch_id + 1), (agg_content_loss + agg_style_loss) / (batch_id + 1))
            
            print(message)
            

transformer.eval().cpu()
#weights_filename = 'candy.pth'
weights_filename = 'skyscraper_single.pth'
#weights_filename = 'ocean_single.pth'
#weights_filename = 'hot-spring.pth'
#weights_filename = 'desert-sand.pth'
save_model_dir = save_model_dir + weights_filename
torch.save(transformer.state_dict(), save_model_dir)

