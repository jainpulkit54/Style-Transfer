from __future__ import print_function

import os
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lib.utils import as_numpy
from lib.nn import user_scattered_collate, async_copy_to

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as transforms
import torchvision.models as models

import test
from yacs.config import CfgNode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_id=0

total_steps=int(sys.argv[1])
style_loss_weight=float(sys.argv[2])
content_loss_weight=float(sys.argv[3])
seg_loss_weight=float(sys.argv[4])
style_img_name = sys.argv[5]
content_img_name = sys.argv[6]
if style_img_name==content_img_name: sys.exit()

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    image = image.resize((imsize,imsize))
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001) # pause a bit so that plots are updated

def imsave(tensor, filepath):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imsave(filepath, np.array(image))	
    
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
    
class SegLoss(nn.Module):
    def __init__(self, target_seg):
        super(SegLoss, self).__init__()
        self.target = target_seg

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return self.loss

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# create a module to normalize input image so we can easily put it in a
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_seg_model_loss(segmentation_module, gpu, content_img):
    segSize = (as_numpy(content_img.squeeze(0).cpu()).shape[0], as_numpy(content_img.squeeze(0).cpu()).shape[1])
    feed_dict = {'img_data': content_img.clone()}
    feed_dict = async_copy_to(feed_dict, gpu)
    target_seg = segmentation_module(feed_dict, segSize=segSize)
    seg_loss = SegLoss(target_seg)
    return seg_loss

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

#cfg file required for segmentation network
def get_cfg():
	return CfgNode({'DIR': 'ckpt/ade20k-resnet50dilated-ppm_deepsup', 'DATASET': CfgNode({'root_dataset': './data/', 'list_train': './data/training.odgt', 'list_val': './data/validation.odgt', 'num_class': 150, 'imgSizes': (300, 375, 450, 525, 600), 'imgMaxSize': 1000, 'padding_constant': 8, 'segm_downsampling_rate': 8, 'random_flip': True}), 'MODEL': CfgNode({'arch_encoder': 'resnet50dilated', 'arch_decoder': 'ppm_deepsup', 'weights_encoder': 'ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth', 'weights_decoder': 'ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth', 'fc_dim': 2048}), 'TRAIN': CfgNode({'batch_size_per_gpu': 2, 'num_epoch': 20, 'start_epoch': 0, 'epoch_iters': 5000, 'optim': 'SGD', 'lr_encoder': 0.02, 'lr_decoder': 0.02, 'lr_pow': 0.9, 'beta1': 0.9, 'weight_decay': 0.0001, 'deep_sup_scale': 0.4, 'fix_bn': False, 'workers': 16, 'disp_iter': 20, 'seed': 304}), 'VAL': CfgNode({'batch_size': 1, 'visualize': False, 'checkpoint': 'epoch_20.pth'}), 'TEST': CfgNode({'batch_size': 1, 'checkpoint': 'epoch_20.pth', 'result': 'segmented/'}), 'list_test': [{'fpath_img': ''}]})

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=total_steps,
                       style_weight=style_loss_weight, content_weight=content_loss_weight, seg_weight=seg_loss_weight):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,normalization_mean, normalization_std, style_img, content_img)
    seg_loss = get_seg_model_loss(segmentation_module, gpu_id, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    if seg_weight!=0: loss_vs_run={'style':[], 'content':[], 'segmentation':[]}
    else: loss_vs_run={'style':[], 'content':[]}

    run = [0]
    while run[0] <= num_steps:
        def closure():
        # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            seg_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            style_score *= style_weight
            content_score *= content_weight
            
            if seg_weight!=0:
                #get seg score
                segSize = (as_numpy(input_img.squeeze(0).cpu()).shape[0], as_numpy(input_img.squeeze(0).cpu()).shape[1])
                feed_dict = {'img_data': input_img}
                feed_dict = async_copy_to(feed_dict, gpu)
                input_seg = segmentation_module(feed_dict, segSize=segSize)
                seg_score = seg_loss.forward(input_seg)
                seg_score *= seg_weight

            if seg_weight!=0: loss = style_score + content_score + seg_score
            else: loss = style_score + content_score

            loss.backward(retain_graph=True)
            
            loss_vs_run['style'].append(style_score.item())
            loss_vs_run['content'].append(content_score.item())
            if seg_weight!=0: loss_vs_run['segmentation'].append(seg_score.item())
			
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                if seg_weight!=0:
                    print('Style Loss : {:4f} Content Loss: {:4f} Segmentation Loss: {:4f}'.format(
                        style_score.item(), content_score.item(), seg_score.item() ))
                else:
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                
                print()
                plt.clf()
                imshow(input_img, title='Output Image')
                plt.savefig(img_savepath+'transferred/%d.png'%int(run[0]/10))

            run[0] += 1

            return style_score + content_score + seg_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img, loss_vs_run

#paths to images
style_img_path = '../images/'+style_img_name+'.jpg'
content_img_path = '../images/'+content_img_name+'.jpg'
style_img = image_loader(style_img_path)
content_img = image_loader(content_img_path)

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"
    
unloader = transforms.ToPILImage()  # reconvert into PIL image

#output path
if seg_loss_weight!=0: img_savepath='st_outputs/with_segloss/{:s}_{:s}/expt_{:d}_steps_{:1.0e}_styleweight_{:1.0e}_segweight/'.format(style_img_name, content_img_name, total_steps, style_loss_weight, seg_loss_weight)
else: img_savepath='st_outputs/without_segloss/{:s}_{:s}/expt_{:d}_steps_{:1.0e}_weight/'.format(style_img_name, content_img_name, total_steps, style_loss_weight)

if not os.path.exists(img_savepath):
    os.makedirs(img_savepath+'transferred')

#vgg-19 network
cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

input_img = content_img.clone()

#load segmentation network
cfg=get_cfg()
segmentation_module, loader, gpu = test.main(cfg, gpu_id)
segmentation_module.eval()
output, loss_vs_run = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

#saving figures
plt.figure()
imshow(style_img, title='Style Image')
plt.savefig(img_savepath+'style.png')

plt.figure()
imshow(content_img, title='Content Image')
plt.savefig(img_savepath+'content.png')

plt.figure()
for item in loss_vs_run.keys():
    plt.plot(range(len(loss_vs_run[item])), np.array(loss_vs_run[item])/max(loss_vs_run[item]), label=item)
plt.legend()
plt.savefig(img_savepath+'Loss_contours.png')
