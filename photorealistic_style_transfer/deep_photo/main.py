import torch
import torchvision.models as models

from image_preprocessing import image_loader, masks_loader, plt_images
from neural_style import run_style_transfer
from torchvision.utils import save_image

idx = 1
path = 'data/'

style_image = './data/night.jpg'
# content_image = './data/skyscraper.jpg'
content_image = './data/desert_sand.jpg'
output_image = 'photorealistic_output.jpg'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imsize = (256, 256)

style_img = image_loader(style_image.format(idx), imsize).to(device, torch.float)
content_img = image_loader(content_image.format(idx), imsize).to(device, torch.float)
input_img = content_img.clone()

plt_images(style_img, input_img, content_img)

vgg = models.vgg19(pretrained=True).features.to(device).eval()

vgg_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
vgg_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
content_layers = ["conv4_2"]

output = run_style_transfer(
    vgg,
    vgg_normalization_mean,
    vgg_normalization_std,
    style_layers,
    content_layers,
    style_img,
    content_img,
    input_img,
    device,
    style_weight=1e7,
    content_weight=1e4,
    reg_weight=1e-3,
    num_steps=500,
)

save_image(output, output_image)
