import torch
from PIL import Image
from skimage.transform import resize
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def tensor_to_image(x):
    """
    Transforms torch.Tensor to np.array
        (1, C, W, H) -> (W, H, C)
        (B, C, W, H) -> (B, W, H, C) 

    """
    return x.detach().numpy().transpose(0, 2, 3, 1).squeeze().clip(0, 1)


def image_to_tensor(x):
    """
    Transforms np.array to torch.Tensor
        (W, H)       -> (1, 1, W, H)
        (W, H, C)    -> (1, C, W, H)
        (B, W, H, C) -> (B, C, W, H)

    """
    if x.ndim == 2:
        return torch.Tensor(x).unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return torch.Tensor(x.transpose(2, 0, 1)).unsqueeze(0)
    if x.ndim == 4:
        return torch.Tensor(x.transpose(0, 3, 1, 2))
    raise RuntimeError("np.array's ndim is out of range 2, 3 or 4.")


def masks_loader(path_style, path_content, size):
    """
    Loads masks.
    """
    style_masks, content_masks = get_masks(path_style, path_content)
    style_masks, content_masks = resize_masks(style_masks, content_masks, size)
    style_masks, content_masks = masks_to_tensor(style_masks, content_masks)

    return style_masks, content_masks


def image_loader(image_name, size):
    """
    Loads images.
    """
    loader = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image


def plt_images(
    style_img,
    output_img,
    content_img,
    style_title="Style Image",
    output_title="Output Image",
    content_title="Content Image",
):
    """
    Plots style, output and content images to ease comparison.
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(tensor_to_image(style_img))
    plt.title("Style Image")

    plt.subplot(1, 3, 2)
    plt.imshow(tensor_to_image(output_img))
    plt.title("Output Image")

    plt.subplot(1, 3, 3)
    plt.imshow(tensor_to_image(content_img))
    plt.title("Content Image")

    plt.tight_layout()
    plt.show()
