import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms

import copy

from closed_form_matting import compute_laplacian
from image_preprocessing import tensor_to_image, image_to_tensor


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):
    """
    See Gatys et al. for the details.
    """

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    B, C, H, W = input.size()
    features = input.view(B * C, H * W)
    gram = torch.mm(features, features.t())

    return gram.div(B * C * H * W)


class StyleLoss(nn.Module):
    """
    See Gatys et al. for the details.
    """

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        gram = gram_matrix(input)
        self.loss = F.mse_loss(gram, self.target)
        return input


def get_style_model_and_losses(
    cnn,
    normalization_mean,
    normalization_std,
    style_layers,
    content_layers,
    style_img,
    content_img,
    device,
):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    num_pool, num_conv = 0, 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            num_conv += 1
            name = "conv{}_{}".format(num_pool, num_conv)

        elif isinstance(layer, nn.ReLU):
            name = "relu{}_{}".format(num_pool, num_conv)
            layer = nn.ReLU(inplace=False)

        elif isinstance(layer, nn.MaxPool2d):
            num_pool += 1
            num_conv = 0
            name = "pool_{}".format(num_pool)
            layer = nn.AvgPool2d(
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
            )

        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn{}_{}".format(num_pool, num_conv)

        else:
            raise RuntimeError(
                "Unrecognized layer: {}".format(layer.__class__.__name__)
            )

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(num_pool), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()

            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(num_pool), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break

    model = model[: (i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(
    cnn,
    normalization_mean,
    normalization_std,
    style_layers,
    content_layers,
    style_img,
    content_img,
    input_img,
    device,
    num_steps=300,
    style_weight=100000,
    content_weight=1000,
    reg_weight=1000,
):
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn,
        normalization_mean,
        normalization_std,
        style_layers,
        content_layers,
        style_img,
        content_img,
        device,
    )
    optimizer = get_input_optimizer(input_img)

    L = compute_laplacian(tensor_to_image(content_img))

    def regularization_grad(input_img):
        """
        Photorealistic regularization
        See Luan et al. for the details.
        """
        im = tensor_to_image(input_img)
        grad = L.dot(im.reshape(-1, 3))
        loss = (grad * im.reshape(-1, 3)).sum()
        return loss, 2. * grad.reshape(*im.shape)

    step = 0
    
    while step <= num_steps:

        def closure():
            """
            https://pytorch.org/docs/stable/optim.html#optimizer-step-closure
            """
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            get_loss = lambda x: x.loss
            style_score = style_weight * sum(map(get_loss, style_losses))
            content_score = content_weight * sum(map(get_loss, content_losses))

            loss = style_score + content_score
            loss.backward()

            # Add photorealistic regularization
            reg_loss, reg_grad = regularization_grad(input_img)
            reg_grad_tensor = image_to_tensor(reg_grad)

            input_img.grad += reg_weight * reg_grad_tensor

            loss += reg_weight * reg_loss

            nonlocal step
            step += 1

            if step % 50 == 0:
                print(
                    "Step {:>4d}:".format(step),
                    "Style loss: {:.3f} Content loss: {:.3f} Photorealism regularization:{:.3f}".format(
                        style_score.item(), content_score.item(), reg_loss
                    ),
                )

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img
