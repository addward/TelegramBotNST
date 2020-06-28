from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
import uuid
import copy

import os


class NeuralStyleTransfer(object):
    """
    Class for NeuralStyleTransfer
    """
    def __init__(self, im_size, device, cnn, normalization_mean, normalization_std):
        """
        Initialization
        Parameters
        ----------
        im_size : image size to which the input and output images will be transformed
        device : torch.device('cuda') or torch.device('cpu') - device that will be used for calculations
        cnn : convolution neural network for feature extraction and loss calculation.
        normalization_mean, normalization_std : parameters used for pretrained cnn
        """
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4']
        self.content_layers = ['conv_2', 'conv_5']
        self.im_size = im_size

        self.im_transform = transforms.Compose([
            transforms.Resize(self.im_size),
            transforms.CenterCrop(self.im_size),
            transforms.ToTensor()
        ])
        self.device = device
        self.max_iter = 5
        self.style_weight = 1e6
        self.content_weight = 1
        self.lr = 1

        self.content_losses = []
        self.style_losses = []

        self.model = self.get_style_model(cnn, normalization_mean, normalization_std)

    def image_loader(self, image_name):
        """
        Load image from local folder
        Parameters
        ----------
        image_name : path to local image
        """
        image = Image.open(image_name)
        image_tensor = self.im_transform(image).unsqueeze(0)
        image.close()
        image_tensor = image_tensor.to(
            self.device,
            torch.float
        )
        return image_tensor

    class Normalization(nn.Module):
        """
        Normalization layer
        """
        def __init__(self, mean, std, device):
            super().__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).to(device).view(-1, 1, 1).detach()
            self.std = torch.tensor(std).to(device).view(-1, 1, 1).detach()
            self.to(device)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std

    class ContentLoss(nn.Module):
        """
        ContentLoss layer that calculate content loss and store the result
        """
        def __init__(self):
            super().__init__()
            self.target = torch.rand(1)
            self.loss = torch.rand(1)
            self.set_target_mode = False

        def forward(self, input_img):
            if self.set_target_mode:
                self.target = input_img
            else:
                self.loss = F.mse_loss(input_img, self.target)
            return input_img

    class StyleLoss(nn.Module):
        """
        StyleLoss layer that calculate Gram matrix, style loss, and store the result
        """
        def __init__(self):
            super().__init__()
            self.target = torch.rand(1)
            self.loss = torch.rand(1)  # to initialize with something
            self.set_target_mode = False

        def forward(self, input):
            if self.set_target_mode:
                self.target = self.gram_matrix(input).detach()
            else:
                G = self.gram_matrix(input)
                self.loss = F.mse_loss(G, self.target)
            return input

        def gram_matrix(self, input):
            batch_size, features_num, h, w = input.shape

            flatten_features = input.view(batch_size * features_num, h * w)  # resise F_XL into \hat F_XL

            G = torch.mm(flatten_features, flatten_features.t())  # compute the gram product

            # we 'normalize' the values of the gram matrix
            # by dividing by the number of element in each feature maps.

            return G.div(batch_size * features_num * h * w)

    def get_style_model(self, cnn, normalization_mean, normalization_std):
        """
        Returns cnn with content and style loss layers, that will be used for style transfer
        Parameters
        ----------
        cnn : convolution neural network for feature extraction and loss calculation.
        normalization_mean, normalization_std : parameters used for pretrained cnn
        """
        normalization = self.Normalization(normalization_mean, normalization_std, self.device)

        # construct model from cnn as Sequential

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

            if name in self.content_layers:
                # add content loss:
                content_loss = self.ContentLoss()
                model.add_module("content_loss_{}".format(i), content_loss)
                self.content_losses.append(content_loss)

            if name in self.style_layers:
                style_loss = self.StyleLoss()
                model.add_module("style_loss{}".format(i), style_loss)
                self.style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], self.ContentLoss) or isinstance(model[i], self.StyleLoss):
                break

        final_model = model[:(i + 1)]
        del(model[(i+1):])
        return final_model

    def update_style_model(self, style_img, content_img):
        """
        Update the self.model target images in ContentLoss and StyleLoss layers
        Parameters
        ----------
        content_img      : image containing the desired content.
        style_img        : image containing the desired style
        """
        # Set lossLayers regime to set target image
        with torch.no_grad():
            for layer in self.style_losses + self.content_losses:
                layer.set_target_mode = True

            # Set all targets to style image

            self.model(style_img)

            # Freeze styleLoss layers and set target as content_img for remaining layers

            for layer in self.style_losses:
                layer.set_target_mode = False

            self.model(content_img)
            for layer in self.content_losses:
                layer.set_target_mode = False

    def get_input_optimizer(self, input_img):
        """
        Return LBFGS optimizer with pixels of input_img as learnable parameters.
        Parameters
        ----------
        input_img: input image for future style transfer
        """

        optimizer = optim.LBFGS([input_img.requires_grad_()],
                                lr=self.lr,
                                max_iter=self.max_iter,
                                history_size=20)
        return optimizer

    def run_style_transfer(self, content_img, style_img):
        """
        Run the style transfer.
        Parameters
        ----------
        content_img      : image containing the desired content.
        style_img        : image containing the desired style
        input_img        : input image for style transfer procedure
        """
        content_img = self.image_loader(content_img)
        style_img   = self.image_loader(style_img)
        input_img   = copy.deepcopy(content_img)
        # Update current ContentLoss and StyleLoss target images
        self.update_style_model(style_img, content_img)

        # Create optimizer

        optimizer = self.get_input_optimizer(input_img)

        # Array for images
        unload = transforms.ToPILImage()
        images = [unload(input_img.cpu().detach().squeeze(0))]

        def closure(model):
            # correct the values
            input_img.data.clamp_(0, 1)

            images.append(unload(input_img.cpu().detach().squeeze(0)))

            optimizer.zero_grad()
            model(input_img)

            style_score = 0
            content_score = 0

            for sl in self.style_losses:
                style_score += sl.loss
            for cl in self.content_losses:
                content_score += cl.loss

            print('Content loss: {}, style loss: {}'.format(content_score.item(),
                                                              style_score.item()))

            style_score *= self.style_weight
            content_score *= self.content_weight

            # Summarize content and style losses

            loss = style_score + content_score

            loss.backward()

            return style_score + content_score

        optimizer.step(lambda: closure(self.model))

        # clamp and detach image

        input_img.data.clamp_(0, 1)
        input_img = input_img.cpu().detach().squeeze(0)

        # save image with unique name

        input_img = unload(input_img)
        images.append(input_img)
        unq_pic_name = str(uuid.uuid4())
        input_img.save(unq_pic_name + '.jpg')
        images[0].save(unq_pic_name + '.gif', save_all=True,
                       append_images=images[1:], optimize=True,
                       duration = 40, loop = 0)
        # return name of saved image
        del(content_img)
        del(style_img)
        del(input_img)
        return unq_pic_name

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    cnn_normalization_mean = [0.485, 0.456, 0.406]
    cnn_normalization_std = [0.229, 0.224, 0.225]

    nst = NeuralStyleTransfer(256, device, cnn, cnn_normalization_mean, cnn_normalization_std)
    picname = nst.run_style_transfer('Examples/mozaic.jpg', 'Examples/oak_in.jpg')
    Image.open(picname + '.jpg').show()

