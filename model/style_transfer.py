"""
Main class for style transferring
"""
import os
from PIL import Image
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models
from datetime import datetime
from model.data_processing import IDataLoader, CFileDataLoader

class CStyleTransferConfig():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_size = 512 if self.device == 'cuda' else 128
        self.data_loader = CFileDataLoader()
        self.mean_norm = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std_norm = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        if os.path.isfile('/data/cnn/vgg19.pth'):
            self.cnn = models.vgg19() # we do not specify ``weights``, i.e. create untrained model
            self.cnn.load_state_dict(torch.load('/data/cnn/vgg19.pth'))
        else:
            self.cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.cnn = self.cnn.features.to(self.device).eval().requires_grad_(False)
        for (i, layer) in enumerate(self.cnn):
            if isinstance(layer, torch.nn.MaxPool2d):
                self.cnn[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.layers = {'0': 'conv1_1',
                    '5': 'conv2_1',
                    '10': 'conv3_1',
                    '19': 'conv4_1',
                    '21': 'conv4_2',  # content layer
                    '28': 'conv5_1'}
        self.content_loss_layer = 'conv4_2'
        self.style_weights = {'conv1_1': 1.,
                    'conv2_1': 0.75,
                    'conv3_1': 0.2,
                    'conv4_1': 0.2,
                    'conv5_1': 0.2}
        self.content_weight = 1
        self.style_weight = 1e6
        self.epoch = 2000
        self.optimizer = optim.Adam #LBFGS #Adam

    def get_data_loader(self) -> IDataLoader:
        return self.data_loader

class CStyleTransfer():
    def __init__(self, config: CStyleTransferConfig = CStyleTransferConfig()):
        self.config = config

    def get_features(self, image, model, layers=None) -> dict:
        if layers is None:
            layers = self.config.layers

        features = {}
        x = image

        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def gram_matrix(self, tensor):
        batch_size, channels, height, width = tensor.size()
        tensor = tensor.view(batch_size * channels, height * width)
        gram = tensor @ tensor.t()
        return gram

    def load_and_transform_image(self, wid: str, ftype: str, shape=None):
        image_io = self.config.get_data_loader().load_data(wid, ftype)
        image = Image.open(image_io).convert('RGB')

        size = shape or min(max(image.size), self.config.max_size)
        in_transform = transforms.Compose([
                            transforms.Resize(size),
                            transforms.ToTensor(),
                            transforms.Normalize(self.config.mean_norm,
                                                self.config.std_norm)])

        # discard the alpha channel (that's the :3) and add the batch dimension
        image = in_transform(image)[:3,:,:].unsqueeze(0)
        return image

    def im_convert(self, tensor):
        # convert tesnor to image, denormalize
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1,2,0)
        image = image * np.array(self.config.std_norm) + np.array(self.config.mean_norm)

        return (image.clip(0, 1)*255).astype('uint8')

    def transfer(self, wid: str):
        # load content
        content_img = self.load_and_transform_image(wid, 'content').to(self.config.device)
        # load styles, resize style to content
        style_img = self.load_and_transform_image(wid, 'style', shape=content_img.shape[-2:]).to(self.config.device)
        # TODO: load mask, do not normalize and resize mask, sicne we will resize and binarize it later anyway
        # try:
        #     mask = self.config.get_data_loader().load_data(wid, 'mask')
        # except ValueError:
        #     mask = None

        content_features = self.get_features(content_img, self.config.cnn)
        style_features = self.get_features(style_img, self.config.cnn)
        style_grams = {layer: self.gram_matrix(style_features[layer]) for layer in style_features}

        target_img = content_img.clone().requires_grad_(True).to(self.config.device)
        optimizer = self.config.optimizer([target_img], lr=3e-3)

        for i in range(self.config.epoch):
            target_features = self.get_features(target_img, self.config.cnn)
            content_loss = torch.mean((target_features[self.config.content_loss_layer] - content_features[self.config.content_loss_layer])**2)
            style_loss = 0

            for layer in self.config.style_weights:
                # get the "target" style representation for the layer
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                _, d, h, w = target_feature.shape
                # get the "style" style representation
                style_gram = style_grams[layer]
                # the style loss for one layer, weighted appropriately
                layer_style_loss = 2*self.config.style_weights[layer]*torch.mean((target_gram - style_gram)**2)
                # add to the style loss
                style_loss += layer_style_loss / (d*h*w)

            total_loss = self.config.content_weight*content_loss + self.config.style_weight*style_loss
            # update your target image
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # display intermediate images and print the loss
            if  i % 50 == 0:
                dt = datetime.now()
                print(f'{dt.strftime("%H:%M:%S")}: {i}/{self.config.epoch}')

        return Image.fromarray(self.im_convert(target_img), 'RGB')
