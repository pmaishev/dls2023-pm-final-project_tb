"""
Main class for style transferring
"""
import sys
sys.path.insert(1, 'C:/Dev/dls2023-pm-final-project/bot')

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models
from model.data_processing import IDataLoader, CFileDataLoader
from datetime import datetime
from model.style_transfer2 import im_convert

class CStyleTransferConfig():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_size = 512 if self.device == 'cuda' else 256
        self.data_loader = CFileDataLoader()
        self.mean_norm = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std_norm = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        self.cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(self.device).eval().requires_grad_(False)
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
        self.epoch = 500
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
        image = self.config.get_data_loader().load_data(wid, ftype)
        size = shape or min(max(image.size), self.config.max_size)
        in_transform = transforms.Compose([
                            transforms.Resize(size),
                            transforms.ToTensor(),
                            transforms.Normalize(self.config.mean_norm, 
                                                self.config.std_norm)])

        # discard the alpha channel (that's the :3) and add the batch dimension
        image = in_transform(image)[:3,:,:].unsqueeze(0)
        return image

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
        show_every = 10
        optimizer = self.config.optimizer([target_img], lr=3e-3)
        num_epochs = 500 


        for ii in range(1, num_epochs+1):
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
                layer_style_loss = 2 * self.config.style_weights[layer] * torch.mean((target_gram - style_gram)**2)
                # add to the style loss
                style_loss += layer_style_loss / (d * h * w)

            total_loss = self.config.content_weight*content_loss + self.config.style_weight*style_loss
            # update your target image
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # display intermediate images and print the loss
            if  ii % show_every == 0:
                #clear_output()
                plt.imshow(im_convert(target_img))
                plt.suptitle('Epoch %d/%d'%(ii, num_epochs))
                plt.show()



        # """
        # Method for style transferring:
        # - wid - path identifier to images with content, style and mask
        # """
        # content_img = self.load_and_transform_image(wid, 'content').to(self.config.device)
        # style_img = self.load_and_transform_image(wid, 'style', shape=content_img.shape[-2:]).to(self.config.device)
        # # try:
        # #     mask = self.config.get_data_loader().load_data(wid, 'mask')
        # # except ValueError:
        # #     mask = None
        # target_img = content_img.clone().requires_grad_(True).to(self.config.device)

        # #optimizer = self.config.optimizer([target_img])
        # content_features = self.get_features(target_img, cnn)
        # style_features = self.get_features(style_img, cnn)
        # style_grams = {layer: self.gram_matrix(style_features[layer]) for layer in style_features}

        # style_weights = {'conv1_1': 1.,
        #                 'conv2_1': 0.75,
        #                 'conv3_1': 0.2,
        #                 'conv4_1': 0.2,
        #                 'conv5_1': 0.2}
        # content_weight = 1 
        # style_weight = 1e9


        # print('Optimizing..')
        # show_every = 50
        # optimizer = optim.Adam([target_img], lr=3e-3)
        # num_epochs = 500 
        # for ii in range(1, num_epochs+1):
        #     target_features = self.get_features(target_img, cnn)
        #     content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        #     style_loss = 0
            
        #     for layer in style_weights:
        #         # get the "target" style representation for the layer
        #         target_feature = target_features[layer]
        #         target_gram = self.gram_matrix(target_feature)
        #         _, d, h, w = target_feature.shape
        #         # get the "style" style representation
        #         style_gram = style_grams[layer]
        #         # the style loss for one layer, weighted appropriately
        #         layer_style_loss = 2 * style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        #         # add to the style loss
        #         style_loss += layer_style_loss / (d * h * w)

        #     total_loss = content_weight * content_loss + style_weight * style_loss
        #     #print(total_loss)
        #     # update your target image
        #     optimizer.zero_grad()
        #     total_loss.backward()
        #     optimizer.step()
            
        #     # display intermediate images and print the loss
        #     if  ii % show_every == 0:
        #         #clear_output()
        #         plt.imshow(im_convert(target_img))
        #         plt.suptitle('Epoch %d/%d'%(ii, num_epochs))
        #         plt.show()




        # # for i in range(self.config.epoch):
        # #     target_features = self.get_features(target_img, cnn)
        # #     content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        # #     style_loss = 0
                
        # #     for layer in self.config.style_weights:
        # #         # get the "target" style representation for the layer
        # #         target_feature = target_features[layer]
                
        # #         target_gram = self.gram_matrix(target_feature)
                
        # #         _, d, h, w = target_feature.shape
        # #         # get the "style" style representation
        # #         style_gram = style_grams[layer]
        # #         # the style loss for one layer, weighted appropriately
        # #         layer_style_loss = self.config.style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        # #         # add to the style loss
        # #         style_loss += layer_style_loss / (d * h * w)

        # #     total_loss = self.config.content_weight*content_loss + self.config.style_weight*style_loss
        # #     #print(total_loss)
            
        # #     # update your target image
        # #     optimizer.zero_grad()
        # #     total_loss.backward()
        # #     optimizer.step()
            
        # #     # display intermediate images and print the loss
        # #     if  i % 10 == 0:
        # #         dt = datetime.now().strftime("%H:%M:%S")
        # #         print(f'{dt}: Epoche: {i} from {self.config.epoch}')


        # # for i in range(self.config.epoch):
        # #     def closure():
        # #         # correct the values of updated input image
        # #         with torch.no_grad():
        # #             target_img.clamp_(0, 1)

        # #         target_features = self.get_features(target_img, self.config.cnn)
        # #         content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        # #         style_loss = 0
                    
        # #         for layer in self.config.style_weights:
        # #             # get the "target" style representation for the layer
        # #             target_feature = target_features[layer]
                    
        # #             target_gram = self.gram_matrix(target_feature)
                    
        # #             _, d, h, w = target_feature.shape
        # #             # get the "style" style representation
        # #             style_gram = style_grams[layer]
        # #             # the style loss for one layer, weighted appropriately
        # #             layer_style_loss = self.config.style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        # #             # add to the style loss
        # #             style_loss += layer_style_loss / (d * h * w)

        # #         return self.config.content_weight*content_loss + self.config.style_weight*style_loss
        # #     optimizer.step(closure)
        # #     if  i % 10 == 0:
        # #         dt = datetime.now().strftime("%H:%M:%S")
        # #         print(f'{dt}: Epoche: {i} from {self.config.epoch}')

        # # # a last correction...
        # # with torch.no_grad():
        # #     target_img.clamp_(0, 1)
        return target_img
