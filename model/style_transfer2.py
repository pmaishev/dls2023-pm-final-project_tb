# import sys
# sys.path.insert(1, 'C:/Dev/dls2023-pm-final-project/bot')

# from model.data_processing import CFileDataLoader
# from model.style_transfer import CStyleTransferConfig, CStyleTransfer



# from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np

# import torch
# import torch.optim as optim
# from torchvision import transforms, models

# from IPython.display import clear_output

# torch.backends.cudnn.benchmark = True

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval().requires_grad_(False)
# for (i, layer) in enumerate(cnn):
#     if isinstance(layer, torch.nn.MaxPool2d):
#         cnn[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

def im_convert(tensor):
    
    # convert tesnor to image, denormalize
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

# def load_image(img_path, max_size=256, shape=None):
    
#     image = Image.open(img_path).convert('RGB')
    
#     size = shape or min(max(image.size), max_size)
    
#     in_transform = transforms.Compose([
#                         transforms.Resize(size),
#                         transforms.ToTensor(),
#                         transforms.Normalize((0.485, 0.456, 0.406), 
#                                              (0.229, 0.224, 0.225))])

#     # discard the alpha channel (that's the :3) and add the batch dimension
    
#     image = in_transform(image)[:3,:,:].unsqueeze(0)
    
#     return image

# def get_features1(image, model, layers=None):
    
#     if layers is None:
#         layers = {'0': 'conv1_1',
#                   '5': 'conv2_1', 
#                   '10': 'conv3_1', 
#                   '19': 'conv4_1',
#                   '21': 'conv4_2',  # content layer
#                   '28': 'conv5_1'}
    
#     features = {}
#     x = image
    
#     for name, layer in model._modules.items():
#         x = layer(x)
#         if name in layers:
#             features[layers[name]] = x
            
#     return features

# def gram_matrix(tensor):

#     batch_size, channels, height, width = tensor.size()
    
#     tensor = tensor.view(batch_size * channels, height * width)
    
#     gram = tensor @ tensor.t()
    
#     return gram 

# def load_and_transform_image(config, wid: str, ftype: str, shape=None):
#     image = config.get_data_loader().load_data(wid, ftype)
#     size = shape or min(max(image.size), config.max_size)
#     in_transform = transforms.Compose([
#                         transforms.Resize(size),
#                         transforms.ToTensor(),
#                         transforms.Normalize(config.mean_norm, 
#                                             config.std_norm)])

#         # discard the alpha channel (that's the :3) and add the batch dimension
#     image = in_transform(image)[:3,:,:].unsqueeze(0)
#     return image

# def get_features(config, image, model, layers=None) -> dict:
#     if layers is None:
#         layers = config.layers

#     features = {}
#     x = image

#     for name, layer in model._modules.items():
#         x = layer(x)
#         if name in layers:
#             features[layers[name]] = x
#     return features

# def gram_matrix(tensor):
#     batch_size, channels, height, width = tensor.size()
#     tensor = tensor.view(batch_size * channels, height * width)
#     gram = tensor @ tensor.t()
#     return gram 


# # def process():
# #     config = CStyleTransferConfig()
# #     config.data_loader = CFileDataLoader(basedir='C:/Dev/dls2023-pm-final-project/bot/data_test')

# #     # load content
# #     #content = load_image('C:/Users/pmaishev/Documents/old/python/DLS/dls/19. ДЗ. GAN и Style Transfer/bruges.jpg').to(device)
# #     content = load_and_transform_image(config, '1234567890', 'content').to(config.device)
# #     # load styles, resize style to content
# #     #style = load_image('C:/Users/pmaishev/Documents/old/python/DLS/dls/19. ДЗ. GAN и Style Transfer/starry_night.jpg', shape=content.shape[-2:]).to(device)
# #     style = load_and_transform_image(config, '1234567890', 'style', shape=content.shape[-2:]).to(config.device)
# #     # load mask, do not normalize and resize mask, sicne we will resize and binarize it later anyway

# #     content_features = get_features(config, content, cnn)
# #     style_features = get_features(config, style, cnn)
# #     style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# #     style_weights = {'conv1_1': 1.,
# #                     'conv2_1': 0.75,
# #                     'conv3_1': 0.2,
# #                     'conv4_1': 0.2,
# #                     'conv5_1': 0.2}
# #     content_weight = 1 
# #     style_weight = 1e9

# #     target = content.clone().requires_grad_(True).to(device)
# #     show_every = 50
# #     optimizer = optim.Adam([target], lr=3e-3)
# #     num_epochs = 500 


# #     for ii in range(1, num_epochs+1):
# #         target_features = get_features(config, target, cnn)
# #         content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
# #         style_loss = 0
        
# #         for layer in style_weights:
# #             # get the "target" style representation for the layer
# #             target_feature = target_features[layer]
# #             target_gram = gram_matrix(target_feature)
# #             _, d, h, w = target_feature.shape
# #             # get the "style" style representation
# #             style_gram = style_grams[layer]
# #             # the style loss for one layer, weighted appropriately
# #             layer_style_loss = 2 * style_weights[layer] * torch.mean((target_gram - style_gram)**2)
# #             # add to the style loss
# #             style_loss += layer_style_loss / (d * h * w)

# #         total_loss = content_weight * content_loss + style_weight * style_loss
# #         #print(total_loss)
# #         # update your target image
# #         optimizer.zero_grad()
# #         total_loss.backward()
# #         optimizer.step()
        
# #         # display intermediate images and print the loss
# #         if  ii % show_every == 0:
# #             #clear_output()
# #             plt.imshow(im_convert(target))
# #             plt.suptitle('Epoch %d/%d'%(ii, num_epochs))
# #             plt.show()
