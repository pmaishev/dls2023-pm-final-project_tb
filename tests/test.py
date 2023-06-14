import sys
sys.path.insert(1, 'C:/Dev/dls2023-pm-final-project/bot')

from model.data_processing import CFileDataLoader
from model.style_transfer import CStyleTransferConfig, CStyleTransfer
#from model.style_transfer2 import process
import matplotlib.pyplot as plt
import numpy as np

def im_convert(tensor):
    # convert tesnor to image, denormalize
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

config = CStyleTransferConfig()
config.data_loader = CFileDataLoader(basedir='C:/Dev/dls2023-pm-final-project/bot/data_test')

trans = CStyleTransfer(config)

img = trans.transfer('0123456789') #'00000000-0000-0000-0000-000000000000')
plt.imshow(img)
plt.show()

#process()