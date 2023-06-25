# Notebook display
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# PyTorch
import torch
from torchvision import transforms
from torch.nn import Module, Parameter, Sequential, Upsample, ReflectionPad2d, Conv2d, InstanceNorm2d, ReLU

#DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CMsgStyleTransferConfig():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_path = '/data/cnn/msgnet_21_styles.pth'
        self.size = 1024
#        self.


class CoMatchLayer(Module):
    def __init__(self, channels, batch_size=1):
        super().__init__()

        self.C = channels
        self.weight = Parameter(torch.FloatTensor(1, channels, channels), requires_grad=True)
        self.GM_t = torch.FloatTensor(batch_size, channels, channels).requires_grad_()

        # Weight Initialization
        self.weight.data.uniform_(0.0, 0.02)

    def set_targets(self, GM_t):
        self.GM_t = GM_t

    def forward(self, x):
        self.P = torch.bmm(self.weight.expand_as(self.GM_t), self.GM_t)
        return torch.bmm(self.P.transpose(1, 2).expand(x.size(0), self.C, self.C), x.view(x.size(0), x.size(1), -1)).view_as(x)


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=False):
        super().__init__()

        self.upsample = Upsample(scale_factor=2) if upsample else None
        self.padding = ReflectionPad2d(kernel_size // 2) if kernel_size // 2 else None
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x) # pylint:disable=E1102
        if self.padding:
            x = self.padding(x)
        return self.conv(x)


class ResBlock(Module):

    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=False, upsample=False):
        super().__init__()

        self.down_conv = Conv2d(in_channels, channels * self.expansion, kernel_size=1, stride=stride) if downsample else None
        self.up_conv = ConvBlock(in_channels, channels * self.expansion, kernel_size=1, stride=1, upsample=upsample) if upsample else None

        self.conv_block = Sequential(InstanceNorm2d(in_channels), ReLU(),
                                     Conv2d(in_channels, channels, kernel_size=1, stride=1),
                                     InstanceNorm2d(channels), ReLU(),
                                     ConvBlock(channels, channels, kernel_size=3, stride=stride, upsample=upsample),
                                     InstanceNorm2d(channels), ReLU(),
                                     Conv2d(channels, channels * self.expansion, kernel_size=1, stride=1))

    def forward(self, x):
        residual = x
        if self.down_conv:
            residual = self.down_conv(x) # pylint:disable=E1102
        if self.up_conv:
            residual = self.up_conv(x) # pylint:disable=E1102
        return self.conv_block(x) + residual


class MSGNet(Module):
    def __init__(self, in_channels=3, out_channels=3, channels=128, num_res_blocks=6):
        super().__init__()

        # Siamese Network
        self.siamese_network = Sequential(ConvBlock(in_channels, 64, kernel_size=7, stride=1),
                                          InstanceNorm2d(64), ReLU(),
                                          ResBlock(64, 32, stride=2, downsample=True),
                                          ResBlock(32 * ResBlock.expansion, channels, stride=2, downsample=True))

        # CoMatch Layer
        self.comatch_layer = CoMatchLayer(channels * ResBlock.expansion)

        # Transformation Network
        self.transformation_network = Sequential(self.siamese_network,
                                                 self.comatch_layer,
                                                 *[ResBlock(channels * ResBlock.expansion, channels) for _ in range(num_res_blocks)],
                                                 ResBlock(channels * ResBlock.expansion, 32, stride=1, upsample=True),
                                                 ResBlock(32 * ResBlock.expansion, 16, stride=1, upsample=True),
                                                 InstanceNorm2d(16 * ResBlock.expansion), ReLU(),
                                                 ConvBlock(16 * ResBlock.expansion, out_channels, kernel_size=7, stride=1))

    def gram_matrix(self, inputs):
        BS, C, H, W = inputs.size()
        inputs = inputs.view(BS, C, H * W)
        GM = inputs.bmm(inputs.transpose(1, 2))
        return GM.div_(C * H * W)
    # def gram_matrix(self, tensor):
    #     batch_size, channels, height, width = tensor.size()
    #     tensor = tensor.view(batch_size * channels, height * width)
    #     gram = tensor @ tensor.t()
    #     return gram

    def set_targets(self, x):
        targets = self.siamese_network(x)
        GM_t = self.gram_matrix(targets)
        self.comatch_layer.set_targets(GM_t)

    def forward(self, x):
        return self.transformation_network(x)

class CStyleTransferMsg():
    def __init__(self, config: CMsgStyleTransferConfig = CMsgStyleTransferConfig()):
        self.config = config
        self.msg_net = MSGNet().to(config.device)
        self.msg_net.load_state_dict(torch.load(config.weights_path))

    def load_and_transform_image(self, path: str, ftype: str, shape=None):
        if ftype == 'url':
            image_io = BytesIO(requests.get(path).content)
        elif ftype == 'file':
            image_io = open(path, 'rb')
        elif ftype == 's3':
            raise NotImplementedError()
        else:
            raise ValueError(f'{ftype} is not valid ftype. ftype must be url, file or s3')
        image = Image.open(image_io).convert('RGB')
        return image

    def prep(self, image, keep_aspect_ratio=False, to_tensor=False):
        image = image.convert('RGB')
        if keep_aspect_ratio:
            size2 = int(self.config.size / image.size[0] * image.size[1])
            image = image.resize((self.config.size, size2), Image.LANCZOS)
        else:
            image = image.resize((self.config.size, self.config.size), Image.LANCZOS)
        if to_tensor:
            image2tensor = transforms.Compose([transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
                                            transforms.Lambda(lambda x: x.mul_(255))])
            return image2tensor(image).unsqueeze(0).to(self.config.device)
        else:
            return image

    def post(self, tensor):
        tensor = tensor.detach().cpu().squeeze(0).clamp_(0, 255)
        tensor2image = transforms.Compose([transforms.Lambda(lambda x: x.div_(255)),
                                        transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
                                        transforms.ToPILImage()])
        return tensor2image(tensor)

    def transfer(self, links: dict):
        # load content
        content_img = self.load_and_transform_image(links['content'], links.get('type', 'url'))
        # load styles, resize style to content
        style_img = self.load_and_transform_image(links['style'], links.get('type', 'url'))
        content, style = self.prep(content_img, keep_aspect_ratio=True, to_tensor=True), self.prep(style_img, to_tensor=True)
        self.msg_net.set_targets(style)
        self.msg_net.eval()
        with torch.no_grad():
            res = self.msg_net(content) # pylint:disable=E1102
        return self.post(res)

