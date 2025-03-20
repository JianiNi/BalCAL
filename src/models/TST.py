import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten
import torch.optim
# from src.utils.utils import *

class TST(nn.Module):
    def __init__(self, dataset="MNIST", latent_dim=128, num_classes=10, separate_body=False, pretrained_qyx = None, accelerator="cpu", paper=None, simple_CNN=False, ViT_experiment=False):
        super().__init__()
        self.latent_dim = latent_dim
        
        if dataset == "CIFAR10" or separate_body:
            self.separate_body = True
        else:
            self.separate_body = False
        if dataset == "CIFAR10" or self.separate_body:
            self.qzx_body = construct_ClassYEncoderBody(pretrained_model=pretrained_qyx)
        self.qzx_model = construct_ClassYEncoder(dataset, latent_dim)

        self.pyz = construct_LabelDecoder(dataset, self.latent_dim, num_classes=num_classes)

        self.return_z = False

        self.num_classes = num_classes
        if dataset.find("MNIST") == -1:
            self.input_h = 28
            self.input_w = 28
        elif dataset=="CIFAR10":
            self.input_h = 32
            self.input_w = 32  

        if accelerator == "gpu":
            self.device = "cuda:0"
        else:
            self.device = "cpu"

    def encode(self, x):
        if self.separate_body:
            x = self.qzx_body(x)
        return self.qzx_model(x)

    def decode(self, z):
        pyz = self.pyz(z)
        return pyz

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        if self.training:
            return y, z
        else:
            if self.return_z:
                return y, z
            else:
                return y



from collections import OrderedDict
from src.models.ClassYEncoders import *
from src.models.LabelDecoders import *
def construct_ClassYEncoder(dataset, latent_dim):
    return WRN2810Head(latent_dim)

def construct_EncoderVar(dataset, latent_dim):
    return WRN2810VarHead(latent_dim)

def construct_LabelDecoder(dataset, latent_dim, num_classes):
    return CIFAR10SimpelLabelDecoder(latent_dim, num_classes=num_classes)

def construct_ClassYEncoderBody(pretrained_model=None):
    if pretrained_model is None:
        return WRN2810Body(num_classes=10, depth=28, width=10, num_input_channels=3)
    else:
        pretrained_dict =  pretrained_model.state_dict()
        encoder_model = WRN2810Body(num_classes=10, depth=28, width=10, num_input_channels=3)
        encoder_dict = encoder_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        encoder_dict.update(pretrained_dict)
        encoder_model.load_state_dict(encoder_dict)
        return encoder_model