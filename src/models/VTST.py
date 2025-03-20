import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.utils.utils import *
from torch import flatten
import torch.optim

class VTST(nn.Module):
    def __init__(self, dataset="MNIST", latent_dim=128, num_classes=10, separate_body=False, beta=0.01, pretrained_qyx = None, accelerator="cpu", bound_qzx_var=False, paper=None, simple_CNN=False, ViT_experiment=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.bound_qzx_var = bound_qzx_var
        
        if dataset == "CIFAR10" or separate_body:
            self.separate_body = True
        else:
            self.separate_body = False
        if dataset == "CIFAR10" or self.separate_body:
            self.qzx_body = construct_ClassYEncoderBody(pretrained_model=pretrained_qyx)
        self.qzx_model = construct_ClassYEncoder(dataset, self.latent_dim)
        if self.bound_qzx_var:
            self.qzx_var = construct_EncoderVar(dataset, self.latent_dim)

        self.pyz = construct_LabelDecoder(dataset, self.latent_dim, num_classes=num_classes)

        self.return_z = True

        self.num_classes = num_classes
        self.beta = beta
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode_fixed_var(self, x, y=None):
        if self.separate_body:
            x = self.qzx_body(x)
        z_logvar = self.qzx_var(x)
        z_mean = self.qzx_model(x)
        return z_mean, z_logvar

    def encode(self, x, y=None):
        if self.separate_body:
            x = self.qzx_body(x)
        z_mean, z_logvar = self.qzx_model(x)
        return z_mean, z_logvar

    def decode(self, z, y):
        pyz = self.pyz(z)
        return pyz
    
    def forward(self, x, y=None):
        if self.bound_qzx_var:
            z_mean, z_logvar = self.encode_fixed_var(x, y)
        else:
            z_mean, z_logvar = self.encode(x, y)
#        if not self.training:
#            z = z_mean
#        else:
        z = self.reparameterize(z_mean, z_logvar)
        pyz = self.decode(z, y)

        if self.training == True:
            return pyz, z_mean, z_logvar, z
        else:
            if self.return_z:
                return pyz, z_mean, z_logvar, z
            else:
                return pyz

    def forward_multisample(self, x, y=None, num_samples=10):
        #OBS: Outputs from this are already softmaxed!!!
        if self.bound_qzx_var:
            z_mean, z_logvar = self.encode_fixed_var(x, y)
        else:
            z_mean, z_logvar = self.encode(x, y)
        MC_probs = []
        for i in range(num_samples):
            z = self.reparameterize(z_mean, z_logvar)
            pyz = self.decode(z, y)
            MC_probs.append(F.softmax(pyz, dim=1))
        prob_tensor = torch.stack(MC_probs)
        preds = torch.mean(prob_tensor, dim=0)

        return preds


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