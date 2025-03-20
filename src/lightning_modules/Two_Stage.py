import lightning.pytorch as pl
from torch import nn, optim
from torchmetrics.classification import Accuracy
from torchmetrics import KLDivergence
import foolbox.attacks as fa
import warnings
from foolbox import PyTorchModel, accuracy, samples
import torch
from lightning.pytorch.utilities import grad_norm
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torchmetrics.classification import MulticlassCalibrationError
from src.utils.losses import *
from src.utils.metrics import Calibration

class TS_Module(pl.LightningModule):
    def __init__(self, model, num_classes, freeze_qyx=False, device="cpu", dataset="MNIST"):
        super().__init__()
        self.model = model
        if device == "gpu":
            self.accelerator = "cuda:0"
        else:
            self.accelerator = "cpu"
        self._device = device
        self.num_classes = model.num_classes
        self.ece = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='l1')
        self.mce = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='max')
        self.ece = self.ece.to(self.accelerator)
        self.mce = self.mce.to(self.accelerator)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.dataset = dataset
        self.freeze_qyx=freeze_qyx
        self.save_hyperparameters()

    def configure_optimizers(self):
        if self.freeze_qyx:
            if self.model.separate_body:
                for param in self.model.qzx_body.parameters():
                    param.requires_grad = False
            optimizer=optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4)
        else:
            params = list(self.parameters())
            optimizer = optim.Adam(params, lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        pyz, z = self.model(x)
        class_loss = nn.functional.cross_entropy(pyz, y)
        loss = class_loss
        train_acc = self.train_acc(pyz, y)
        self.log("train_accuracy", train_acc, on_step=True, on_epoch=False)
        self.log("class_loss", class_loss, on_step=True, on_epoch=False)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pyz = self.model(x)
        class_loss = nn.functional.cross_entropy(pyz, y)
        loss = class_loss
        val_acc= self.valid_acc(pyz, y)
        self.log("valid_ECE_epoch", self.ece(pyz, y), on_step=False, on_epoch=True)
        self.log("valid_MCE_epoch", self.mce(pyz, y), on_step=False, on_epoch=True)
        self.log("valid_accuracy", val_acc, on_step=True, on_epoch=False)
        self.log("class_loss_valid", class_loss, on_step=False, on_epoch=True)
        self.log("valid_loss", loss, on_step=True, on_epoch=False)
        self.log("valid_loss_epoch", loss, on_step=False, on_epoch=True)

    def on_train_epoch_end(self):
        # log epoch metric
        self.log('train_acc_epoch', self.train_acc, on_step=False, on_epoch=True)
        self.log('valid_acc_epoch', self.valid_acc, on_step=False, on_epoch=True)

class VTST_Module(pl.LightningModule):
    def __init__(self, model, num_classes, freeze_qyx=False, device="cpu", dataset="MNIST"):
        super().__init__()
        self.model = model
        if device == "gpu":
            self.accelerator = "cuda:0"
        else:
            self.accelerator = "cpu"
        self._device = device
        self.num_classes = model.num_classes
        self.ece = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='l1')
        self.mce = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='max')
        self.ece = self.ece.to(self.accelerator)
        self.mce = self.mce.to(self.accelerator)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.dataset = dataset
        self.freeze_qyx=freeze_qyx
        self.save_hyperparameters()

    def configure_optimizers(self):
        if self.freeze_qyx:
            if self.model.separate_body:
                for param in self.model.qzx_body.parameters():
                    param.requires_grad = False
            optimizer=optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4)
        else:
            params = list(self.parameters())
            optimizer = optim.Adam(params, lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        pyz, z_mean, z_logvar, z = self.model(x)
        class_loss = nn.functional.cross_entropy(pyz, y)
        #KL Term
        kld_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp(), dim = 1)/self.model.latent_dim, dim = 0)
        loss = kld_loss + class_loss
        train_acc = self.train_acc(pyz, y)
        self.log("train_accuracy", train_acc, on_step=True, on_epoch=False)
        self.log("Z mean", torch.mean(z_mean), on_step=True, on_epoch=False)
        self.log("Z var", torch.mean(torch.exp(z_logvar)), on_step=True, on_epoch=False)
        self.log("kld_loss", kld_loss, on_step=True, on_epoch=False)
        self.log("class_loss", class_loss, on_step=True, on_epoch=False)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.model.return_z = True
        pyz, z_mean, z_logvar, z = self.model(x)
        self.model.return_z = False
        kld_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp(), dim = 1)/self.model.latent_dim, dim = 0)
        class_loss = nn.functional.cross_entropy(pyz, y)
        loss = kld_loss+class_loss
        val_acc= self.valid_acc(pyz, y)
        self.log("valid_ECE_epoch", self.ece(pyz, y), on_step=False, on_epoch=True)
        self.log("valid_MCE_epoch", self.mce(pyz, y), on_step=False, on_epoch=True)
        self.log("valid_accuracy", val_acc, on_step=True, on_epoch=False)
        self.log("kld_loss_valid", kld_loss, on_step=False, on_epoch=True)
        self.log("class_loss_valid", class_loss, on_step=False, on_epoch=True)
        self.log("valid_loss", loss, on_step=True, on_epoch=False)
        self.log("valid_loss_epoch", loss, on_step=False, on_epoch=True)

    def on_train_epoch_end(self):
        # log epoch metric
        self.log('train_acc_epoch', self.train_acc, on_step=False, on_epoch=True)
        self.log('valid_acc_epoch', self.valid_acc, on_step=False, on_epoch=True)
