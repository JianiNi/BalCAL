import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from torch import nn, optim
import torch
from torchmetrics.classification import Accuracy
from src.utils.losses import *
from src.utils.PLP_adam import *
from src.utils.metrics import Calibration
from src.utils.BayesAggMTL import BayesAggMTL
from src.utils.weight_methods import WeightMethods
import logging
import gc
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8).cuda()

def pearson_correlation(x, y):
    return d_cosine(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1))

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


class lt_disc_models(pl.LightningModule):
    def __init__(self, model, num_classes, loss='CE', MIT=False, mixup=False, alpha=1.0, device="cpu"):
        super().__init__()
        self.MIT = MIT
        self.mixup = mixup
        self.alpha = alpha
        self.model = model
        self._device= device
        if device == "gpu":
            self._device = "cuda:0"
        else:
            self._device = "cpu"
        assert isinstance(num_classes, int), "num_classes must be an integer"
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.criterion = loss
        if self.criterion == 'CE':
            self.criterion = nn.CrossEntropyLoss().cuda(self._device)
        if self.criterion == 'LS':
            self.criterion = LabelSmoothingCrossEntropy(alpha=0.05, ignore_index=-100, reduction="mean").cuda(self._device)
        if self.criterion == 'MbLS':
            self.criterion = LogitMarginL1(margin=6.0, alpha=0.1,ignore_index=-100,schedule= "",mu=0,max_alpha=100.0,step_size=100).cuda(self._device)        
        if self.criterion == 'ECP':
            self.criterion = PenaltyEntropy(alpha=0.1,ignore_index=-100).cuda(self._device)        
        if self.criterion == 'FL':
            self.criterion = FocalLoss(gamma=3.0,ignore_index=-100,size_average=True).cuda(self._device)        
        if self.criterion == 'FLSD':
            self.criterion = FocalLossAdaptive(gamma=3.0,ignore_index=-100,size_average=True,device=self._device).cuda(self._device)        
        if self.criterion == 'MDCA':
            self.criterion = MDCA().cuda(self._device)        
        if self.criterion == 'MMCE':
            self.criterion = MMCE(device=self._device).cuda(self._device) 
        if self.criterion == 'MMCE0':
            self.criterion = MMCE_weighted(device=self._device).cuda(self._device)    
        if self.criterion == 'ACLS':
            self.criterion = ACLS(pos_lambda=1.0,neg_lambda=0.1,alpha=0.1,margin=10.0,num_classes=num_classes).cuda(self._device)
        if self.criterion == 'ESD':
            self.criterion = ESD(lamda=0.5,device=self._device).cuda(self._device)
        if self.criterion == 'CPC':
            self.criterion = CPCLoss(lambd_bdc=0.1,lambd_bec=1.0,ignore_index=-100).cuda(self._device)
        # self.save_hyperparameters(ignore=['model'])
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.MIT is True:
            idx = torch.randperm(x.size(0))
            x_a, x_b = x, x[idx]
            y_a, y_b = y, y[idx]

            l1 = np.random.beta(self.alpha, self.alpha)      
            l1 = max(l1, 1-l1) 
            l2 = np.random.beta(self.alpha, self.alpha)
            l2 = min(l2, 1-l2)

            mixed_x1 = l1 * x_a + (1 - l1) * x_b     
            mixed_x2 = l2 * x_a + (1 - l2) * x_b      

            mixed_y_preds1 = self.model(mixed_x1)
            mixed_y_preds2 = self.model(mixed_x2)

            coef1 = (1 - l1) / (1 - l2)
            y_preds_a = (mixed_y_preds1 - coef1 * mixed_y_preds2) / (l1 - l2 * coef1)
            y_preds = y_preds_a

            coef2 = l2 / l1
            y_preds_b = (mixed_y_preds2 - coef2 * mixed_y_preds1) / (1 - l2 - (1 - l1) * coef2)

            loss1 = self.criterion(y_preds_a, y_a)
            loss2 = self.criterion(y_preds_b, y_b)
            loss = 0.5 * loss1 + 0.5 * loss2
        elif self.mixup is True:
            x, y_a, y_b, lam = mixup_data(x, y, alpha=self.alpha)
            y_preds = self.model(x)
            loss = lam * self.criterion(y_preds, y_a) + (1 - lam) * self.criterion(y_preds, y_b)
        else:
            y_preds = self.model(x)
            loss = self.criterion(y_preds, y)
        train_acc = self.train_acc(y_preds, y)
        self.log("train_accuracy", train_acc, on_step=True, on_epoch=False)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_preds = self.model(x)
        loss = nn.functional.cross_entropy(y_preds, y)
        val_acc= self.valid_acc(y_preds, y)
        self.log("valid_accuracy", val_acc, on_step=True, on_epoch=False)
        self.log("valid_loss", loss, on_step=True, on_epoch=False)
        self.log("valid_loss_epoch", loss, on_step=False, on_epoch=True)

    def on_train_epoch_end(self):
        # log epoch metric
        self.log('train_acc_epoch', self.train_acc, on_step=False, on_epoch=True)
        self.log('valid_acc_epoch', self.valid_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
class BalCAL_models(pl.LightningModule):
    def __init__(self, model, num_classes, delta=1.0 , lamd=0.5, loss='CE', MIT=False,mixup=False, alpha=1.0, device="cpu"):
        super().__init__()
        self.mixup = mixup
        self.alpha = alpha
        self.delta = delta
        self.lamd = model.lamd.detach()
        self.model = model
        self.num_classes = num_classes
        self._device = device
        self.criterion = loss
        if device == "gpu":
            self._device = "cuda:0"
        else:
            self._device = "cpu"
        assert isinstance(num_classes, int), "num_classes must be an integer"
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.cal1 = Calibration(num_classes,n_bins = 15)
        self.cal2 = Calibration(num_classes,n_bins = 15)
        self.criterion = loss
        if self.criterion == 'CE':
            self.criterion = nn.CrossEntropyLoss().cuda(self._device)
        if self.criterion == 'LS':
            self.criterion = LabelSmoothingCrossEntropy(alpha=0.03, ignore_index=-100, reduction="mean").cuda(self._device)
        if self.criterion == 'MbLS':
            self.criterion = LogitMarginL1(margin=6.0, alpha=0.1,ignore_index=-100,schedule= "",mu=0,max_alpha=100.0,step_size=100).cuda(self._device)        
        if self.criterion == 'ECP':
            self.criterion = PenaltyEntropy(alpha=0.1,ignore_index=-100).cuda(self._device)        
        if self.criterion == 'FL':
            self.criterion = FocalLoss(gamma=3.0,ignore_index=-100,size_average=True).cuda(self._device)        
        if self.criterion == 'FLSD':
            self.criterion = FocalLossAdaptive(gamma=3.0,ignore_index=-100,size_average=True,device=self._device).cuda(self._device)        
        if self.criterion == 'MDCA':
            self.criterion = MDCA().cuda(self._device)        
        if self.criterion == 'MMCE':
            self.criterion = MMCE(device=self._device).cuda(self._device) 
        if self.criterion == 'MMCE0':
            self.criterion = MMCE_weighted(device=self._device, lamda=1.5).cuda(self._device)    
        if self.criterion == 'ACLS':
            self.criterion = ACLS(pos_lambda=1.0,neg_lambda=0.1,alpha=0.1,margin=10.0,num_classes=num_classes).cuda(self._device)
        if self.criterion == 'ESD':
            self.criterion = ESD(lamda=4.0,device=self._device).cuda(self._device)
        if self.criterion == 'CPC':
            self.criterion = CPCLoss(lambd_bdc=0.1,lambd_bec=1.0,ignore_index=-100).cuda(self._device)
        # self._automatic_optimization = False
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        learned_norm = self.produce_Ew(y, self.num_classes)
        cur_M = learned_norm * self.model.ori_M
        y_linear, feat_ETF, y_preds = self.model(x,return_all = True)
        y_ETF = torch.matmul(feat_ETF, cur_M)
        
        train_acc = self.train_acc(y_preds, y)

        self.cal1.update(y_linear.detach(), y)
        self.cal2.update(y_ETF.detach(), y)
        
        loss1 = self.criterion(y_linear, y)
        loss2 = self.criterion(y_ETF, y)
        
        self.log("loss1", loss1, on_step=True, on_epoch=False)
        self.log("loss2", loss2, on_step=True, on_epoch=False)
        
        loss = self.lamd * loss1 + (1-self.lamd)* loss2
       
        self.log("train_accuracy", train_acc, on_step=True, on_epoch=False)
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch

        learned_norm = self.produce_Ew(y, self.num_classes)
        cur_M = learned_norm * self.model.ori_M
        y_linear, feat_ETF, y_preds = self.model(x,return_all = True)
        y_ETF = torch.matmul(feat_ETF, cur_M)

        loss1 = self.criterion(y_linear, y)
        loss2 = self.criterion(y_ETF, y)
      
        loss = loss1 + loss2
        val_acc= self.valid_acc(y_preds, y)
        conf, acc = self.cal1.handel(y_preds, y)
        avg_conf = conf.mean()
        ece, mce, _= self.cal1.calibrate(conf,acc)
        self.log("valid_accuracy", val_acc, on_step=True, on_epoch=False)
        self.log("valid_loss", loss, on_step=True, on_epoch=False)
        self.log("valid_loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("valid_conf", avg_conf, on_step=True, on_epoch=False)
        self.log("valid_conf_epoch", avg_conf, on_step=False, on_epoch=True)
        self.log("valid_ece", ece, on_step=True, on_epoch=False)
        self.log("valid_ece_epoch", ece, on_step=False, on_epoch=True)
        self.log("valid_mce", mce, on_step=True, on_epoch=False)
        self.log("valid_mce_epoch", mce, on_step=False, on_epoch=True)

    def on_train_epoch_start(self):
        self.cal1.reset()
        self.cal2.reset()

    def on_train_epoch_end(self):
        print("====================="+str(self.current_epoch)+"=====================")
        with torch.no_grad():
            self.lamd = self.find_lamd(delta=self.delta, tol=1e-4)
            self.model.lamd = self.lamd
        self.log('train_acc_epoch', self.train_acc)
        self.log('valid_acc_epoch', self.valid_acc)
        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
        
    def produce_Ew(self, label, num_classes):
        uni_label, count = torch.unique(label, return_counts=True)
        batch_size = label.size(0)
        uni_label_num = uni_label.size(0)
        assert batch_size == torch.sum(count)
        gamma = batch_size / uni_label_num
        Ew = torch.ones(1, num_classes).cuda(label.device)
        for i in range(uni_label_num):
            label_id = uni_label[i]
            label_count = count[i]
            length = torch.sqrt(gamma / label_count)
            Ew[0, label_id] = length
        return Ew
    
    def find_lamd(self, delta=0.9, tol=1e-3):
        left = 0.0
        right = 1.0
        best_lamd = 1.0

        conf1, acc1, preds1, targets = self.cal1.switch()
        conf2, acc2, preds2, _ = self.cal2.switch()
        avg_conf1, avg_acc1 = conf1.mean(), acc1.mean()

        if (avg_acc1 * delta) > avg_conf1:
            return nn.Parameter(torch.tensor(best_lamd))
        
        min_diff = abs(avg_acc1 * delta - avg_conf1)
        
        while right - left > tol:
            mid = (left + right) / 2
            mixed_preds = mid * preds1 + (1 - mid) * preds2
            conf, acc = self.cal1.handel(mixed_preds, targets)
            avg_conf, avg_acc = conf.mean(), acc.mean()

            diff = abs(avg_acc * delta - avg_conf)
            
            if diff < min_diff:
                best_lamd = mid

            if (avg_acc * delta) > avg_conf:
                left = mid
            else:
                right = mid
            
        mixed_preds = best_lamd * preds1 + (1 - best_lamd) * preds2
        avg_conf, avg_acc = self.cal1.handel(mixed_preds , targets)
        avg_conf, avg_acc = avg_conf.mean(), avg_acc.mean()

        return nn.Parameter(torch.tensor(best_lamd))
    