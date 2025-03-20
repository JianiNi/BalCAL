from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, SVHN, CIFAR100
import torch
import os
from torchvision.transforms import ToTensor
from src.lightning_modules.One_Stage import *
from src.lightning_modules.Two_Stage import *
import wandb
import torchvision.datasets as datasets
from src.models.WRN import *
import torch.utils.data as data
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from pathlib import Path
import argparse
from src.utils.utils import *
import torchvision.transforms as transforms
from lightning.pytorch.callbacks import ModelCheckpoint
import time

parser = ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--seeds_per_job", type=int, default=1)
parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--model", type=str, default="WRN")
parser.add_argument("--loss", type=str, default="CE")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--pretrained_qyx", type=str, default=None)
parser.add_argument("--freeze_qyx", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--model_name", type=str, default="Unknown")
parser.add_argument("--simple_decoder", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--fix_varq", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--PLP", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--MIT", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--mixup", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--delta", type=float, default=1.0)
if torch.cuda.is_available():
    parser.add_argument("--accelerator", type=str, default="gpu")
else:
    parser.add_argument("--accelerator", type=str, default="cpu")
args = parser.parse_args()

if args.dataset == "CIFAR10":
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.247, 0.243, 0.261))])
    train_dataset = CIFAR10(os.getcwd()+"/data/", download=True, transform=transform, train=True)
    num_classes = 10 
elif args.dataset == "CIFAR100":
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))])
    train_dataset = CIFAR100(os.getcwd()+"/data/", download=True, transform=transform, train=True)
    num_classes = 100
elif args.dataset == "SVHN":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))])
    train_dataset = SVHN(os.getcwd()+"/data/", download=True, transform=transform, split="train")
    num_classes = 10 
else:
    raise Exception("Oops, requested dataset does not exist!")

if args.dataset == "CIFAR100":
    #The validation propertion is actually 1-valid_proportion
    valid_proportion = 0.95
else:
    valid_proportion = 0.8

train_set_size = int(len(train_dataset) * valid_proportion)
valid_set_size = len(train_dataset) - train_set_size


torch_seed = torch.Generator()
torch_seed.manual_seed(1)
train_set, valid_set = data.random_split(train_dataset, [train_set_size, valid_set_size], generator=torch_seed)
train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=18)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size,num_workers=18)  

for i in range(args.seeds_per_job):
    seed = args.seed + i
    if args.model=="BalCAL":
        model_name = args.dataset+"_"+args.model+"_"+str(args.delta)+"_"+str(args.loss)+"_seed"+str(seed)
    elif args.mixup is True:
        model_name = args.dataset+"_"+args.model+"_"+str(args.loss)+"_mixup_"+str(args.alpha)+"_seed"+str(seed)
    elif args.MIT is True:
        model_name = args.dataset+"_"+args.model+"_"+str(args.loss)+"_MIT_"+str(args.alpha)+"_seed"+str(seed)
    elif args.model=="TST" or args.model=="VTST":
        model_name = args.dataset+"_"+args.model+"_"+str(args.latent_dim)+"_"+str(args.loss)+"_seed"+str(seed)
    else:
        model_name = args.dataset+"_"+args.model+"_"+str(args.loss)+"_"+"_seed"+str(seed)
    torch_seed.manual_seed(seed)
    seed_everything(seed, workers=True)

    save_dir = "./experiment_results/"+ args.dataset+"_" + args.model+"/"+time.strftime('%y%m%d%H%M%S')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if args.model=="BalCAL":
        wandb_logger = WandbLogger(project='ETF_WRN', save_dir=save_dir, name=str(seed)+'_'+str(args.model)+"_"+str(args.delta)+"_"+str(args.loss))
    elif args.mixup is True:
        wandb_logger = WandbLogger(project='ETF_WRN', save_dir=save_dir, name=str(seed)+'_'+str(args.model)+'_'+str(args.loss)+"_mixup_"+str(args.alpha))
    elif args.MIT is True:
        wandb_logger = WandbLogger(project='ETF_WRN', save_dir=save_dir, name=str(seed)+'_'+str(args.model)+'_'+str(args.loss)+"_MIT_"+str(args.alpha))
    elif args.model=="TST" or args.model=="VTST":
        wandb_logger = WandbLogger(project='ETF_WRN', save_dir=save_dir, name=str(seed)+'_'+str(args.model)+"_"+str(args.latent_dim)+'_'+str(args.loss))
    else:
        wandb_logger = WandbLogger(project='ETF_WRN', save_dir=save_dir, name=str(seed)+'_'+str(args.model)+'_'+str(args.loss))
    for arg in vars(args):
        wandb_logger.experiment.config[arg] = getattr(args, arg)

    checkpoint_callback = ModelCheckpoint(dirpath=save_dir+"/checkpoints/", every_n_epochs=1, save_top_k=1, mode="min", monitor='valid_loss_epoch', auto_insert_metric_name=False, filename=model_name+'-epoch={epoch:02d}')
    trainer = Trainer(logger=wandb_logger, max_epochs=args.epochs, callbacks=[checkpoint_callback], accelerator=args.accelerator, devices=args.devices, enable_progress_bar=False)
    
    if args.model == "WRN" and (args.dataset == "SVHN" or args.dataset=="CIFAR100" or args.dataset=="CIFAR10"):
        model = WideResNet(num_classes=num_classes, depth=28, width=10, num_input_channels=3)
    elif (args.dataset == "SVHN" or args.dataset =="CIFAR10" or args.dataset=="CIFAR100") and args.model=="TST" and args.pretrained_qyx is not None:
        model = TST(dataset=args.dataset, num_classes=num_classes, latent_dim=args.latent_dim, accelerator=args.accelerator, pretrained_qyx=load_WRN_model(args.pretrained_qyx, dataset=args.dataset), separate_body=True)
    elif (args.dataset == "SVHN" or args.dataset =="CIFAR10" or args.dataset=="CIFAR100") and args.model=="TST" and args.pretrained_qyx is None:
        model = TST(dataset=args.dataset, num_classes=num_classes, latent_dim=args.latent_dim, accelerator=args.accelerator, pretrained_qyx=None, separate_body=True)
    elif (args.dataset == "SVHN" or args.dataset =="CIFAR10" or args.dataset=="CIFAR100") and args.model=="VTST" and args.pretrained_qyx is not None:
        model = VTST(dataset=args.dataset, num_classes=num_classes, latent_dim=args.latent_dim, accelerator=args.accelerator, bound_qzx_var=True, pretrained_qyx=load_WRN_model(args.pretrained_qyx, dataset=args.dataset), separate_body=True)
    elif (args.dataset == "SVHN" or args.dataset =="CIFAR10" or args.dataset=="CIFAR100") and args.model=="VTST" and args.pretrained_qyx is None:
        model = VTST(dataset=args.dataset, num_classes=num_classes, latent_dim=args.latent_dim, accelerator=args.accelerator, bound_qzx_var=True, pretrained_qyx=None, separate_body=True)
    elif args.model == "BalCAL" and (args.dataset == "SVHN" or args.dataset=="CIFAR100" or args.dataset=="CIFAR10"):
        model = WideResNet_BalCAL(num_classes=num_classes, depth=28, width=10, num_input_channels=3)
    else:
        raise Exception("Oops, requested model does not exist for this specific dataset!")

    if args.model =="WRN":
        lightning_module = lt_disc_models(model, num_classes, loss=args.loss, MIT=args.MIT, mixup=args.mixup, alpha=args.alpha, device= args.accelerator)
    elif args.model == "BalCAL":
        lightning_module = BalCAL_models(model, num_classes, delta=args.delta,  loss=args.loss, MIT=args.MIT, mixup=args.mixup, alpha=args.alpha, device= args.accelerator)
    elif args.model == "TST":
        lightning_module = TS_Module(model, num_classes, device=args.accelerator, freeze_qyx=args.freeze_qyx, dataset=args.dataset)
    elif args.model == "VTST":
        lightning_module = VTST_Module(model, num_classes, device=args.accelerator, freeze_qyx=args.freeze_qyx, dataset=args.dataset)
    else:
        raise Exception("Oops, requested model does not have an accompanying lightning module!")

    print("===================================")
    print('save_dir:'+save_dir)
    print("===================================")

    trainer.fit(model=lightning_module, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    wandb.finish()