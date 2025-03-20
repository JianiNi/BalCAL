import torchvision.transforms as transforms
import os
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, SVHN, CIFAR100
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassCalibrationError, Accuracy
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate
from ood_metrics import fpr_at_95_tpr, auroc
import torch.nn as nn
import torch.nn.functional as F
import torch
from src.utils.metrics import *
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.utils.data as data
from scipy.stats import entropy

SVHN_ROTATIONS = [10., 45., 90., 135., 180.]
CORRUPTIONS = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'glass_blur', 'impulse_noise', 'jpeg_compression',
    'motion_blur', 'pixelate', 'shot_noise', 'snow'
]

CIFAR100_C_CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                'defocus_blur', 'glass_blur', 'motion_blur',
                'zoom_blur', 'snow', 'frost',
                'brightness', 'contrast', 'elastic_transform',
                'pixelate', 'jpeg_compression', 'speckle_noise',
                'gaussian_blur', 'spatter', 'saturate']

CIFAR10_C_PATH = "./data/CIFAR10-C/"
CIFAR100_C_PATH = "./data/CIFAR100-C/"

def eval_train_data(model, dataset, batch_size, device, num_samples=1):
    train_dataset, num_classes, _, _ = get_dataset(dataset, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    y_preds = []
    y_targets = []
    with torch.no_grad():
        for j, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            if num_samples == 1:
                preds = model(images)
                probs = nn.functional.softmax(preds, dim=1)
                y_targets.append(labels)
                y_preds.append(probs)
            else:
                probs = model.forward_multisample(images, num_samples=num_samples)
                y_targets.append(labels)
                y_preds.append(probs)
    y_preds = torch.cat(y_preds, dim=0)
    y_targets = torch.cat(y_targets, dim=0)
    nll_value = nll(y_preds, y_targets) 
    return nll_value

def eval_test_data(model, dataset, batch_size, device, num_samples=1):
    test_dataset, num_classes, _, _ = get_dataset(dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    accuracy = accuracy.to(device)
    ece = MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='l1')
    mce = MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='max')
    aece = AdaptiveECELoss(n_bins=15).to(device)
    cece = ClasswiseECELoss(n_bins=15).to(device)
    ece = ece.to(device)
    mce = mce.to(device)
    y_preds = []
    y_targets = []
    OOD_labels = []
    OOD_y_preds_logits = []

    # white = torch.ones(1, 3, 32, 32).cuda()
    # black = torch.zeros(1, 3, 32, 32).cuda()

    # blue = torch.zeros(1, 3, 32, 32).cuda()
    # blue[:, 2, :, :] = 1 

    # green = torch.zeros(1, 3, 32, 32).cuda()
    # green[:, 1, :, :] = 1

    # red = torch.zeros(1, 3, 32, 32).cuda()
    # red[:, 0, :, :] = 1 

    # biaseddegree_black = model(black)
    # biaseddegree_blue = model(blue)
    # biaseddegree_green = model(green)
    # biaseddegree_red = model(red)
    # biaseddegree_white = model(white)

    with torch.no_grad():
        for j, (images, labels) in enumerate(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            # print(images.shape)
            if num_samples == 1:
                preds = model(images)
                # preds, z = model(images)
                # preds = preds - biaseddegree
                probs = nn.functional.softmax(preds, dim=1)
                # print(probs[0])
            else:
                probs = model.forward_multisample(images, num_samples=num_samples)
            acc = accuracy(probs, labels)
            ece.update(probs, labels)
            mce.update(probs, labels)
            y_targets.append(labels)
            y_preds.append(probs)
            max_predictions = torch.max(probs.data, 1).values
            OOD_y_preds_logits.append(max_predictions)
            OOD_labels.append(torch.tensor([1]*len(labels)))
    
    
    y_preds = torch.cat(y_preds, dim=0)
    y_targets = torch.cat(y_targets, dim=0)

    ece_calc = ece.compute()
    mce_calc = mce.compute()
    acc = accuracy.compute()
    conf = torch.cat(ece.confidences, dim=0).mean()
    aece_calc = aece(y_preds, y_targets) 
    cece_calc = cece(y_preds, y_targets) 
    nll_value = nll(y_preds, y_targets) 
    brier_score = brier(y_preds, y_targets)
    return ece_calc, mce_calc, aece_calc, cece_calc, acc, conf, nll_value, brier_score, OOD_y_preds_logits, OOD_labels

def get_dataset(dataset, train=False):
    if dataset == "SVHN":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))])
        if train==True:
            dataset = SVHN(os.getcwd()+"/data/", download=True, transform=transform, split="train")
        else:
            dataset = SVHN(os.getcwd()+"/data/", download=True, transform=transform, split="test")
        num_classes = 10
        n_samples = 10000
        input_shape = [1, 32, 32]
    elif dataset == "CIFAR10":
        dataset = CIFAR10(os.getcwd()+"/data/", download=True, transform=transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.247, 0.243, 0.261))]), train=train)
        input_shape=[1, 32, 32]
        n_samples=10000
        num_classes=10
    elif dataset == "CIFAR100":
        dataset = CIFAR100(os.getcwd()+"/data/", download=True, transform=transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))]), train=train)
        input_shape=[1, 32, 32]
        n_samples=10000
        num_classes=100
    return dataset, num_classes, n_samples, input_shape

def get_mydataset(dataset, train=False, eval=False):
    if dataset == "SVHN":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))])
        if train==True:
            dataset = SVHN(os.getcwd()+"/data/", download=True, transform=transform, split="train")
            valid_proportion = 0.8
            train_set_size = int(len(dataset) * valid_proportion)
            valid_set_size = len(dataset) - train_set_size
            torch_seed = torch.Generator()
            torch_seed.manual_seed(1)
            dataset, _ = data.random_split(dataset, [train_set_size, valid_set_size], generator=torch_seed)
        elif eval==True:
            dataset = SVHN(os.getcwd()+"/data/", download=True, transform=transform, split="train")
            valid_proportion = 0.8
            train_set_size = int(len(dataset) * valid_proportion)
            valid_set_size = len(dataset) - train_set_size
            torch_seed = torch.Generator()
            torch_seed.manual_seed(1)
            _, dataset = data.random_split(dataset, [train_set_size, valid_set_size], generator=torch_seed)
        else:
            dataset = SVHN(os.getcwd()+"/data/", download=True, transform=transform, split="test")
        num_classes = 10
        n_samples = 10000
        input_shape = [1, 32, 32]
    elif dataset == "CIFAR10":
        dataset = CIFAR10(os.getcwd()+"/data/", download=True, transform=transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.247, 0.243, 0.261))]), train=train)
        if train==True:
            valid_proportion = 0.8
            train_set_size = int(len(dataset) * valid_proportion)
            valid_set_size = len(dataset) - train_set_size
            torch_seed = torch.Generator()
            torch_seed.manual_seed(1)
            dataset, _ = data.random_split(dataset, [train_set_size, valid_set_size], generator=torch_seed)
        elif eval==True:
            valid_proportion = 0.8
            train_set_size = int(len(dataset) * valid_proportion)
            valid_set_size = len(dataset) - train_set_size
            torch_seed = torch.Generator()
            torch_seed.manual_seed(1)
            _, dataset = data.random_split(dataset, [train_set_size, valid_set_size], generator=torch_seed)
        input_shape=[1, 32, 32]
        n_samples=10000
        num_classes=10
    elif dataset == "CIFAR100":
        dataset = CIFAR100(os.getcwd()+"/data/", download=True, transform=transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))]), train=train)
        if train==True:
            valid_proportion = 0.95
            train_set_size = int(len(dataset) * valid_proportion)
            valid_set_size = len(dataset) - train_set_size
            torch_seed = torch.Generator()
            torch_seed.manual_seed(1)
            dataset, _ = data.random_split(dataset, [train_set_size, valid_set_size], generator=torch_seed)
        elif eval==True:
            valid_proportion = 0.95
            train_set_size = int(len(dataset) * valid_proportion)
            valid_set_size = len(dataset) - train_set_size
            torch_seed = torch.Generator()
            torch_seed.manual_seed(1)
            _, dataset = data.random_split(dataset, [train_set_size, valid_set_size], generator=torch_seed)
        input_shape=[1, 32, 32]
        n_samples=10000
        num_classes=100
    return dataset, num_classes, n_samples, input_shape

def eval_shift_data(model, dataset, batch_size, device, num_samples=1):
    if dataset == "SVHN":
        shift_dataset, num_classes, _, _ = get_dataset(dataset)
        shift_dataloader = DataLoader(shift_dataset, batch_size=batch_size)
    elif dataset == "CIFAR10":
        shift_dataset, num_classes, _, _ = get_dataset(dataset)
        shift_dataloader = DataLoader(shift_dataset, batch_size=batch_size)
    elif dataset == "CIFAR100":
        shift_dataset, num_classes, _, _ = get_dataset(dataset)
        shift_dataloader = DataLoader(shift_dataset, batch_size=batch_size)

    ece_overall = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='l1')
    mce_overall = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='max')
    accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    accuracy = accuracy.to(device)
    ece_overall = ece_overall.to(device)
    mce_overall = mce_overall.to(device)
    corruption_ece_dict = {}
    corruption_mce_dict = {}
    corruption_acc_dict = {}
    corruption_conf_dict = {}
    if dataset == "SVHN":
        with torch.no_grad():
            for i, rotation in enumerate(SVHN_ROTATIONS):
                ece_rotation = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='l1')
                mce_rotation = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='max')
                for j, (images, labels) in enumerate(shift_dataloader):
                    images = images.to(device)
                    images = rotate(images, angle=rotation)
                    labels = labels.to(device)
                    if num_samples==1:
                        preds = model(images)
                        probs = nn.functional.softmax(preds, dim=1)
                    else:
                        probs = model.forward_multisample(images, num_samples=num_samples)
                    acc = accuracy(probs, labels)
                    ece_overall.update(probs, labels)
                    mce_overall.update(probs, labels)
                    ece_rotation.update(probs, labels)
                    mce_rotation.update(probs, labels)
                ece_rotation_calc = ece_rotation.compute()
                mce_rotation_calc = mce_rotation.compute()
                conf = torch.cat(ece_rotation.confidences, dim=0).mean()
                acc = torch.cat(ece_rotation.accuracies, dim=0).mean()
                corruption_ece_dict[i+1] = ece_rotation_calc.to("cpu").numpy().tolist()*100
                corruption_mce_dict[i+1] = mce_rotation_calc.to("cpu").numpy().tolist()*100
                corruption_acc_dict[i+1] = acc.to("cpu").numpy().tolist()*100
                corruption_conf_dict[i+1] = conf.to("cpu").numpy().tolist()*100
    elif dataset == "CIFAR10":
        for corruption in CORRUPTIONS:
            data = np.load(CIFAR10_C_PATH + corruption + '.npy')
            targets = np.load(CIFAR10_C_PATH + 'labels.npy')
            for intensity in range(5):
                ece_corruption = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='l1')
                mce_corruption = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='max')
                shift_dataset.data = data[intensity*10000:(intensity+1)*10000]
                shift_dataset.targets = torch.LongTensor(targets[intensity*10000:(intensity+1)*10000])

                shift_dataloader = DataLoader(
                    shift_dataset,
                    batch_size=batch_size,
                    )
                with torch.no_grad():
                    for j, (images, labels) in enumerate(shift_dataloader):
                        images = images.to(device)
                        labels = labels.to(device)
                        if num_samples==1:
                            preds = model(images)
                            probs = nn.functional.softmax(preds, dim=1)
                        else:
                            probs = model.forward_multisample(images, num_samples=num_samples)
                        # print(probs.shape)
                        acc = accuracy(probs, labels)
                        ece_overall.update(probs, labels)
                        mce_overall.update(probs, labels)
                        ece_corruption.update(probs, labels)
                        mce_corruption.update(probs, labels)
                ece_corruption_calc = ece_corruption.compute()
                mce_corruption_calc = mce_corruption.compute()
                conf = torch.cat(ece_corruption.confidences, dim=0).mean()
                acc = torch.cat(ece_corruption.accuracies, dim=0).mean()
                if intensity not in corruption_ece_dict.keys():
                    corruption_ece_dict[intensity] = ece_corruption_calc.to("cpu").numpy().tolist()*100
                    corruption_mce_dict[intensity] = mce_corruption_calc.to("cpu").numpy().tolist()*100
                    corruption_acc_dict[intensity] = acc.to("cpu").numpy().tolist()*100
                    corruption_conf_dict[intensity] = conf.to("cpu").numpy().tolist()*100
                else:
                    corruption_ece_dict[intensity] += ece_corruption_calc.to("cpu").numpy().tolist()*100
                    corruption_mce_dict[intensity] += mce_corruption_calc.to("cpu").numpy().tolist()*100
                    corruption_acc_dict[intensity] += acc.to("cpu").numpy().tolist()*100
                    corruption_conf_dict[intensity] += conf.to("cpu").numpy().tolist()*100
        for key in corruption_ece_dict.keys():
            corruption_ece_dict[key] /= len(CORRUPTIONS)
            corruption_mce_dict[key] /= len(CORRUPTIONS)
            corruption_acc_dict[key] /= len(CORRUPTIONS)
            corruption_conf_dict[key] /= len(CORRUPTIONS)
    elif dataset == "CIFAR100":
        for corruption in CIFAR100_C_CORRUPTIONS:
            data = np.load(CIFAR100_C_PATH + corruption + '.npy')
            targets = np.load(CIFAR100_C_PATH + 'labels.npy')
            for intensity in range(5):
                shift_dataset.data = data[intensity*10000:(intensity+1)*10000]
                shift_dataset.targets = torch.LongTensor(targets[intensity*10000:(intensity+1)*10000])
                ece_corruption = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='l1')
                mce_corruption = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='max')
                shift_dataloader = DataLoader(
                    shift_dataset,
                    batch_size=batch_size,
                    )
                with torch.no_grad():
                    for j, (images, labels) in enumerate(shift_dataloader):
                        images = images.to(device)
                        labels = labels.to(device)
                        if num_samples==1:
                            preds = model(images)
                            probs = nn.functional.softmax(preds, dim=1)
                        else:
                            probs = model.forward_multisample(images, num_samples=num_samples)
                        acc = accuracy(probs, labels)
                        ece_overall.update(probs, labels)
                        mce_overall.update(probs, labels)
                        ece_corruption.update(probs, labels)
                        mce_corruption.update(probs, labels)
                ece_corruption_calc = ece_corruption.compute()
                mce_corruption_calc = mce_corruption.compute()
                conf = torch.cat(ece_corruption.confidences, dim=0).mean()
                acc = torch.cat(ece_corruption.accuracies, dim=0).mean()
                if intensity not in corruption_ece_dict.keys():
                    corruption_ece_dict[intensity] = ece_corruption_calc.to("cpu").numpy().tolist()*100
                    corruption_mce_dict[intensity] = mce_corruption_calc.to("cpu").numpy().tolist()*100
                    corruption_acc_dict[intensity] = acc.to("cpu").numpy().tolist()*100
                    corruption_conf_dict[intensity] = conf.to("cpu").numpy().tolist()*100
                else:
                    corruption_ece_dict[intensity] += ece_corruption_calc.to("cpu").numpy().tolist()*100
                    corruption_mce_dict[intensity] += mce_corruption_calc.to("cpu").numpy().tolist()*100
                    corruption_acc_dict[intensity] += acc.to("cpu").numpy().tolist()*100
                    corruption_conf_dict[intensity] += conf.to("cpu").numpy().tolist()*100
        for key in corruption_ece_dict.keys():
            corruption_ece_dict[key] /= len(CORRUPTIONS)
            corruption_mce_dict[key] /= len(CORRUPTIONS)
            corruption_acc_dict[key] /= len(CORRUPTIONS)
            corruption_conf_dict[key] /= len(CORRUPTIONS)
    acc = accuracy.compute()
    conf = torch.cat(ece_overall.confidences, dim=0).mean()
    ece_overall_calc = ece_overall.compute()
    mce_overall_calc = mce_overall.compute()

    return ece_overall_calc, mce_overall_calc, acc, conf, corruption_ece_dict, corruption_mce_dict, corruption_acc_dict, corruption_conf_dict

    
def eval_ood_data(model, dataset, batch_size, device, OOD_y_preds_logits, OOD_labels, alpha=1.0, num_samples=1):
    ood_dataloaders, num_classes, ood_dataset = get_ood_datasets(dataset, batch_size)
    OOD_label = []
    OOD_y_preds_logit = []
    auroc_calc = []
    fpr95_calc = []
    confidences = []
    entropies = []

    with torch.no_grad():
        for i, ood_test_dataloader in enumerate(ood_dataloaders):
            OOD_label.append(OOD_labels)
            OOD_y_preds_logit.append(OOD_y_preds_logits)
            all_confidences = []
            all_entropies = []
            cal = Calibration(num_classes[0],n_bins = 35)
            for j, (images, labels) in enumerate(ood_test_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                if num_samples==1:
                    preds = model(images)
                    probs = nn.functional.softmax(preds, dim=1)
                else:
                    probs = model.forward_multisample(images, num_samples=num_samples)
                cal.update(probs, labels)
                max_predictions = torch.max(probs.data, 1).values
                OOD_y_preds_logit[i].append(max_predictions)
                OOD_label[i].append(torch.tensor([0]*len(labels)))

                confidences_batch = torch.max(probs, dim=1).values
                entropies_batch = -torch.sum(probs * torch.log(probs), dim=1)
                all_confidences.append(confidences_batch)
                all_entropies.append(entropies_batch)
            
            print('#dataset{}'.format(i))
            cal.compute()
            all_confidences = torch.cat(all_confidences)
            all_entropies = torch.cat(all_entropies)
            confidences.append(all_confidences.mean())
            entropies.append(all_entropies.mean())
            OOD_label[i] = torch.cat(OOD_label[i])
            OOD_y_preds_logit[i] = torch.cat(OOD_y_preds_logit[i])
            auroc_calc.append(auroc(OOD_y_preds_logit[i].to("cpu").numpy().tolist(), OOD_label[i].to("cpu").numpy().tolist()))
            fpr95_calc.append(fpr_at_95_tpr(OOD_y_preds_logit[i].to("cpu").numpy().tolist(), OOD_label[i].to("cpu").numpy().tolist()))

    return ood_dataset, auroc_calc, fpr95_calc, confidences, entropies

def get_ood_datasets(dataset, batch_size):
    if dataset == "SVHN":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))])  
        ood_test_dataloaders = [
        DataLoader(CIFAR10(os.getcwd()+"/data/", download=True, transform=transform, train=False), batch_size=batch_size),
        DataLoader(CIFAR100(os.getcwd()+"/data/", download=True, transform=transform, train=False), batch_size=batch_size)
        ]
        num_classes = [10, 10, 100]
        ood_dataset = ['CIFAR10', 'CIFAR100']
        return ood_test_dataloaders, num_classes, ood_dataset
    elif dataset == "CIFAR10":
        ood_test_dataloaders = [
            DataLoader(SVHN(os.getcwd()+"/data/", download=True, transform=transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.247, 0.243, 0.261))]), split="test"), batch_size=batch_size),
            DataLoader(CIFAR100(os.getcwd()+"/data/", download=True, transform=transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.247, 0.243, 0.261))]), train=False), batch_size=batch_size)
        ]
        num_classes = [10,10, 100]
        ood_dataset = ['SVHN', 'CIFAR100']
        return ood_test_dataloaders, num_classes, ood_dataset
    elif dataset == "CIFAR100":
        transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))])
        ood_test_dataloaders = [
            DataLoader(SVHN(os.getcwd()+"/data/", download=True, transform=transform, split="test"), batch_size=batch_size),
            DataLoader(CIFAR10(os.getcwd()+"/data/", download=True, transform=transform, train=False), batch_size=batch_size)
        ]
        num_classes = [100, 10, 10]
        ood_dataset = ['SVHN', 'CIFAR10']
        return ood_test_dataloaders, num_classes, ood_dataset
    
def produce_Ew(label, num_classes):
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
