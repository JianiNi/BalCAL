from sklearn import metrics
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
#Metrics from https://github.com/runame/laplace-refinement/blob/main/utils/metrics.py
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.classification import Accuracy

def nll(y_pred, y_true):
    """
    Mean Categorical negative log-likelihood. `y_pred` is a probability vector.
    """
    try:
        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.cpu().numpy()
    finally:
        return metrics.log_loss(y_true, y_pred)

def brier(y_pred, y_true):
    try:
        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.cpu().numpy()
    finally:
        def one_hot(targets, nb_classes):
            res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
            return res.reshape(list(targets.shape)+[nb_classes])

        return metrics.mean_squared_error(y_pred, one_hot(y_true, y_pred.shape[-1]))

def get_auroc(py_in, py_out):
    py_in, py_out = py_in.cpu().numpy(), py_out.cpu().numpy()
    labels = np.zeros(len(py_in)+len(py_out), dtype='int32')
    labels[:len(py_in)] = 1
    examples = np.concatenate([py_in.max(1), py_out.max(1)])
    return metrics.roc_auc_score(labels, examples)


def get_fpr95(py_in, py_out):
    py_in, py_out = py_in.cpu().numpy(), py_out.cpu().numpy()
    conf_in, conf_out = py_in.max(1), py_out.max(1)
    tpr = 95
    perc = np.percentile(conf_in, 100-tpr)
    fp = np.sum(conf_out >=  perc)
    fpr = np.sum(conf_out >=  perc) / len(conf_out)
    return fpr.item(), perc.item()

def get_calib(pys, y_true, M=10):
    # Put the confidence into M bins
    pys, y_true = pys.cpu().numpy(), y_true.cpu().numpy()
    _, bins = np.histogram(pys, M, range=(0, 1))

    labels = pys.argmax(1)
    confs = np.max(pys, axis=1)
    conf_idxs = np.digitize(confs, bins)

    # Accuracy and avg. confidence per bin
    accs_bin = []
    confs_bin = []
    nitems_bin = []

    for i in range(M):
        labels_i = labels[conf_idxs == i]
        y_true_i = y_true[conf_idxs == i]
        confs_i = confs[conf_idxs == i]

        acc = np.nan_to_num(np.mean(labels_i == y_true_i), 0)
        conf = np.nan_to_num(np.mean(confs_i), 0)

        accs_bin.append(acc)
        confs_bin.append(conf)
        nitems_bin.append(len(labels_i))

    accs_bin, confs_bin = np.array(accs_bin), np.array(confs_bin)
    nitems_bin = np.array(nitems_bin)

    ECE = np.average(np.abs(confs_bin-accs_bin), weights=nitems_bin/nitems_bin.sum())
    MCE = np.max(np.abs(accs_bin - confs_bin))

    # In percent
    ECE, MCE = ECE*100, MCE*100

    return accs_bin, confs_bin, ECE.item(), MCE.item()

def calculate_ECE_per_bin(y_preds, y_targets, n_bins=10, ECE_type="1"):
    #https://arxiv.org/pdf/1706.04599.pdf

    y_preds, y_targets = y_preds.cpu(), y_targets.cpu()
    
    if ECE_type=="K":
        output_dims = y_preds.shape[1]

        #We flatten probability list which previously had shape (n, K) where n=#test_samples, K=#classes
        y_preds = torch.flatten(y_preds)

        #We create a list of labels with shape (n*K) using repeat interleave. That gives us a label for each
        #p_{n,i}. If previous labels were [0, 1, 0] and K=2, then new list is [0, 0, 1, 1, 0, 0]
        labels = torch.repeat_interleave(y_targets, output_dims)

        #Now we simply need to make a list for comparing to the labels, i.e. [0, 1, 0, 1, .... , 0, 1]
        #until we have (n*K) length list
        probability_labels = torch.tensor(np.arange(output_dims))
        probability_labels = probability_labels.repeat(len(y_targets))
    elif ECE_type=="1":
        prob_list, probability_labels = torch.max(y_preds, dim=1)
        labels = y_targets

    
    #We create the bin limits.
    bin_limits = torch.arange(0,1, 1/n_bins)
    
    ECE_dict = {}
    ECE_dict_count = {}
    weighted_ECE = 0
    
    #We now iterate over the number of bins in ECE calc.
    for i in range(n_bins):
        #If statement to get the correct "bin limits" (i.e. in first bin case limits of bin are (0.0, 0.1)
        #if #bins=10).
        if i != n_bins-1:
            #Here we get all $p_{i, n}$ that lie within this specific bin limits, i.e. first bin
            #we get all indeces of $p_{i, n}$ where probability is in range [0, 0.1) if #bins=10.
            bin_indeces = torch.where((prob_list >= bin_limits[i])&(prob_list < bin_limits[i+1]), True, False)
            bin_mid = (bin_limits[i]+bin_limits[i+1])/2
        else:
            #Same idea as previous if statement, just in the final bin case.
            bin_indeces = torch.where(prob_list >= bin_limits[i], True, False)
            bin_mid = (bin_limits[i]+1)/2
        #Get all prbabilites that "go" in this bin.
        bin_probs = prob_list[bin_indeces]
        #Mean predicted probabilities (confidence) in bin.
        mean_prob = torch.mean(bin_probs)
        
        #Get predictions and labels of $p_{i, n}$ that belong to this bin.
        bin_preds = probability_labels[bin_indeces]
        bin_labels = labels[bin_indeces]
        #Check to ensure that there actaully are samples in this bin, otherwise we get nan.
        if len(bin_probs) != 0:
            #Compute accuracy inside this specific bin.
            accuracy = torch.sum(bin_preds==bin_labels)/len(bin_probs)
            
            #Save accuracy within specific bin in a dictionary. 
            ECE_dict[str(round(bin_mid.item(), 4))] = (accuracy).item()
            ECE_dict_count[str(round(bin_mid.item(), 4))+ " samples in bin"] = len(bin_probs)
            
            #Compute ECE by subtracting accuracy in bin with confidence in bin, and weight it according
            #to number of $p_{i, n}$ that are in this bin out of (n*K).
            weighted_ECE += torch.abs(accuracy-mean_prob).item()*len(bin_probs)/len(prob_list)
        else:
            ECE_dict[str(round(bin_mid.item(), 4))] = 0
    return weighted_ECE, ECE_dict

class Calibration(object):
    def __init__(self,num_classes: int, n_bins: int = 15):
        super(Calibration, self).__init__()
        self.num_classes = num_classes
        self.n_bins = n_bins
        self.confidences = []
        self.accuracies = []
        self.targets = []
        self.preds = []

    def reset(self):
        self.confidences = []
        self.accuracies = []
        self.targets = []
        self.preds = []

    def update(self, pred, target):
        confidences, accuracies = self.handel(pred, target)
        self.confidences.append(confidences.float())
        self.accuracies.append(accuracies.float())
        self.targets.append(target)
        self.preds.append(pred)

    def compute(self):
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        ece, mce, m_ce = self.calibrate(confidences,accuracies)
        return confidences.mean(), accuracies.mean(), ece, mce, m_ce
    
    def switch(self):
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        targets  = dim_zero_cat(self.targets)
        preds  = dim_zero_cat(self.preds)
        return confidences, accuracies, preds, targets
    
    def handel(self, preds, target):
        if not torch.all((preds >= 0) * (preds <= 1)):
            preds = preds.softmax(1)
        if len(target.size()) > 1:

            confidences, predictions = preds.max(dim=1)
            target = target.argmax(dim=1)
        else:
            confidences, predictions = preds.max(dim=1)
        accuracies = predictions.eq(target)
        accuracies = accuracies.to(dtype=confidences.dtype)
        return confidences, accuracies

    def compute1(self,confidences,accuracies):
        confidence = dim_zero_cat(confidences)
        accuracy = dim_zero_cat(accuracies)
        ece, mce, m_ce = self.calibrate(confidence,accuracy)
        return ece
    
    def calibrate(self,confidences,accuracies):
        if isinstance(self.n_bins, int):
            self.n_bins = torch.linspace(0, 1, self.n_bins + 1, dtype=confidences.dtype, device=confidences.device)
        acc_bin = torch.zeros(len(self.n_bins), device=confidences.device, dtype=confidences.dtype)
        conf_bin = torch.zeros(len(self.n_bins), device=confidences.device, dtype=confidences.dtype)
        count_bin = torch.zeros(len(self.n_bins), device=confidences.device, dtype=confidences.dtype)

        indices = torch.bucketize(confidences, self.n_bins, right=True) - 1
        count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))
        

        conf_bin.scatter_add_(dim=0, index=indices, src=confidences)
        conf_bin = torch.nan_to_num(conf_bin / count_bin)

        acc_bin.scatter_add_(dim=0, index=indices, src=accuracies)
        acc_bin = torch.nan_to_num(acc_bin / count_bin)
        prop_bin = count_bin / count_bin.sum()

        gaps = torch.abs(acc_bin - conf_bin)
        ece = torch.sum(gaps * prop_bin)
        mce = torch.max(gaps)
        m_ce = torch.min(gaps)

        return ece, mce, m_ce


class AdaptiveECELoss(nn.Module):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15):
        super(AdaptiveECELoss, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, logits, labels):
        if not torch.all((logits >= 0) * (logits <= 1)):
            logits = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(logits, 1)
        accuracies = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece.squeeze()
    
class ClasswiseECELoss(nn.Module):
    '''
    Compute Classwise ECE
    '''
    def __init__(self, n_bins=15):
        super(ClasswiseECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        if not torch.all((logits >= 0) * (logits <= 1)):
            logits = F.softmax(logits, dim=1)
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = logits[:, i]
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce