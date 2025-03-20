import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from scipy.special import lambertw
import numpy as np

def dot_loss(output, label, cur_M, criterion, H_length, reg_lam=0):
    target = cur_M[:, label].T ## B, d  output: B, d
    if criterion == 'dot_loss':
        loss = - torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1).mean()
    elif criterion == 'reg_dot_loss':
        dot = torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1) #+ classifier.module.bias[label].view(-1)

        with torch.no_grad():
            M_length = torch.sqrt(torch.sum(target ** 2, dim=1, keepdims=False))
        loss = (1/2) * torch.mean(((dot-(M_length * H_length)) ** 2) / H_length)

        if reg_lam > 0:
            reg_Eh_l2 = torch.mean(torch.sqrt(torch.sum(output ** 2, dim=1, keepdims=True)))
            loss = loss + reg_Eh_l2*reg_lam

    return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self,reduction="mean"):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self,input, target):
        # return F.nll_loss(torch.log(input + 1e-20), target,reduction=self.reduction)
        return F.nll_loss(torch.log(input.softmax(dim=1) + 1e-20), target,reduction=self.reduction)

class ESD(nn.Module):
    def __init__(self,lamda,device):
        super(ESD, self).__init__()
        self.device = device
        self.lamda=lamda
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self,input, target):
        confidence1, prediction_calset = torch.max(input, dim = 1)
        correct = target.eq(prediction_calset)
        N1 = len(confidence1) 
        val = correct.float() - confidence1 
        val = val.view(1,N1) 
        mask = torch.ones(N1,N1) - torch.eye(N1)
        mask = mask.to(self.device)
        confidence1_matrix = confidence1.expand(N1,N1)
        temp = (confidence1.view(1,N1).T).expand(N1,N1)
        tri = torch.le(confidence1_matrix,temp).float() 
        val_matrix = val.expand(N1,N1)
        x_matrix = torch.mul(val_matrix,tri)*mask
        mean_row = torch.sum(x_matrix, dim = 1)/(N1-1)
        x_matrix_squared = torch.mul(x_matrix, x_matrix)
        var = 1/(N1-2) * torch.sum(x_matrix_squared,dim=1) - (N1-1)/(N1-2) * torch.mul(mean_row,mean_row)
        d_k_sq_vector = torch.mul(mean_row, mean_row) - var/(N1-1)
        reg_loss = torch.sum(d_k_sq_vector)/N1
        loss_ce = self.cross_entropy(input, target)
        return loss_ce+self.lamda * torch.sqrt(torch.nn.functional.relu(reg_loss))

class CPCLoss(nn.Module):
    def __init__(self, lambd_bdc=1.0, lambd_bec=1.0, ignore_index=-100):
        super().__init__()
        self.lambd_bdc = lambd_bdc
        self.lambd_bec = lambd_bec
        self.ignore_index = ignore_index
        self.EPS = 1e-10
        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_bdc", "loss_bec"

    def bdc(self, logits, targets_one_hot):
        # 1v1 Binary Discrimination Constraints (BDC)
        logits_y = logits[targets_one_hot == 1].view(logits.size(0), -1)
        logits_rest = logits[targets_one_hot == 0].view(logits.size(0), -1)
        loss_bdc = - F.logsigmoid(logits_y - logits_rest).sum() / (logits.size(1) - 1) / logits.size(0)

        return loss_bdc

    def bec(self, logits, targets_one_hot):
        # Binary Exclusion COnstraints (BEC)
        logits_rest = logits[targets_one_hot == 0].view(logits.size(0), -1)
        diff = logits_rest.unsqueeze(2) - logits_rest.unsqueeze(1)
        loss_bec = - torch.sum(
            0.5 * F.logsigmoid(diff + self.EPS)
            / (logits.size(1) - 1) / (logits.size(1) - 2) / logits.size(0)
        )

        return loss_bec

    def one_hot(x, num_classes, on_value=1, off_value=0, device='cuda'):
        x = x.long().view(-1, 1)
        return torch.full(
            (x.size()[0], num_classes), off_value, device=device
        ).scatter_(1, x, on_value)

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)

        targets_one_hot = F.one_hot(targets, inputs.size(1))
        loss_bdc = self.bdc(inputs, targets_one_hot)
        loss_bec = self.bec(inputs, targets_one_hot)

        loss = loss_ce + self.lambd_bdc * loss_bdc + self.lambd_bec * loss_bec

        return loss

class ACLS(nn.Module):

    def __init__(self,
                 pos_lambda: float = 1.0,
                 neg_lambda: float = 0.1,
                 alpha: float = 0.1,    
                 margin: float = 10.0,
                 num_classes: int = 200,
                 ignore_index: int = -100):
        super().__init__()
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.alpha = alpha
        self.margin = margin
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "reg"

    def get_reg(self, inputs, targets):
        max_values, indices = inputs.max(dim=1)
        max_values = max_values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        indicator = (max_values.clone().detach() == inputs.clone().detach()).float()

        batch_size, num_classes = inputs.size()
        num_pos = batch_size * 1.0
        num_neg = batch_size * (num_classes - 1.0)

        neg_dist = max_values.clone().detach() - inputs
        
        pos_dist_margin = F.relu(max_values - self.margin)
        neg_dist_margin = F.relu(neg_dist - self.margin)

        pos = indicator * pos_dist_margin ** 2
        neg = (1.0 - indicator) * (neg_dist_margin ** 2)

        reg = self.pos_lambda * (pos.sum() / num_pos) + self.neg_lambda * (neg.sum() / num_neg)
        return reg


    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        loss_ce = self.cross_entropy(inputs, targets)

        loss_reg = self.get_reg(inputs, targets)
        loss = loss_ce + self.alpha * loss_reg

        return loss

class MMCE(nn.Module):
    """
    Computes MMCE_m loss.
    """
    def __init__(self, device, lamda=1.0):
        super(MMCE, self).__init__()
        self.device = device
        self.lamda = lamda

    def torch_kernel(self, matrix):
        return torch.exp(-1.0*torch.abs(matrix[:, :, 0] - matrix[:, :, 1])/(0.4))

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        target = target.view(-1) #For CIFAR-10 and CIFAR-100, target.shape is [N] to begin with

        predicted_probs = F.softmax(input, dim=1)
        predicted_probs, pred_labels = torch.max(predicted_probs, 1)
        correct_mask = torch.where(torch.eq(pred_labels, target),
                          torch.ones(pred_labels.shape).to(self.device),
                          torch.zeros(pred_labels.shape).to(self.device))

        c_minus_r = correct_mask - predicted_probs

        dot_product = torch.mm(c_minus_r.unsqueeze(1),
                        c_minus_r.unsqueeze(0))

        prob_tiled = predicted_probs.unsqueeze(1).repeat(1, predicted_probs.shape[0]).unsqueeze(2)
        prob_pairs = torch.cat([prob_tiled, prob_tiled.permute(1, 0, 2)],
                                    dim=2)

        kernel_prob_pairs = self.torch_kernel(prob_pairs)

        numerator = dot_product*kernel_prob_pairs
        #return torch.sum(numerator)/correct_mask.shape[0]**2

        # ce
        ce = F.cross_entropy(input, target)
        return ce + self.lamda * torch.sum(numerator) / torch.pow(torch.tensor(correct_mask.shape[0]).type(torch.FloatTensor), 2)


class MMCE_weighted(nn.Module):
    """
    Computes MMCE_w loss.
    """
    def __init__(self, device, lamda=1.0):
        super(MMCE_weighted, self).__init__()
        self.device = device
        self.lamda = lamda

    def torch_kernel(self, matrix):
        return torch.exp(-1.0*torch.abs(matrix[:, :, 0] - matrix[:, :, 1])/(0.4))

    def get_pairs(self, tensor1, tensor2):
        correct_prob_tiled = tensor1.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2)
        incorrect_prob_tiled = tensor2.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)

        correct_prob_pairs = torch.cat([correct_prob_tiled, correct_prob_tiled.permute(1, 0, 2)],
                                    dim=2)
        incorrect_prob_pairs = torch.cat([incorrect_prob_tiled, incorrect_prob_tiled.permute(1, 0, 2)],
                                    dim=2)

        correct_prob_tiled_1 = tensor1.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)
        incorrect_prob_tiled_1 = tensor2.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2)

        correct_incorrect_pairs = torch.cat([correct_prob_tiled_1, incorrect_prob_tiled_1.permute(1, 0, 2)],
                                    dim=2)
        return correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs

    def get_out_tensor(self, tensor1, tensor2):
        return torch.mean(tensor1*tensor2)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        target = target.view(-1)  #For CIFAR-10 and CIFAR-100, target.shape is [N] to begin with

        predicted_probs = F.softmax(input, dim=1)
        predicted_probs, predicted_labels = torch.max(predicted_probs, 1)

        correct_mask = torch.where(torch.eq(predicted_labels, target),
                                    torch.ones(predicted_labels.shape).to(self.device),
                                    torch.zeros(predicted_labels.shape).to(self.device))

        k = torch.sum(correct_mask).type(torch.int64)
        k_p = torch.sum(1.0 - correct_mask).type(torch.int64)
        cond_k = torch.where(torch.eq(k,0),torch.tensor(0).to(self.device),torch.tensor(1).to(self.device))
        cond_k_p = torch.where(torch.eq(k_p,0),torch.tensor(0).to(self.device),torch.tensor(1).to(self.device))
        k = torch.max(k, torch.tensor(1).to(self.device))*cond_k*cond_k_p + (1 - cond_k*cond_k_p)*2 
        k_p = torch.max(k_p, torch.tensor(1).to(self.device))*cond_k_p*cond_k + ((1 - cond_k_p*cond_k)*
                                            (correct_mask.shape[0] - 2))


        correct_prob, _ = torch.topk(predicted_probs*correct_mask, k)
        incorrect_prob, _ = torch.topk(predicted_probs*(1 - correct_mask), k_p)

        correct_prob_pairs, incorrect_prob_pairs,\
               correct_incorrect_pairs = self.get_pairs(correct_prob, incorrect_prob)

        correct_kernel = self.torch_kernel(correct_prob_pairs)
        incorrect_kernel = self.torch_kernel(incorrect_prob_pairs)
        correct_incorrect_kernel = self.torch_kernel(correct_incorrect_pairs)  

        sampling_weights_correct = torch.mm((1.0 - correct_prob).unsqueeze(1), (1.0 - correct_prob).unsqueeze(0))

        correct_correct_vals = self.get_out_tensor(correct_kernel,
                                                          sampling_weights_correct)
        sampling_weights_incorrect = torch.mm(incorrect_prob.unsqueeze(1), incorrect_prob.unsqueeze(0))

        incorrect_incorrect_vals = self.get_out_tensor(incorrect_kernel,
                                                          sampling_weights_incorrect)
        sampling_correct_incorrect = torch.mm((1.0 - correct_prob).unsqueeze(1), incorrect_prob.unsqueeze(0))

        correct_incorrect_vals = self.get_out_tensor(correct_incorrect_kernel,
                                                          sampling_correct_incorrect)

        correct_denom = torch.sum(1.0 - correct_prob)
        incorrect_denom = torch.sum(incorrect_prob)

        m = torch.sum(correct_mask)
        n = torch.sum(1.0 - correct_mask)
        mmd_error = 1.0/(m*m + 1e-5) * torch.sum(correct_correct_vals) 
        mmd_error += 1.0/(n*n + 1e-5) * torch.sum(incorrect_incorrect_vals)
        mmd_error -= 2.0/(m*n + 1e-5) * torch.sum(correct_incorrect_vals)

        # # ce
        ce = F.cross_entropy(input, target)

        return ce + self.lamda * torch.max((cond_k*cond_k_p).type(torch.FloatTensor).to(self.device).detach()*torch.sqrt(mmd_error + 1e-10), torch.tensor(0.0).to(self.device))

class MDCA(nn.Module):
    def __init__(self):
        super(MDCA, self).__init__()
        self.ls = LabelSmoothingCrossEntropy()

    @property
    def names(self):
        return "loss", "loss_ce",  "loss_mdca"

    def forward(self, output, target):
        # output = torch.softmax(output, dim=1)
        # # [batch, classes]
        # loss_mdca = torch.tensor(0.0).cuda()
        # batch, classes = output.shape
        # for c in range(classes):
        #     avg_count = (target == c).float().mean()
        #     avg_conf = torch.mean(output[:,c])
        #     loss_mdca += torch.abs(avg_conf - avg_count)
        # denom = classes
        # loss_mdca /= denom
        # loss_mdca *= 1.0
        # loss_ce = self.ls(output, target)
        # loss = loss_ce + loss_mdca
        
        probs = torch.softmax(output, dim=1)

        batch_size, num_classes = probs.shape

        avg_conf = torch.mean(probs, dim=0)

        avg_count = torch.zeros(num_classes, device=output.device)
        for c in range(num_classes):
            avg_count[c] = (target == c).float().mean()

        loss_mdca = torch.mean(torch.abs(avg_conf - avg_count))

        return loss_mdca


def get_gamma(p=0.2):
    '''
    Get the gamma for a given pt where the function g(p, gamma) = 1
    '''
    y = ((1-p)**(1-(1-p)/(p*np.log(p)))/(p*np.log(p)))*np.log(1-p)
    gamma_complex = (1-p)/(p*np.log(p)) + lambertw(-y + 1e-12, k=-1)/np.log(1-p)
    gamma = np.real(gamma_complex) #gamma for which p_t > p results in g(p_t,gamma)<1
    return gamma

ps = [0.2, 0.5]
gammas = [5.0, 3.0]
i = 0
gamma_dic = {}
for p in ps:
    gamma_dic[p] = gammas[i]
    i += 1

class FocalLossAdaptive(nn.Module):
    def __init__(self, gamma=0, ignore_index=-100, size_average=False, device=None):
        super(FocalLossAdaptive, self).__init__()
        self.size_average = size_average
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.device = device

    def get_gamma_list(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            if (pt_sample >= 0.5):
                gamma_list.append(self.gamma)
                continue
            # Choosing the gamma for the sample
            for key in sorted(gamma_dic.keys()):
                if pt_sample < key:
                    gamma_list.append(gamma_dic[key])
                    break
        return torch.tensor(gamma_list).to(self.device)

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        if self.ignore_index >= 0:
            index = torch.nonzero(target.squeeze() != self.ignore_index).squeeze()
            input = input[index, :]
            target = target[index, :]

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        gamma = self.get_gamma_list(pt)
        loss = -1 * (1-pt)**gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, ignore_index=-100, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        if self.ignore_index >= 0:
            index = torch.nonzero(target.squeeze() != self.ignore_index).squeeze()
            input = input[index, :]
            target = target[index, :]

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class PenaltyEntropy(nn.Module):
    """Regularizing neural networks by penalizing confident output distributions, 2017. <https://arxiv.org/pdf/1701.06548>

        loss = CE - alpha * Entropy(p)
    """
    def __init__(self, alpha=1.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.EPS = 1e-10

    @property
    def names(self):
        return "loss", "loss_ce", "loss_ent"

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        # cross entropy
        loss_ce = F.cross_entropy(inputs, targets)

        # entropy
        prob = F.log_softmax(inputs, dim=1).exp()
        prob = torch.clamp(prob, self.EPS, 1.0 - self.EPS)
        ent = - prob * torch.log(prob)
        loss_ent = ent.mean()

        loss = loss_ce - self.alpha * loss_ent

        return loss

class LogitMarginL1(nn.Module):
    """Add marginal penalty to logits:
        CE + alpha * max(0, max(l^n) - l^n - margin)

    Args:
        margin (float, optional): The margin value. Defaults to 10.
        alpha (float, optional): The balancing weight. Defaults to 0.1.
        ignore_index (int, optional):
            Specifies a target value that is ignored
            during training. Defaults to -100.

        The following args are related to balancing weight (alpha) scheduling.
        Note all the results presented in CHM paper are obtained without the scheduling strategy.
        So it's fine to ignore if you don't want to try it.

        schedule (str, optional):
            Different stragety to schedule the balancing weight alpha or not:
            "" | add | multiply | step. Defaults to "" (no scheduling).
            To activate schedule, you should call function
            `schedula_alpha` every epoch in yCHM training code.
        mu (float, optional): scheduling weight. Defaults to 0.
        max_alpha (float, optional): Defaults to 100.0.
        step_size (int, optional): The step size for updating alpha. Defaults to 100.
    """
    def __init__(self,
                 margin: float = 10,
                 alpha: float = 0.1,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size

        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin = F.relu(diff-self.margin).mean()
        loss = loss_ce + self.alpha * loss_margin

        return loss



class LogitMarginPlus(nn.Module):
    """Add marginal penalty to logits:
        CE + alpha * max(0, max(l^n) - l^n - margin)
    """
    def __init__(self,
                 num_classes,
                 margin=6,
                 alpha=0.1,
                 ignore_index=-100,
                 gamma=1.1,
                 tao=1.1,
                 lambd_min: float = 1e-6,
                 lambd_max: float = 1e6,
                 step_size=1):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.tao = tao
        self.lambd_min = lambd_min
        self.lambd_max = lambd_max
        self.step_size = step_size

        # alpha for each class
        self.lambd = self.alpha * torch.ones(self.num_classes, requires_grad=False).cuda()
        self.prev_score, self.curr_score = (
            torch.zeros(self.num_classes, requires_grad=False).cuda(),
            torch.zeros(self.num_classes, requires_grad=False).cuda()
        )

        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def reset_update_lambd(self):
        self.prev_score, self.curr_score = self.curr_score, torch.zeros(self.num_classes).cuda()
        self.count = 0

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)

        diff = self.get_diff(inputs)
        # loss_margin = torch.clamp(diff - self.margin, min=0).mean()
        loss_margin = F.relu(diff-self.margin)
        loss_margin = torch.einsum("ik,k->ik", loss_margin, self.lambd).mean()

        # loss = loss_ce + self.alpha * loss_margin
        loss = loss_ce + loss_margin

        return loss

    def update_lambd(self, logits):
        diff = self.get_diff(logits)
        loss_margin = F.relu(diff-self.margin)
        loss_margin = torch.einsum("ik,k->ik", loss_margin, self.lambd).sum(dim=0)

        self.curr_score += loss_margin
        self.count += logits.shape[0]

    def set_lambd(self, epoch):
        self.curr_score = self.curr_score / self.count
        if dist.is_initialized():
            self.curr_score = reduce_tensor(self.curr_score, dist.get_world_size())
        if (epoch + 1) % self.step_size == 0 and self.prev_score.sum() != 0:
            self.lambd = torch.where(
                self.curr_score > self.prev_score * self.tao,
                self.lambd * self.gamma,
                self.lambd
            )
            self.lambd = torch.where(
                self.curr_score < self.prev_score / self.tao,
                self.lambd / self.gamma,
                self.lambd
            )
            self.lambd = torch.clamp(self.lambd, min=self.lambd_min, max=self.lambd_max).detach()

    def get_lambd_metric(self):
        return self.lambd.mean().item(), self.lambd.max().item()



class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, alpha=0.2, ignore_index=-100, reduction="mean"):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
            target = target.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(target != self.ignore_index).squeeze()
            input = input[index, :]
            target = target[index]

        confidence = 1. - self.alpha
        logprobs = F.log_softmax(input, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.alpha * smooth_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt

