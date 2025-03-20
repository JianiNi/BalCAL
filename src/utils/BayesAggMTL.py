from typing import Any, List, Tuple, Union, Dict
import torch
import torch.nn.functional as F
import logging
import torch.nn as nn
from tqdm import trange
import warnings


MAX_SIZE = 10000

# Adapted from: https://github.com/cornellius-gp/linear_operator/blob/main/linear_operator/utils/cholesky.py
def psd_safe_cholesky(A, upper=False, out=None, jitter=None, max_tries=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted,
            uses settings.cholesky_jitter.value()
        :attr:`max_tries` (int, optional):
            Number of attempts (with successively increasing jitter) to make before raising an error.
    """
    L = _psd_safe_cholesky(A, out=out, jitter=jitter, max_tries=max_tries)
    if torch.any(torch.isnan(L)):
        L = _psd_safe_cholesky(A.cpu(), out=out,
                               jitter=jitter, max_tries=max_tries).to(A.device)

    if upper:
        if out is not None:
            out = out.transpose_(-1, -2)
        else:
            L = L.mT
    return L


def _psd_safe_cholesky(A, out=None, jitter=None, max_tries=None):

    L, info = torch.linalg.cholesky_ex(A, out=out)

    if not torch.any(info):
        return L

    isnan = torch.isnan(A)
    if isnan.any():
        raise warnings.warn(f"cholesky_cpu: {isnan.sum().item()} of "
                            f"{A.numel()} elements of the {A.shape} tensor are NaN.")
        exit(0)

    if jitter is None:
        jitter = 1e-7 if A.dtype == torch.float32 else 1e-9
    if max_tries is None:
        max_tries = 25
    Aprime = A.clone()
    jitter_prev = 0
    for i in range(max_tries):
        jitter_new = jitter * (10**i)
        # add jitter only where needed
        diag_add = ((info > 0) * (jitter_new - jitter_prev)).unsqueeze(-1).expand(*Aprime.shape[:-1])
        Aprime.diagonal(dim1=-1, dim2=-2).add_(diag_add)
        jitter_prev = jitter_new
        warnings.warn(f"A not p.d., added jitter of {jitter_new:.1e} to the diagonal")
        L, info = torch.linalg.cholesky_ex(Aprime, out=out)

        if not torch.any(info):
            return L
    raise ValueError(f"Matrix not positive definite after repeatedly adding jitter up to {jitter_new:.1e}.")


class BayesAggMTL:
    def __init__(self,
                 num_tasks: int,
                 n_outputs_per_task_group: Union[Tuple[int], List[int]],
                 task_types: Union[List[str], Tuple[str]],
                 agg_scheme_hps: dict = {},
                 cls_hps: dict = {},
                 ):
        """
        Main class for running BayesMTL in all flavCHMs. Note that task types should be grouped together
        in construction and when running the model
        :param num_tasks:
        :param n_outputs_per_task_group: a sequence of number of outputs per task group; [n_tasks * n_outputs] for
        regression, [n_tasks] for binary classification tasks, [n_classes] for multi-class classification task
        :param task_types: either 'regression', 'binary_tasks', or 'multiclass'
        :param agg_scheme: How to combine the task statistics
        :param agg_scheme_hps:
        :param cls_hps:
        :param reg_hps:
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.n_outputs_per_task_group = n_outputs_per_task_group
        self.task_types = task_types
        self.posterior_modules = []
        self.moments_modules = []
        for i, task_type in enumerate(task_types):
            if task_type == 'multiclass':
                self.posterior_modules.append(
                        LastLayerPosteriorTaylorApprox(gamma=cls_hps['gamma'], setting='multiclass')
                )
                self.moments_modules.append(
                        ApproxMomentsMultiClass(num_classes=n_outputs_per_task_group[i],
                                                sqrt_power=cls_hps['sqrt_power'],
                                                num_mc_samples=cls_hps['num_mc_samples'])
                )
            else:
                raise Exception("task type not recognized")

        self.agg_scheme = GaussianAgg(num_tasks=self.num_tasks)

    @staticmethod
    def backward_last_layer(losses: torch.Tensor,
                            last_layer_params: Union[List[torch.nn.parameter.Parameter],
                                                     Tuple[torch.nn.parameter.Parameter]]):

        grad = torch.autograd.grad(
            losses.sum(),
            last_layer_params,
            retain_graph=True,
            allow_unused=True
        )

        for p, g in zip(last_layer_params, grad):
            p.grad = g

    def backward(self,
                 losses: torch.Tensor,
                 last_layer_params: Union[List[torch.nn.parameter.Parameter],
                                          Tuple[torch.nn.parameter.Parameter]],
                 representation: torch.Tensor,
                 labels: Union[List[torch.Tensor], Tuple[torch.Tensor]],
                 ETF: List,
                 **kwargs):

        """
        Set the weighted gradient by task uncertainties w.r.t the shared parameters
        :param losses: tensor of per task loss
        :param last_layer_params: parameters of the last layer per task. Must correspond to the order of pre-defined modules
        :param representation: shared representation
        :param labels: sequence of labels. Must correspond to the order of pre-defined modules
        :return:
        """

        # backprop last layer
        self.backward_last_layer(losses, last_layer_params)

        task_probs = []
        features = torch.clone(representation).detach()
        E_g, Σ_g, sample_losses = [], [], []
        acc_num_out = 0
        for i, (post_module, moment_module) in enumerate(zip(self.posterior_modules, self.moments_modules)):
            # multiply by 2 for wights and biases
            num_output_items = self.n_outputs_per_task_group[i] * 2 if self.task_types[i] != 'multiclass' else 2
            if isinstance(post_module, LastLayerPosteriorTaylorApprox):
                print("acc_num_out"+str(acc_num_out))
                p_t = post_module.compute_posterior(last_layer_params=last_layer_params[acc_num_out: acc_num_out + num_output_items],
                                                    features=features,
                                                    labels=labels[i],
                                                    ETF = ETF,
                                                    full_train_features=kwargs['full_train_features'],
                                                    full_train_labels=kwargs['full_train_labels'][i] if isinstance(kwargs['full_train_labels'], list) else None
                                                    )
            else:
                raise Exception("Unsupported posterior module")

            task_probs.append(p_t)
            moments = moment_module.compute_moments(features=features,
                                                    labels=labels[i],
                                                    p_t=p_t)

            if len(moments) == 2:
                E_g_t, Σ_g_t = moments[0], moments[1]
            else:
                E_g_t, Σ_g_t, sample_losses_t = moments[0], moments[1], moments[2]
                sample_losses.append(sample_losses_t)

            E_g.append(E_g_t)
            Σ_g.append(Σ_g_t)
            acc_num_out += num_output_items

        μ_g = torch.cat(E_g, dim=1)
        Σ_g = torch.cat(Σ_g, dim=1)
        dL_dh = self.agg_scheme.aggregate(μ_g=μ_g, Σ_g=Σ_g)
        representation.backward(gradient=dL_dh.to(representation.dtype))

class AggScheme:
    """
    Base class for aggregating the gradient mean and gradient covariance obtained
    from all tasks
    """
    def __init__(self):
        pass

    def aggregate(self, *args, **kwargs) -> torch.Tensor:
        pass


class GaussianAgg(AggScheme):

    def __init__(self,
                 num_tasks,
                 ):
        super().__init__()
        self.num_tasks = num_tasks

    def aggregate(self,
                  μ_g: torch.Tensor,
                  Σ_g: torch.Tensor) -> torch.Tensor:
        """
        Aggregate over tasks, assuming diagonal covariance
        :param μ_g: mean of the gradient [bs, num_tasks, dim]
        :param Σ_g: variance of the gradient [bs, num_tasks, dim]
        :return: The gradient of the combined loss w.r.t the shared hidden layer
        """
        Λ_g = (1 / Σ_g)
        Λ_μ_g = Λ_g * μ_g

        bs = μ_g.shape[0]
        sum_inv_Λ_g = 1 / Λ_g.sum(dim=1)
        dL_dh = sum_inv_Λ_g * Λ_μ_g.sum(1) / bs

        return dL_dh


class Moments:
    """
    Base class for obtaining the first and second moment of the gradient of the loss for each task
    or set of tasks w.r.t the hidden layer
    """
    def __init__(self):
        pass

    def first_moment(self, *args, **kwargs) -> torch.Tensor:
        pass

    def second_moment(self, *args, **kwargs) -> torch.Tensor:
        pass

    def compute_moments(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class ExactMoments(Moments):
    """
    Computing the moments in closed-form; Supports only MSE loss with Gaussian distribution over the parameters
    """

    def __init__(self,
                 obs_noise: float = 1.0,
                 sqrt_power: float = 1.0):
        super().__init__()
        self.obs_noise = obs_noise
        self.sqrt_power = sqrt_power

    def first_moment(self,
                     features: torch.Tensor,
                     labels: torch.Tensor,
                     E_w: torch.Tensor,
                     E_ww: torch.Tensor) -> torch.Tensor:
        E_ww_h = torch.einsum('ode,bd->boe', E_ww, features)
        E_w_y = torch.einsum('bo,od->bod', labels, E_w)
        E_g = 2 * (E_ww_h - E_w_y)
        return E_g

    def second_moment(self,
                      labels: torch.Tensor,
                      E_ww: torch.Tensor,
                      E_wxww: torch.Tensor,
                      E_wxwxww: torch.Tensor) -> torch.Tensor:
        E_ww_y_2 = torch.einsum('bo,ode->bode', labels ** 2, E_ww)
        E_wxww_y = torch.einsum('bo,bode->bode', labels, E_wxww)
        E_g_g = (2 ** 2) * (E_ww_y_2 - 2 * E_wxww_y + E_wxwxww)
        return E_g_g

    def compute_moments(self,
                        features: torch.Tensor,
                        labels: torch.Tensor,
                        p_t: torch.distributions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assume independence between all outputs
        :param features: hidden layer representation [bs, dim]
        :param labels: labels for each task and outputs [bs, n_tasks * n_outputs]
        :param p_t: A Gaussian distribution over the parameters assuming [n_tasks * n_outputs, dim]
        :return: expected value and covariance of the gradient
        """

        rep_dim = features.shape[-1]
        h_L = features.detach().clone()

        μ = p_t.mean[:, :rep_dim]
        Σ = p_t.covariance_matrix[:, :rep_dim, :rep_dim]

        μμ = torch.einsum('od,oe->ode', μ, μ)
        Σ_μμ = Σ + μμ
        Σ_μμ_neg = Σ - μμ

        E_w = μ
        E_ww = Σ_μμ

        μ_h = torch.einsum('od,be->obde', μ, h_L).permute(1, 0, 2, 3)  # verified
        μ_h_Σ_μμ = torch.einsum('bode,oef->bodf', μ_h, Σ_μμ)  # verified
        Σ_μμ_μ_h = torch.einsum('ode,bofe->bodf', Σ_μμ, μ_h)  # verified
        h_μ_Σ_μμ_neg = torch.einsum('bd,od,oef->boef', h_L, μ, Σ_μμ_neg)  # verified
        E_wxww = μ_h_Σ_μμ + Σ_μμ_μ_h + h_μ_Σ_μμ_neg  # verified

        hh = torch.einsum('bd,be->bde', h_L, h_L)
        hh_hh = hh + hh.permute(0, 2, 1)
        Σ_μμ_hh_hh_Σ_μμ = torch.einsum('ode,bef,ofg->bodg', Σ_μμ, hh_hh, Σ_μμ)  # verified
        μ_hh_μ_Σ_μμ_neg = torch.einsum('od,bde,oe,ofg->bofg', μ, hh, μ, Σ_μμ_neg)  # verified
        hhΣ = torch.einsum('bdf,ofe->bode', hh, Σ)
        tr_hhΣ = torch.diagonal(hhΣ, dim2=-2, dim1=-1).sum(-1)
        tr_hhΣ_Σ_μμ = torch.einsum('bo,ode->bode', tr_hhΣ, Σ_μμ)  # verified
        E_wxwxww = Σ_μμ_hh_hh_Σ_μμ + μ_hh_μ_Σ_μμ_neg + tr_hhΣ_Σ_μμ

        E_g = self.first_moment(h_L, labels, E_w, E_ww)
        E_g_g = self.second_moment(labels, E_ww, E_wxww, E_wxwxww)

        E_gE_g = torch.einsum('bod,boe->bode', E_g, E_g)
        Σ_g = E_g_g - E_gE_g
        Σ_g = torch.clamp(torch.diagonal(Σ_g.detach(), dim1=-2, dim2=-1), min=1e-8)
        Σ_g = Σ_g ** self.sqrt_power

        return (E_g, Σ_g)


class ApproxMoments(Moments):

    def __init__(self,
                 sqrt_power: float = 1.0,
                 num_mc_samples: int = 512):
        super().__init__()
        self.sqrt_power = sqrt_power
        self.num_mc_samples = num_mc_samples

    def first_moment(self,
                     dL_dh: torch.Tensor) -> torch.Tensor:
        return dL_dh.mean(0)

    def second_moment(self,
                      dL_dh: torch.Tensor) -> torch.Tensor:
        return (dL_dh ** 2).mean(0)

    def dL_dh(self, features, labels, p_t, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def compute_moments(self,
                        features: torch.Tensor,
                        labels: torch.Tensor,
                        p_t: torch.distributions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param features: hidden layer representation [bs, dim]
        :param labels: labels [bs]
        :param p_t: A Gaussian distribution over the parameters [n_clases * (dim + 1)]
        :return: expected value, covariance, and mean losses per task group
        """

        dL_dhidden, sample_losses = self.dL_dh(features, labels, p_t)  # [n_samples, bs, *, dim]; [n_tasks * n_outputs] / [1]

        E_g = self.first_moment(dL_dhidden)
        E_g_g = self.second_moment(dL_dhidden)

        E_gE_g = E_g ** 2
        Σ_g = E_g_g - E_gE_g
        Σ_g = torch.clamp(Σ_g.detach(), min=1e-8)
        Σ_g = Σ_g ** self.sqrt_power

        return (E_g, Σ_g, sample_losses)


class ApproxMomentsMultiClass(ApproxMoments):
    """
    Computing an approximation of the moments using Monte-Carlo for a multi-classification task
    """

    def __init__(self,
                 num_classes: int,
                 sqrt_power: float = 1.0,
                 num_mc_samples: int = 512):
        super().__init__(sqrt_power, num_mc_samples)
        self.num_classes = num_classes
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    def dL_dh(self,
              features: torch.Tensor,
              labels: torch.Tensor,
              p_t: torch.distributions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param features: hidden layer representation [bs, dim]
        :param labels: labels [bs]
        :param p_t: A Gaussian distribution over the parameters [n_clases * (dim + 1)]
        :return: approximate expected value and covariance of the gradient for the task
        """

        rep_dim = features.shape[-1]
        h_L = features.detach().clone()#.requires_grad_()

        post_samples = p_t.rsample(sample_shape=(self.num_mc_samples,))  # [num_mc_samples, n_outputs * dim]
        # reshape to the size of the weight matrix
        post_samples = post_samples.view(self.num_mc_samples, self.num_classes, -1)
        sampled_weights = post_samples[:, :, :-1]  # [num_mc_samples, n_outputs, dim-1]
        sampled_biases = post_samples[:, :, -1]

        logits = torch.einsum('bd,sod->sbo', h_L, sampled_weights) + sampled_biases[:, None, :]
        y_expend = labels.clone().unsqueeze(0).repeat(self.num_mc_samples, 1)
        dL_dout = torch.softmax(logits, dim=-1) - F.one_hot(y_expend, num_classes=self.num_classes)
        dL_dhidden = torch.einsum('sod,sbo->sbod', sampled_weights, dL_dout).sum(dim=-2, keepdim=True)  # verified

        # check gradient with autograd
        sample_losses = self.loss_func(logits.permute(1, 2, 0), y_expend.t()).detach()
        return dL_dhidden, sample_losses.mean()


class ApproxMomentsBinaryTasks(ApproxMoments):
    """
    Computing an approximation of the moments using Monte-Carlo for a set of binary tasks
    """

    def __init__(self,
                 num_outputs: int,
                 sqrt_power: float = 1.0,
                 num_mc_samples: int = 512):
        super().__init__(sqrt_power, num_mc_samples)

        self.num_outputs = num_outputs
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

    def dL_dh(self,
              features: torch.Tensor,
              labels: torch.Tensor,
              p_t: torch.distributions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param features: hidden layer representation [bs, dim]
        :param labels: labels [bs, n_tasks * n_outputs]
        :param p_t: A Gaussian distribution over the parameters [n_tasks * n_outputs * (dim + 1)]
        :return: approximate expected value and covariance of the gradient for all tasks
        """

        rep_dim = features.shape[-1]
        h_L = features.detach().clone()

        hidden_dim = rep_dim + 1  # add bias
        μ_w = p_t.mean.view(self.num_outputs, -1)
        n_out_dim = self.num_outputs * hidden_dim
        Σ_w = [p_t.covariance_matrix[i:i+hidden_dim, i:i+hidden_dim] for i in range(0, n_out_dim, hidden_dim)]
        Σ_w = torch.stack(Σ_w, dim=0)
        scale_tri_Σ_w = psd_safe_cholesky(Σ_w)
        p_t_ = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=μ_w,
            scale_tril=scale_tri_Σ_w,
        )
        post_samples = p_t_.rsample(sample_shape=(self.num_mc_samples,))  # [num_samples, num_outputs, dim]

        # reshape to the size of the weight matrix
        sampled_weights = post_samples[:, :, :-1]  # [num_mc_samples, n_outputs, dim-1]
        sampled_biases = post_samples[:, :, -1]

        logits = torch.einsum('bd,sod->sbo', h_L, sampled_weights) + sampled_biases[:, None, :]
        y_expend = labels.clone().unsqueeze(0).repeat(self.num_mc_samples, 1, 1)
        sample_losses = self.loss_func(logits, y_expend).detach()
        dL_dout = torch.sigmoid(logits) - y_expend
        dL_dhidden = torch.einsum('sod,sbo->sbod', sampled_weights, dL_dout)

        return dL_dhidden, sample_losses.mean(dim=(0, 1))


class LastLayerPosterior:
    """
    Base class for obtaining the posterior distribution over the final layer parameters
    """
    def __init__(self,
                 gamma: float):
        self.gamma = gamma

    def prior(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def posterior(self, *args, **kwargs) -> torch.distributions:
        pass


class LastLayerPosteriorTaylorApprox(LastLayerPosterior):
    """
    Approximate inference for classification tasks using Taylor expansion
    """

    def __init__(self,
                 gamma: float = 0.001,
                 setting: str = "multiclass"):

        super().__init__(gamma)

        if setting not in ("multiclass", "binary_tasks"):
            raise Exception("Unsupported setting")

        self.setting = setting
        self.loss_fun = torch.nn.CrossEntropyLoss(reduction='none') if setting == "multiclass" \
            else torch.nn.BCEWithLogitsLoss(reduction='none')
        self.full_data_posterior = None

    def last_layer_jacobians(self,
                             features: torch.Tensor,
                             logits: torch.Tensor,) -> torch.Tensor:
        """
        Based on https://github.com/aleximmer/Laplace/blob/main/laplace/curvature/curvature.py
        Compute Jacobian matrix for each example in the batch
        :param features: last layer representation  [bs, dim]
        :param logits: output of the network [bs, outputs]
        :return: Jacobians [bs, outputs, last-layer-parameters]
        """
        bsize = features.shape[0]
        output_size = int(logits.numel() / bsize)

        # calculate Jacobians using the feature vector 'features'; add bias here to ensure compatibility with
        # the convention that the feature bias always comes after the data features
        bias_term = torch.ones(bsize, device=features.device)[:, None]
        features = torch.cat((features, bias_term), dim=1)
        identity = torch.eye(output_size, device=features.device).unsqueeze(0).tile(bsize, 1, 1)
        # Jacobians are batch x output x params
        Js = torch.einsum('kp,kij->kijp', features, identity).reshape(bsize, output_size, -1)

        return Js
    
    def last_layer_jacobians_ETF(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        计算 BatchNorm1d 的雅可比矩阵
        :param features: BatchNorm1d 前的输入 [bs, dim]
        :param logits: BatchNorm1d 的输出 [bs, dim]
        :return: Jacobians [bs, dim, dim+2]
        """
        bsize, dim = features.shape

        # Jacobian 对 weight 和 bias 的导数分别是 features 和 1
        dL_dweight = features  # 对应 weight 的导数
        dL_dbias = torch.ones(bsize, dim, device=features.device)  # 对应 bias 的导数

        # 将 features 与 Jacobians 一起计算，包括 weight 和 bias 的导数
        identity = torch.eye(dim, device=features.device).unsqueeze(0).tile(bsize, 1, 1)
        Js = torch.cat([identity, dL_dweight.unsqueeze(2), dL_dbias.unsqueeze(2)], dim=2)  # 结果为 [bs, dim, dim+2]

        return Js

    def dL_dlogits_wb(self,
                   features: torch.Tensor,
                   labels: torch.Tensor,
                   ETF: List,
                   w: torch.Tensor,
                   b: torch.Tensor):
        
        w_ = w.clone().requires_grad_()
        b_ = b.clone().requires_grad_()
        if w.dim() == 1:
            feat = F.batch_norm(features, running_mean=ETF[0],running_var=ETF[1],weight=w_, bias=b_)
            feat_ETF = feat / torch.clamp(torch.sqrt(torch.sum(feat ** 2, dim=1, keepdims=True)), 1e-8)
            logits = torch.matmul(feat_ETF, ETF[2])
        else:
            logits = F.linear(features, weight=w_, bias=b_)

        loss = self.loss_fun(logits, labels).sum(dim=0)

        grad = torch.autograd.grad(
            loss.sum(),  # either scalar for multi-class classification or vector
            [logits, w_, b_],
            retain_graph=False,
            allow_unused=True
        )
        dL_dlogits = grad[0]  # note that this is the gradient divided by the batch size
        if grad[1].dim() == 1:
            dL_dwb = torch.cat((grad[1][:, None], grad[2][:, None]), dim=1)  # verified!
        else:
            dL_dwb = torch.cat((grad[1], grad[2][:, None]), dim=1)  # verified!
        if w.dim() == 1:
            logits = feat
        return logits.detach(), dL_dlogits.detach(), dL_dwb.detach()

    def prior(self,
              last_layer_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param last_layer_params: parameters of the last layer,
        in multiclass classification: [n_classes, dim + 1] (+1 for the bias)
        in a set of binary classification tasks: [n_tasks * n_outputs, dim + 1]  (+1 for the bias)
        :return: prior mean and precision
        """

        num_el = torch.numel(last_layer_params)
        prior_mean = torch.zeros(num_el).to(last_layer_params.device)
        prior_precision = (1 / self.gamma) * torch.ones(num_el).diag().to(last_layer_params.device)

        return prior_mean, prior_precision

    def posterior(self,
                 features: torch.Tensor,
                 labels: torch.Tensor,
                 last_layer_params: torch.Tensor,
                 ETF: List,
                 prior_mean: torch.Tensor,
                 prior_precision: torch.Tensor) -> torch.distributions:
        """
        :param features: [bs, dim]
        :param labels: [bs, n_tasks * n_outputs] / [bs]
        :param last_layer_params: [n_classes, dim + 1] or [n_tasks * n_outputs, dim + 1]
        :param prior_mean: mean for last layer parameters
        [n_tasks * n_outputs * (dim + 1)] or [n_classes * (dim + 1)]
        :param prior_precision: precision for last layer parameters
        [n_tasks * n_outputs * (dim + 1), n_tasks * n_outputs * (dim + 1)] or [n_classes * (dim + 1), n_classes * (dim + 1)]
        :return: posterior distribution
        """
        h_L = features.detach().clone()
        bs = h_L.shape[0]
        print(f"last_layer_params shape: {last_layer_params.shape}")
        w = last_layer_params[:, :-1]
        b = last_layer_params[:, -1]
        print(w.shape[-1]==1)
        use_ETF = 0
        if w.shape[-1]==1:
            w = w.squeeze(1)
            use_ETF = 1
        print(f"w shape: {w.shape}")
        print(f"b shape: {b.shape}")
        logits, dL_dout, dL_dwb = self.dL_dlogits_wb(h_L, labels, ETF, w, b)  # [bs, n_classes]; [num_classes, dim+1]

        with torch.no_grad():
            # taken from: https://github.com/aleximmer/Laplace/blob/main/laplace/curvature/curvature.py
            if self.setting == "multiclass" and use_ETF != 1:
                ps = torch.softmax(logits, dim=-1)
            else:
                ps = logits

            if use_ETF:
                batch_size, new_dim = ps.shape
                subset_size = 1
                print(f"ps shape: {ps.shape}")

                H_out_parts = []

                for start_idx in range(0, batch_size, subset_size):
                    torch.cuda.empty_cache()
                    # print(start_idx)
                    end_idx = min(start_idx + subset_size, batch_size)
                    ps_subset = ps[start_idx:end_idx]
                    # diag_embed_part = torch.diag_embed(ps_subset) 
                    ps_subset_cpu = ps_subset.cpu()  # 将ps_subset移动到CPU
                    diag_embed_part = torch.diag_embed(ps_subset_cpu).cuda()
                    einsum_part = torch.einsum('mk,mc->mck', ps_subset, ps_subset)
                    H_out_part = diag_embed_part - einsum_part
                    H_out_parts.append(H_out_part)  

                H_out = torch.cat(H_out_parts, dim=0)  # [bs, dim, dim]
                dout_dwb = self.last_layer_jacobians_ETF(h_L, logits)  # [bs, dim, dim+2]
            else:
                H_out = torch.diag_embed(ps) - torch.einsum('mk,mc->mck', ps, ps)  # [bs, n_classes, n_classes]
                dout_dwb = self.last_layer_jacobians(h_L, logits)  # [bs, n_classes, n_classes * dim+1]

            a = (dL_dwb.flatten() +
                (prior_precision @ (last_layer_params.flatten() - prior_mean)))[:, None]

            H_ggn = torch.zeros_like(prior_precision)
            idx = 0

            for i in range(0, bs, MAX_SIZE):

                dout_dwb_part = dout_dwb[idx * MAX_SIZE: (idx+1) * MAX_SIZE, ...]
                H_out_part = H_out[idx * MAX_SIZE: (idx+1) * MAX_SIZE, ...]
                print("?????????????????????")
                print(H_ggn.shape)
                print(dout_dwb_part.shape)
                print(H_out_part.shape)
                print("?????????????????????")
                H_ggn += torch.einsum('mcp,mck,mkq->pq', dout_dwb_part, H_out_part, dout_dwb_part)
                idx += 1

            #H_ggn = torch.einsum('mcp,mck,mkq->pq', dout_dwb, H_out, dout_dwb).div(scale)

            Λ = (H_ggn + prior_precision).detach()

            scale_tri_Λ = psd_safe_cholesky(Λ)
            B_inv_a = torch.cholesky_solve(a, scale_tri_Λ).squeeze(-1)

            Σ = torch.cholesky_solve(torch.eye(scale_tri_Λ.shape[-1],
                                            dtype=h_L.dtype, device=h_L.device), scale_tri_Λ)
            scale_tri_Σ = psd_safe_cholesky(Σ)

            mean_post = (last_layer_params.flatten() - B_inv_a).detach()

            p_t = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=mean_post,
                scale_tril=scale_tri_Σ,
            )

        return p_t

    def compute_posterior(self,
                          last_layer_params: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
                          features: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
                          labels: torch.Tensor,
                          ETF : List,
                          full_train_features: torch.Tensor = None,
                          full_train_labels: torch.Tensor = None) -> torch.distributions:

        # make last-layer parameters as matrix in case of a list
        if isinstance(last_layer_params, (list, tuple)):
            # print(f"last_layer_params shape: {last_layer_params.shape}")
            # print(f"last_layer_params shape: {last_layer_params}")
            # print(f"last_layer_params: {last_layer_params.shape}")
            for idx, param in enumerate(last_layer_params):
                print(f"Index: {idx}, Shape: {param.shape}")
            ws = torch.cat([w.detach().clone() for w in last_layer_params[::2]], dim=0)
            bs = torch.cat([b.detach().clone() for b in last_layer_params[1::2]], dim=0)
            if ws.dim() == 1:
                ws = ws.unsqueeze(-1)
                bs = bs.unsqueeze(-1)
            else:   
                if bs.dim() == 1:
                    bs = bs.unsqueeze(-1)
                print(f"ws shape: {ws.shape}")
                print(f"bs shape: {bs.shape}")
            W = torch.cat([ws, bs], dim=1)
        else:
            W = last_layer_params.detach().clone()

        if full_train_features is not None:
            self.full_data_posterior = None
            full_data_prior_mean, full_data_prior_precision = self.prior(W)
            print(f"full_data_prior_mean shape: {full_data_prior_mean.shape}")
            print(f"full_data_prior_precision shape: {full_data_prior_precision.shape}")
            full_data_posterior = self.posterior(full_train_features, full_train_labels, W, ETF,
                                                 full_data_prior_mean, full_data_prior_precision)
            self.full_data_posterior = full_data_posterior
        prior_mean, prior_precision = self.full_data_posterior.mean.detach(), \
                                      self.full_data_posterior.precision_matrix.detach()

        p_t = self.posterior(features, labels, W, ETF, prior_mean, prior_precision)

        return p_t
