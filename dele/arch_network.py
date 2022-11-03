"""
Networks that take architectures as inputs.
"""

import abc
from typing import Tuple, List, Dict, Union, Optional

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from . import utils


class ArchNetwork(nn.Module):
    """
    Base class for architecture-performance predictor.
    """
    def __init__(self):
        nn.Module.__init__(self)

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass


class PointwiseComparator(ArchNetwork):
    """
    Compatible to NN regression-based predictor of architecture performance.
    """

    def __init__(
        self, 
        arch_embedder_cls,
        arch_embedder_cfg: dict = None,
        mlp_hiddens = (200, 200, 200), 
        mlp_dropout: float = 0.1,
        optimizer: dict = {
            "type": "Adam",
            "lr": 0.001
        }, 
        scheduler = None,
        compare_loss_type: str = "margin_linear",
        compare_margin: float = 0.01,
        margin_l2: bool = False,
        max_grad_norm: float = None
    ):
        super(PointwiseComparator, self).__init__()

        # configs
        assert compare_loss_type in ["binary_cross_entropy", "margin_linear"],\
                "comparing loss type {} not supported".format(compare_loss_type)
        self.compare_loss_type = compare_loss_type
        self.compare_margin = compare_margin
        self.margin_l2 = margin_l2
        self.max_grad_norm = max_grad_norm

        self.arch_embedder = arch_embedder_cls(**(arch_embedder_cfg or {}))

        dim = self.embedding_dim = self.arch_embedder.out_dim
        # construct MLP from embedding to score
        self.mlp = self.construct_mlp(dim, mlp_hiddens, mlp_dropout)
        
        # init optimizer and scheduler
        self.optimizer = utils.init_optimizer(self.parameters(), optimizer)
        self.scheduler = utils.init_scheduler(self.optimizer, scheduler)

        # used for reinit optimizer and lr scheduler
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler

    def reinit_optimizer(self, only_mlp: bool = False):
        parameters = self.mlp.parameters() if only_mlp else self.parameters()
        self.optimizer = utils.init_optimizer(parameters, self.optimizer_cfg)
    
    def reinit_scheduler(self):
        self.scheduler = utils.init_scheduler(self.optimizer, self.scheduler_cfg)

    @staticmethod
    def construct_mlp(dim: int, mlp_hiddens: Tuple[int], mlp_dropout: float, out_dim: int = 1) -> nn.Module:
        mlp = []
        for hidden_size in mlp_hiddens:
            mlp.append(nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.ReLU(inplace = False),
                nn.Dropout(p = mlp_dropout))
            )
            dim = hidden_size
        mlp.append(nn.Linear(dim, out_dim))
        mlp = nn.Sequential(*mlp)
        return mlp

    def predict(self, arch, sigmoid=True, tanh=False):
        score = self.mlp(self.arch_embedder(arch)).squeeze(-1)
        if sigmoid:
            score = torch.sigmoid(score)
        elif tanh:
            score = torch.tanh(score)
        return score
    
    def update_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return loss.item()

    def update_predict(self, archs, labels):
        mse_loss = self.cal_predict_loss(archs, labels)
        return self.update_step(mse_loss)

    def cal_predict_loss(self, archs, labels):
        return self._cal_predict_loss(archs, labels, self.mlp)
    
    def _cal_predict_loss(self, archs, labels, mlp):
        scores = torch.sigmoid(mlp(self.arch_embedder(archs)))
        mse_loss = F.mse_loss(
            scores.squeeze(), scores.new(labels))
        return mse_loss

    def compare(self, arch_1, arch_2):
        # pointwise score and comparen
        s_1 = self.mlp(self.arch_embedder(arch_1)).squeeze()
        s_2 = self.mlp(self.arch_embedder(arch_2)).squeeze()
        return torch.sigmoid(s_2 - s_1)

    def update_compare(self, arch_1, arch_2, better_labels, margin=None):
        pair_loss = self.cal_compare_loss(arch_1, arch_2, better_labels, margin)
        return self.update_step(pair_loss)

    def cal_compare_loss(self, arch_1, arch_2, better_labels, margin = None):
        return self._cal_compare_loss(arch_1, arch_2, better_labels, self.mlp, margin)

    def _cal_compare_loss(self, arch_1, arch_2, better_labels, mlp, margin = None):
        if self.compare_loss_type == "binary_cross_entropy":
            s_1 = mlp(self.arch_embedder(arch_1)).squeeze()
            s_2 = mlp(self.arch_embedder(arch_2)).squeeze()
            compare_score = torch.sigmoid(s_2 - s_1)
            pair_loss = F.binary_cross_entropy(
                    compare_score, compare_score.new(better_labels))

        elif self.compare_loss_type == "margin_linear":
            s_1 = mlp(self.arch_embedder(arch_1)).squeeze()
            s_2 = mlp(self.arch_embedder(arch_2)).squeeze()
            better_pm = 2 * s_1.new(np.array(better_labels, dtype=np.float32)) - 1
            zero_ = s_1.new([0.])
            margin = [self.compare_margin] if margin is None else margin
            margin = s_1.new(margin)
            if not self.margin_l2:
                pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
            else:
                pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)) \
                        ** 2 / np.maximum(1., margin))
        return pair_loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location = torch.device("cpu")))

    def on_epoch_start(self, epoch):
        if self.scheduler is not None:
            self.scheduler.step(epoch - 1)
            print("Epoch %3d: lr: %.5f", epoch, self.scheduler.get_lr()[0])

    def _clip_grads(self):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)


class DynamicEnsemblePointwiseComparator(PointwiseComparator):
    r"""
    Dynamic ensemble pointwise comparator.

    Args:
        auxiliary_head_num (int): Number of the low-fidelity experts.
        use_uniform_confidence (bool): Whether use uniform confidence. Default: `False`.
    """
    NAME = "dynamic_ensemble_pointwise_comparator"

    def __init__(
        self, 
        auxiliary_head_num: int,
        arch_embedder_cls,
        arch_embedder_cfg = None,
        use_uniform_confidence: bool = False,
        mlp_hiddens: Tuple[int] = (200, 200, 200), 
        mlp_dropout: float = 0.1,
        optimizer: Dict[str, Union[str, float]] = {
            "type": "Adam",
            "lr": 0.001
        }, 
        scheduler: Optional = None,
        compare_loss_type: str = "margin_linear",
        compare_margin: float = 0.01,
        margin_l2: bool = False,
        max_grad_norm: float = None
    ) -> None:
        super(DynamicEnsemblePointwiseComparator, self).__init__(
            arch_embedder_cls, 
            arch_embedder_cfg,
            mlp_hiddens, 
            mlp_dropout,
            optimizer, 
            scheduler,
            compare_loss_type, 
            compare_margin,
            margin_l2, 
            max_grad_norm 
        )

        self.auxiliary_head_num = auxiliary_head_num
        self.use_uniform_confidence = use_uniform_confidence

        if self.use_uniform_confidence:
            self.confidence = nn.Parameter(
                torch.randn((1, self.auxiliary_head_num), 
                requires_grad = True)
            )
            nn.init.constant_(self.confidence, 1. / self.auxiliary_head_num)
        else:
            self.arch_embedder = arch_embedder_cls(**(arch_embedder_cfg or {}))
            dim = self.embedding_dim = self.arch_embedder.out_dim
            # construct MLP from architecture embedding to prediction confidence score
            self.confidence_mlp = self.construct_mlp(
                dim, mlp_hiddens, mlp_dropout, auxiliary_head_num
            )

        self.module_lst = nn.ModuleList([
            PointwiseComparator(
                arch_embedder_cls, 
                arch_embedder_cfg,
                mlp_hiddens, 
                mlp_dropout,
                optimizer, 
                scheduler,
                compare_loss_type, 
                compare_margin,
                margin_l2, 
                max_grad_norm
            ) for i in range(self.auxiliary_head_num)
        ])

        # init optimizer and scheduler
        self.reinit_optimizer(only_mlp = False)
        self.reinit_scheduler()
    
    def init_optimizer(self):
        self.optimizer = utils.init_optimizer(self.parameters(), self.optimizer_cfg)
    
    def init_scheduler(self):
        self.scheduler = utils.init_scheduler(self.optimizer, self.scheduler_cfg)

    def mtl_update_compare(self, auxiliary_datas, margin = None):
        loss = 0.
        for model, (arch_1, arch_2, better_lst) in zip(self.module_lst, auxiliary_datas):
            s_1 = model.predict(arch_1, False, False)
            s_2 = model.predict(arch_2, False, False)
            loss += self._compare_loss(s_1, s_2, better_lst, margin)
        return self.update_step(loss)

    def update_compare(self, arch_1, arch_2, better_labels, margin = None) -> float:
        score_1 = self.predict(arch_1, False, False)
        score_2 = self.predict(arch_2, False, False)
        loss = self._compare_loss(score_1, score_2, better_labels, margin)
        return self.update_step(loss)

    def mtl_predict(self, arch, sigmoid: bool = True, tanh: bool = False) -> Tuple[Tensor, List[Tensor]]:
        score = self.predict(arch, sigmoid, tanh)
        lf_score_lst = [model.predict(arch, sigmoid, tanh) for model in self.module_lst]
        return score, lf_score_lst

    def separate_score_predict(self, arch) -> Tensor:
        score_lst = torch.cat([
            model.predict(arch, False, False).unsqueeze(1) 
            for model in self.module_lst],
        1)
        return score_lst

    def weighted_score(self, arch) -> Tensor:
        score_lst = self.separate_score_predict(arch)
        confidence_ratio = self._confidence_ratio(arch)
        score = (confidence_ratio * score_lst)
        return score

    def predict(self, arch, sigmoid: bool = True, tanh: bool = False) -> Tensor:
        weighted_score = self.weighted_score(arch)
        score = weighted_score.sum(1)
        if sigmoid:
            score = torch.sigmoid(score)
        elif tanh:
            score = torch.tanh(score)
        return score

    def _confidence_ratio(self, archs) -> Tensor:
        if self.use_uniform_confidence:
            confidence_ratio = torch.softmax(self.confidence, 1)
        else:
            arch_embeddings = self.arch_embedder(archs)
            confidence_ratio = torch.softmax(self.confidence_mlp(arch_embeddings), 1) # softmax or sigmoid, maybe ablation
        return confidence_ratio

    def _compare_loss(self, s_1, s_2, better_labels, margin = None):
        s_1 = s_1.squeeze()
        s_2 = s_2.squeeze()
        better_pm = 2 * s_1.new(np.array(better_labels, dtype = np.float32)) - 1
        zero_ = s_1.new([0.])
        margin = [self.compare_margin] if margin is None else margin
        margin = s_1.new(margin)
        if not self.margin_l2:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
        else:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)) ** 2 / np.maximum(1., margin))
        return pair_loss

    def update_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return loss.item()
