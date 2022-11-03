from typing import List
import random
import pickle

import six
import numpy as np
import torch
from torch import optim
from scipy.stats import stats


def set_seed(seed: int):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def init_optimizer(params, cfg):
    if cfg:
        cfg = {k:v for k, v in six.iteritems(cfg)}
        opt_cls = getattr(optim, cfg.pop("type"))
        return opt_cls(params, **cfg)
    return None


def init_scheduler(optimizer, cfg):
    if cfg and optimizer is not None:
        cfg = {k:v for k, v in six.iteritems(cfg)}
        sch_cls = getattr(optim.lr_scheduler, cfg.pop("type"))
        return sch_cls(optimizer, **cfg)
    return None


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def is_empty(self):
        return self.cnt == 0

    def reset(self):
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def test_xp(true_scores, predict_scores):
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1]
    num_archs = len(ranks)
    # calculate precision at each point
    cur_inds = np.zeros(num_archs)
    passed_set = set()
    for i_rank, rank in enumerate(ranks):
        cur_inds[i_rank] = (cur_inds[i_rank - 1] if i_rank > 0 else 0) + \
                           int(i_rank in passed_set) + int(rank <= i_rank)
        passed_set.add(rank)
    patks = cur_inds / (np.arange(num_archs) + 1)
    THRESH = 100
    p_corrs = []
    for prec in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        k = np.where(patks[THRESH:] >= prec)[0][0] + THRESH
        arch_inds = ranks[:k][ranks[:k] < k]
        p_corrs.append((k, float(k)/num_archs, len(arch_inds), prec, stats.kendalltau(
            reorder_true_scores[arch_inds],
            reorder_predict_scores[arch_inds]).correlation))
    return p_corrs


def test_xk(true_scores, predict_scores):
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1]
    num_archs = len(ranks)
    patks = []
    for ratio in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        k = int(num_archs * ratio)
        p = len(np.where(ranks[:k] < k)[0]) / float(k)
        arch_inds = ranks[:k][ranks[:k] < k]
        patks.append((k, ratio, len(arch_inds), p, stats.kendalltau(
            reorder_true_scores[arch_inds],
            reorder_predict_scores[arch_inds]).correlation))
    return patks


def compare_data(
    archs, 
    accs: List[float], 
    max_compare_ratio: float = 4., 
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
):
    n_max_pairs = int(max_compare_ratio * len(archs))
    acc_diff = np.array(accs)[:, None] - np.array(accs)
    acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
    ex_thresh_inds = np.where(acc_abs_diff_matrix > compare_threshold)
    ex_thresh_num = len(ex_thresh_inds[0])
    if ex_thresh_num > n_max_pairs:
        if choose_pair_criterion == "diff":
            keep_inds = np.argpartition(
                    acc_abs_diff_matrix[ex_thresh_inds], -n_max_pairs)[-n_max_pairs:]
            ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])

        elif choose_pair_criterion == "random":
            keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace=False)
            ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
    archs_1, archs_2, better_lst = archs[ex_thresh_inds[1]], archs[ex_thresh_inds[0]], (acc_diff > 0)[ex_thresh_inds]
    return archs_1, archs_2, better_lst


def valid(val_loader, model, save_path: str = None):
    """
    Test the vanilla predictor on the validation set.

    Args:
        val_loader: Validation data loader.
        model (PointwiseComparator): The vanilla architecture performance predictor.
        save_path (str): Path to save the statistics. Default: `None`.

    Returns:
        low_fidelity_corr (float): The Kendall's Tau correlation between the predicted 
                                   scores and the actual low-fidelity information.
        real_corr (float): The Kendall's Tau correlation between the predicted scores 
                           and the actual performance.
        patk: P@topk.
    """
    model.eval()

    all_scores = []
    all_low_fidelity = []
    all_real_accs = []

    for step, (archs, real_accs, low_fidelity) in enumerate(val_loader):
        archs = np.array(archs)
        low_fidelity = np.array(low_fidelity)
        real_accs = np.array(real_accs)
        n = len(archs)
        scores = list(model.predict(archs).cpu().data.numpy())
        all_scores += scores
        all_low_fidelity += list(low_fidelity)
        all_real_accs += list(real_accs)
    
    low_fidelity_corr = stats.kendalltau(all_low_fidelity, all_scores).correlation
    real_corr = stats.kendalltau(all_real_accs, all_scores).correlation
    patk = test_xk(all_real_accs, all_scores)

    if save_path:
        data = {"real_acc": all_real_accs, "score": all_scores}
        with open(save_path, "wb") as wf:
            pickle.dump(data, wf)

    return low_fidelity_corr, real_corr, patk


def mtl_valid(val_loader, model, save_path: str = None):
    """"
    Test the dynamic ensemble predictor on the validation set.

    Args:
        val_loader: Validation data loader.
        model (DynamicEnsemblePointwiseComparator): The architecture performance predictor.
        save_path (str): Path to save the statistics. Default: `None`.

    Returns:
        low_fidelity_corr (Dict[str, float]): The Kendall's Tau correlation between different
            low-fidelity experts' predicted scores and the actual low-fidelity information.
        real_corr (float): The Kendall's Tau correlation between the predicted scores and the
            actual performance.
        patk: P@topk.
    """
    model.eval()

    all_scores = []
    all_real_accs = []

    for step, (archs, real_accs, low_fidelity_perfs) in enumerate(val_loader):
        archs = np.array(archs)
        real_accs = np.array(real_accs)

        n = len(archs)

        scores, auxiliary_scores_lst = model.mtl_predict(archs)
        scores = list(scores.cpu().data.numpy())
        all_scores += scores
        all_real_accs += list(real_accs)

        low_fidelity_types = list(low_fidelity_perfs[0].keys())

        if step == 0:
            all_low_fidelity = {low_fidelity: [] for low_fidelity in low_fidelity_types}
            all_low_fidelity_scores = {low_fidelity: [] for low_fidelity in low_fidelity_types}

        for i, low_fidelity in enumerate(low_fidelity_types):
            all_low_fidelity[low_fidelity] += [perf[low_fidelity] for perf in low_fidelity_perfs]
            all_low_fidelity_scores[low_fidelity] += list(auxiliary_scores_lst[i].cpu().data.numpy())

    low_fidelity_corr = {
        low_fidelity: stats.kendalltau(all_low_fidelity[low_fidelity],
                                       all_low_fidelity_scores[low_fidelity]).correlation
        for low_fidelity in low_fidelity_types
    }
    real_corr = stats.kendalltau(all_real_accs, all_scores).correlation
    patk = test_xk(all_real_accs, all_scores)

    if save_path:
        data = {
            "real_acc": all_real_accs, 
            "score": all_scores, 
            "low_fidelity": all_low_fidelity,
            "low_fidelity_score": all_low_fidelity_scores
        }
        with open(save_path, "wb") as wf:
            pickle.dump(data, wf)

    return low_fidelity_corr, real_corr, patk


def mtl_pretrain(
    train_loader, 
    model, 
    epoch: int, 
    max_compare_ratio: float = 4.,
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
) -> float:
    """
    In the first step of the dynamic ensemble training, train different low-fidelity 
    experts on different types of low-fidelity information.

    Args:
        train_loader: Training data loader.
        model (DynamicEnsemblePointwiseComparator): The dynamic ensemble predictor.
        epoch (int): Current epoch number.
        max_compare_ratio (float): Default: 4.
        compare_threshold (float): Default: 0.
        choose_pair_criterion (str): Default: "random".

    Returns:
        The training loss (float).
    """
    objs = AverageMeter()
    n_diff_pairs_meter = AverageMeter()

    model.train()
    for step, data in enumerate(train_loader):
        archs, accs, low_fidelity_perfs = data
        archs = np.array(archs)
        n = len(archs)

        data_lst = []

        low_fidelity_types = list(low_fidelity_perfs[0].keys())

        for low_fidelity in low_fidelity_types:
            lf_lst = np.array([perf_lst[low_fidelity] for perf_lst in low_fidelity_perfs])
            archs_1, archs_2, better_lst = compare_data(
                archs, 
                lf_lst, 
                max_compare_ratio, 
                compare_threshold, 
                choose_pair_criterion
            )
            data_lst.append((archs_1, archs_2, better_lst))
            n_diff_pairs = len(better_lst)

        n_diff_pairs_meter.update(float(n_diff_pairs))
        loss = model.mtl_update_compare(data_lst)
        objs.update(loss, n_diff_pairs)

    return objs.avg


def mtl_train(
    train_loader, 
    model, 
    epoch: int, 
    max_compare_ratio: float = 4., 
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
) -> float:
    """
    In the second step of the dynamic ensemble training, 
    finetune the entire predictor on the actual performance data.

    Args:
        train_loader: Training data loader.
        model (DynamicEnsemblePointwiseComparator): The dynamic ensemble predictor.
        epoch (int): Current epoch number.
        max_compare_ratio (float): Default: 4.
        compare_threshold (float): Default: 0.
        choose_pair_criterion (str): Default: "random".

    Returns:
        The training loss (float).
    """
    objs = AverageMeter()
    n_diff_pairs_meter = AverageMeter()

    model.train()

    for step, data in enumerate(train_loader):
        archs, accs, _ = data
        archs = np.array(archs)
        accs = np.array(accs)
        n = len(archs)

        archs_1, archs_2, better_lst = compare_data(
            archs, 
            accs, 
            max_compare_ratio, 
            compare_threshold,
            choose_pair_criterion
        )
        n_diff_pairs = len(better_lst)
        n_diff_pairs_meter.update(float(n_diff_pairs))
        loss = model.update_compare(archs_1, archs_2, better_lst)
        objs.update(loss, n_diff_pairs)

    return objs.avg


def train(
    train_loader, 
    model,
    epoch: int,
    compare: bool = True,
    use_low_fidelity: bool = False,
    max_compare_ratio: float = 4.,
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
) -> float:
    """
    Train the vanilla predictor.

    Args:
        train_loader: Training data loader.
        model (PointwiseComparator): The vanilla predictor.
        epoch (int): Current epoch number.
        compare (bool): If `True`, train the predictor with ranking loss.
                        Else if `False`, train the predictor with regression loss.
                        Default: `True`.
        use_low_fidelity (bool): Whether train the predictor on the low-fidelity information.
                                 Default: `False`.
        max_compare_ratio (float): Default: 4.
        compare_threshold (float): Default: 0.
        choose_pair_criterion (str): Default: "random".

    Returns:
        The training loss (float).

    """
    objs = AverageMeter()
    n_diff_pairs_meter = AverageMeter()

    model.train()
    for step, (archs, real_accs, low_fidelity) in enumerate(train_loader):
        archs = np.array(archs)
        real_accs = np.array(real_accs)
        low_fidelity = np.array(low_fidelity)
        accs = low_fidelity if use_low_fidelity else real_accs
        n = len(archs)

        if compare:
            archs_1, archs_2, better_lst = compare_data(
                archs, 
                accs, 
                max_compare_ratio, 
                compare_threshold,
                choose_pair_criterion
            )
            n_diff_pairs = len(better_lst)
            n_diff_pairs_meter.update(float(n_diff_pairs))
            loss = model.update_compare(archs_1, archs_2, better_lst)
            objs.update(loss, n_diff_pairs)
        else:
            loss = model.update_predict(archs, accs)
            objs.update(loss, n)

    return objs.avg
