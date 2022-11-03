# -*- coding: utf-8 -*-
# pylint: disable-all

"""
Train architecture-performance predictor on NAS-Bench-201 / NAS-Bench-301 / NDS-ResNet/ResNeXT-A.

First, pretrain the predictor on a single type of low-fidelity information.
Second, finetune the predictor on the actual performance data.
"""

import os
import sys
import logging
import shutil
import argparse
import pickle
import yaml

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import setproctitle
from torch.utils.data import DataLoader

from dele import utils
from dele.dataset import LFArchDataset
from dele.arch_embedder import NasBench201_LSTMSeqEmbedder
from dele.arch_network import PointwiseComparator


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file")
    parser.add_argument("--train-ratio", type = float, required = True, 
            help = "Proportions of training samples")
    parser.add_argument("--pretrain-ratio", type = float, default = 1., 
            help = "Proportions of pre-training samples")
    parser.add_argument("--train-pkl", type = str, required = True, help = "Training Datasets pickle")
    parser.add_argument("--valid-pkl", type = str, required = True, help = "Evaluate Datasets pickle")
    parser.add_argument("--no-pretrain", default = False, action = "store_true", 
            help = "Directly train the predictor without pretraining")
    parser.add_argument("--low-fidelity-type", type = str, default = "one_shot", 
            help = "Applied low-fidelity information type")
    parser.add_argument("--train-dir", default = None, help = "Save train log / results into TRAIN_DIR")
    parser.add_argument("--seed", default = None, type = int)
    parser.add_argument("--gpu", type = int, default = 0, help = "gpu device id")
    parser.add_argument("--load", default = None, help = "Load comparator from disk.")
    parser.add_argument("--num-workers", default = 4, type = int)
    parser.add_argument("--test-every", default = 10, type = int)
    parser.add_argument("--test-only", default = False, action = "store_true")
    args = parser.parse_args()

    setproctitle.setproctitle("python {} config: {}; train_dir: {}; cwd: {}"\
                              .format(__file__, args.cfg_file, args.train_dir, os.getcwd()))

    # ---- Preparation & Setups ----

    # Set log
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream = sys.stdout, level = logging.INFO, 
        format = log_format, datefmt = "%m/%d %I:%M:%S %p"
    )
    logger = logging.getLogger()

    # Set train_dir
    if not args.test_only:
        assert args.train_dir is not None, "Must specificy `--train-dir` when training"
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)

        backup_cfg_file = os.path.join(args.train_dir, "config.yaml")
        shutil.copyfile(args.cfg_file, backup_cfg_file)
    else:
        backup_cfg_file = args.cfg_file

    # Device setup
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info("GPU device = {}".format(args.gpu))
        device = torch.device("cuda")
    else:
        logging.info("no GPU available, use CPU!!")
        device = torch.device("cpu")

    # Seed setup
    if args.seed is not None:
        utils.set_seed(args.seed)
        logging.info("Set seed: {}".format(args.seed))

    # Load and update the configuration
    with open(backup_cfg_file, "r") as cfg_f:
        cfg = yaml.load(cfg_f, Loader = yaml.FullLoader)

    cfg["train_ratio"] = args.train_ratio
    cfg["pretrain_ratio"] = args.pretrain_ratio
    cfg["low_fidelity_type"] = args.low_fidelity_type

    logging.info("Config: %s", cfg)

    args.__dict__.update(cfg)
    logging.info("Combined args: %s", args)

    # Data setup
    logging.info("Load pkl cache from {} and {}".format(args.train_pkl, args.valid_pkl))
    with open(args.train_pkl, "rb") as rf:
        train_data = pickle.load(rf)
    with open(args.valid_pkl, "rb") as rf:
        valid_data = pickle.load(rf)

    logging.info("Pretrain dataset ratio: %.3f; Train dataset ratio: %.3f", args.pretrain_ratio, args.train_ratio)
    _num = len(train_data)

    real_data = train_data[:int(_num * args.train_ratio)]

    real_data = LFArchDataset(real_data, args.low_fidelity_type)
    train_data = train_data[:int(_num * args.pretrain_ratio)]
    train_data = LFArchDataset(train_data, args.low_fidelity_type)
    valid_data = LFArchDataset(valid_data, args.low_fidelity_type)

    logging.info("Number of architectures: pre-train: %d; train: %d; valid: %d", \
            len(train_data), len(real_data), len(valid_data))

    train_loader = DataLoader(
        train_data, batch_size = args.batch_size, shuffle = True, pin_memory = True, 
        num_workers = args.num_workers, collate_fn = lambda items: list(zip(*items)))
    val_loader = DataLoader(
        valid_data, batch_size = args.batch_size, shuffle = False, pin_memory = True, 
        num_workers = args.num_workers, collate_fn = lambda items: list(zip(*items)))
    real_loader = DataLoader(
        real_data, batch_size = args.batch_size, shuffle = True, pin_memory = True, 
        num_workers = args.num_workers, collate_fn = lambda items: list(zip(*items)))

    # ---- End of Preparation & Setups ----

    # ---- Predictor Initialization ----

    model = PointwiseComparator(
        arch_embedder_cls = NasBench201_LSTMSeqEmbedder, # specify the architecture encoder type
        **cfg.pop("arch_network_cfg")
    )

    if args.load is not None:
        logging.info("Load checkpoint from {}".format(args.load))
        model.load(args.load)
    model.to(device)

    # ---- End of Predictor Initialization ----


    # ---- Initial Test ----

    if args.test_only:
        lf_corr, real_corr, patk = utils.valid(val_loader, model)
        logging.info("Valid: kendall tau {} {:.4f}; real {:.4f}; patk {}".format(
            args.low_fidelity_type, low_fidelity_corr, real_corr, patk))
        return

    # ---- End of Initial Test ----


    # ---- Training Step 1 ----
    # Pre-train the predictor with low_fidelity objective

    if not args.no_pretrain:
        for i_epoch in range(1, args.epochs + 1):
            model.on_epoch_start(i_epoch)
            avg_loss = utils.train(
                train_loader, 
                model, 
                i_epoch, 
                args.compare, 
                True, 
                args.max_compare_ratio,
                args.compare_threshold, 
                args.choose_pair_criterion
            )
            logging.info("Pre-train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))

            if i_epoch == args.finetune_epochs or i_epoch % args.test_every == 0:
                lf_corr, real_corr, patk = utils.valid(
                    val_loader, model, os.path.join(args.train_dir, "stats.pkl"))
                logging.info(
                    "Pre-valid: Epoch {:3d}: kendall tau {} {:.4f}; real {:.4f}; patk {}".\
                    format(i_epoch, args.low_fidelity_type, lf_corr, real_corr, patk)
                )

        save_path = os.path.join(args.train_dir, "pre_train.ckpt")
        model.save(save_path)
        logging.info("Finish pre-training. Save checkpoint to {}".format(save_path))

    # ---- End of Training Step 1 ----


    # ---- Training Step 2 ----
    # Finetune the predictor with the actual performance

    model.reinit_optimizer()
    model.reinit_scheduler()

    for i_epoch in range(1, args.finetune_epochs + 1):
        model.on_epoch_start(i_epoch)
        avg_loss = utils.train(
                train_loader, 
                model, 
                i_epoch, 
                args.compare, 
                False, 
                args.max_compare_ratio,
                args.compare_threshold, 
                args.choose_pair_criterion
        )
        logging.info("Train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))

        if i_epoch == args.finetune_epochs or i_epoch % args.test_every == 0:
            lf_corr, real_corr, patk = utils.valid(
                val_loader, model, os.path.join(args.train_dir, "stats.pkl")
            )
            logging.info(
                "Valid: Epoch {:3d}: kendall tau {} {:.4f}; real {:.4f}; patk {}".\
                format(i_epoch, args.low_fidelity_type, lf_corr, real_corr, patk)
            )

    save_path = os.path.join(args.train_dir, "final.ckpt")
    model.save(save_path)
    logging.info("Save checkpoint to {}".format(save_path))

    # ---- End of Training Step 2 ----


if __name__ == "__main__":
    main(sys.argv[1:])
