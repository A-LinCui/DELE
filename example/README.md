## A Tutorial on NAS-Bench-201 Search Space
--------

There are several python files and `cfgs` dir under this directory. The python files are:

* `train_predictor.py`: Directly train the predictor on the actual performance data, or pretrain the predictor on a single type of low-fidelity information data and finetune on the actual performanda data.
* `dynamic_ensemble_train_predictor.py`: Conduct predictor training with the proposed dynamic ensemble framework.

### Data Preparation
Download the data from [here](https://drive.google.com/drive/folders/1bdqGXNioj0xp_P6TmLiEHScP3aCj4iGl?usp=sharing).

### Run Predictor Training
To run the proposed dynamic ensemble predictor training framework, use the `dynamic_ensemble_train_predictor.py` script. Specifically, run `python dynamic_ensemble_train_predictor.py <CFG_FILE> --train-ratio <TRAIN_RATIO> --pretrain-ratio <PRETRAIN_RATIO> --train-pkl <TRAIN_PKL> --valid-pkl <VALID_pkl> --seed <SEED> --gpu <GPU_ID> --train-dir <TRAIN_DIR>`, where:

* `CFG_FILE`: Path of the configuration file
* `TRAIN_RATIO`: Proportion of training samples used in the second-step training
* `PRETRAIN_RATIO`: Proportion of training sampled used in the first-step training. Default: 1.0
* `TRAIN_PKL`: Path of the training data
* `VALID_PKL`: Path of the validation data
* `SEED`: Seed (optional)
* `GPU_ID`: ID of the used GPU. Currently, we only support single-GPU training. Default: 0
* `TRAIN_DIR`: Path to save the logs and results

To run the pretrain-and-finetune predictor training flow, use the `train_predictor.py` script. Specifically, run `python train_predictor.py <CFG_FILE> --train-ratio <TRAIN_RATIO> --pretrain-ratio <PRETRAIN_RATIO> --train-pkl <TRAIN_PKL> --valid-pkl <VALID_pkl> --low-fidelity-type <LOW_FIDELITY_TYPE> --seed <SEED> --gpu <GPU_ID> --train-dir <TRAIN_DIR>`, where:

* `LOW_FIDELITY_TYPE`: Type of the utilized low-fidelity information. Default: one_shot

By set `--no-pretrain`, we can run the vanilla predictor training flow without utilizing low-fidelity information. 

We provide example predictor training configuration files under `./cfgs`, including:

* `train_nb201_lstm_config.yaml`:
  * Encoder: LSTM
  * Objective: Ranking loss
  * Method: Vanilla or the pretrain & finetune method as described in Introduction.
* `dynamic_ensemble_nb201_lstm_config.yaml`
  * Encoder: LSTM
  * Objective: Ranking loss (DELE only support ranking loss)
  * Method: Dynamic ensemble
  
For example, run dynamic ensemble predictor training on NAS-Bench-201 with 100% training samples by

`python dynamic_ensemble_train_predictor.py cfgs/dynamic_ensemble_nb201_lstm_config.yaml --train-ratio 1.0 --train-pkl data/NAS-Bench-201/nasbench201_train.pkl --valid-pkl data/NAS-Bench-201/nasbench201_valid.pkl --train-dir <TRAIN_DIR>`

### Available Types of Low-fidelity Information
Available types of low-fidelity information for NAS-Bench-201 include: grad_norm, snip, grasp, fisher, jacob_cov, plain, synflow, flops, params, relu, relu_logdet, one_shot, latency.
