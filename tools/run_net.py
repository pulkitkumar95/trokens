#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Wrapper to train and test a video classification model."""
# pylint: disable=wrong-import-position
import os
import sys
sys.path = [x for x  in sys.path if not (os.path.isdir(x) and 'trokens' in os.listdir(x))]
sys.path.append(os.getcwd())
from train_few_shot import train_few_shot
from test_few_shot import test_few_shot
from dist_utils import init_distributed_mode
import trokens
assert trokens.__file__.startswith(os.getcwd()), (f"sys.path: {sys.path}, "
                                                  f"trokens.__file__: {trokens.__file__}")

from trokens.config.defaults import assert_and_infer_cfg
from trokens.utils.misc import launch_job
from trokens.utils.parser import load_config, parse_args



def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    # Set up wandb
    os.environ["WANDB_RUN_GROUP"] = cfg.WANDB.ID
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = cfg.WANDB.ID

    if args.new_dist_init:
        args = init_distributed_mode(args)
    else:
        os.environ["MASTER_PORT"] = str(cfg.MASTER_PORT)


    if cfg.DEBUG:
        os.environ["WANDB_MODE"] = "offline"
        os.environ['DEBUG'] = 'True'
    else:
        os.environ['DEBUG'] = 'False'
    if '$SCRATCH_DIR' in  cfg.DATA.PATH_TO_DATA_DIR:
        cfg.DATA.PATH_TO_DATA_DIR = cfg.DATA.PATH_TO_DATA_DIR.replace(
                                        '$SCRATCH_DIR', os.environ['SCRATCH_DIR'])
    if cfg.CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES

    if cfg.TASK == 'few_shot':
        wandb_run = None
        if cfg.TRAIN.ENABLE:
            wandb_run = launch_job(cfg=cfg, init_method=args.init_method,
                                    func=train_few_shot, args=args)
        if cfg.TEST.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=test_few_shot,
                    args=args, wandb_run=wandb_run)
    else:
        raise ValueError(f"Task {cfg.TASK} not supported")



if __name__ == "__main__":
    main()
