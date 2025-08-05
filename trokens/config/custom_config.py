#!/usr/bin/env python3

"""Add custom configs and default values"""
from fvcore.common.config import CfgNode

#pylint: disable=line-too-long

def add_custom_config(cfg):
    """Add custom configs."""
    cfg.DATA.PATH_TO_TROKEN_PT_DATA = '/fs/cfar-projects/actionloc/bounce_back/camera_ready/data/trokens_pt_data/'

    #wandb config
    cfg.WANDB = CfgNode()
    cfg.WANDB.PROJECT = 'trokens'
    cfg.WANDB.ENTITY = 'act_seg_pi_umd'
    cfg.WANDB.ID = ''
    cfg.WANDB.EXP_NAME = ''

    # few shot config
    cfg.FEW_SHOT = CfgNode()
    cfg.FEW_SHOT.N_WAY = 5
    cfg.FEW_SHOT.K_SHOT = 1
    cfg.FEW_SHOT.TRAIN_QUERY_PER_CLASS = 6
    cfg.FEW_SHOT.TEST_QUERY_PER_CLASS = 1
    cfg.FEW_SHOT.TRAIN_EPISODES = 100000
    cfg.FEW_SHOT.TEST_EPISODES = 10000
    cfg.FEW_SHOT.PATCH_TOKENS_AGG = 'spatial'
    cfg.FEW_SHOT.USE_MODEL = True
    cfg.FEW_SHOT.DIST_NORM = 'none'
    cfg.FEW_SHOT.TRAIN_OG_EPISODES = False
    cfg.FEW_SHOT.CLASS_LOSS_LAMBDA = 1.0
    cfg.FEW_SHOT.Q2S_LOSS_LAMBDA = 1.0

    # point info config
    cfg.POINT_INFO = CfgNode()
    cfg.POINT_INFO.ENABLE = True
    cfg.POINT_INFO.GRID_SIZE = 16
    cfg.POINT_INFO.NAME = ''
    cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE = 256
    cfg.POINT_INFO.SAMPLING_TYPE = 'random'
    cfg.POINT_INFO.PT_FIX_SAMPLING_TRAIN = False
    cfg.POINT_INFO.PT_FIX_SAMPLING_TEST = False
    cfg.POINT_INFO.USE_PT_QUERY_MASK = False
    cfg.POINT_INFO.OBJ_ID_KEY = 'obj_ids'
    cfg.POINT_INFO.HOD = CfgNode()
    cfg.POINT_INFO.HOD.NUM_BINS = 32
    cfg.POINT_INFO.HOD.NUM_CLUSTERS = 16
    cfg.POINT_INFO.HOD_MIN = True
    cfg.POINT_INFO.HOD.GET_FEAT = True
    cfg.POINT_INFO.HOD.TEMPORAL_PYRAMID = False
    cfg.POINT_INFO.HOD.TEMPORAL_PYRAMID_LEVELS = 3
    cfg.POINT_INFO.HOD.PRESERVE_TEMPORAL = True
    cfg.POINT_INFO.USE_CORRELATION = False

    # motion module config
    cfg.MODEL.MOTION_MODULE = CfgNode()
    cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE = False
    cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE = False
    cfg.MODEL.APPEARANCE_MODULE_DISABLE = False

    return cfg
