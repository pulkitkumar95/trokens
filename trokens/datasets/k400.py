#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Kinetics400 dataset"""

import os
import pandas as pd
import trokens.utils.logging as logging
from .base_ds import BaseDataset
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class K400(BaseDataset):
    """Kinetics400 dataset"""
    def __init__(self, cfg, mode):
        super(K400, self).__init__(cfg, mode)
    def _construct_loader(self):
        """Construct the video loader.
        """

        self.data_root = self.cfg.DATA.PATH_TO_DATA_DIR
        csv_name_to_use = 'kinetics100.csv'
        self.dataset_csv_path = os.path.join(self.splits_root, csv_name_to_use)
        self.dataset_df = pd.read_csv(self.dataset_csv_path)
        self.dataset_df['video_path'] = self.dataset_df['vid_base_path'].apply(
                                    lambda x: os.path.join(self.data_root, x))
        #video name might contain time info, so we only keep the first 11 characters
        # corresponding to the youtube id
        self.dataset_df['video_name'] = self.dataset_df['video_path'].apply(
                                lambda x: os.path.basename(x).split('.')[0][:11])
        self.dataset_df['feat_base_name'] = self.dataset_df['video_name'].apply(
                                lambda x: x + '.pkl')
        self.split_df = self.dataset_df[
                        self.dataset_df['split'] == self.mode].reset_index(drop=True)
        self._path_to_videos = []
        self.split_df['feat_path'] = self.split_df['feat_base_name'].apply(
                                lambda x: os.path.join(self.base_feature_path, x))
        self._make_final_lists()
