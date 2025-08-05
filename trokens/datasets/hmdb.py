#!/usr/bin/env python3

"""HDMB dataset"""

import os
import pandas as pd

import trokens.utils.logging as logging

from .build import DATASET_REGISTRY

from .base_ds import BaseDataset


logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Hmdb51(BaseDataset):
    """HDMB dataset"""
    def __init__(self, cfg, mode):
        super(Hmdb51, self).__init__(cfg, mode)
    def _construct_loader(self):
        """
        HMDB data       
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries for reading frames from disk.
        """
        self.data_root = self.cfg.DATA.PATH_TO_DATA_DIR
        csv_name_to_use = 'hmdb_few_shot.csv'
        self.dataset_csv_path = os.path.join(self.splits_root, csv_name_to_use)
        self.dataset_df = pd.read_csv(self.dataset_csv_path)
        if self.cfg.TASK == 'few_shot':
            self.dataset_df['video_path'] = self.dataset_df['vid_base_path'].apply(
                                    lambda x: os.path.join(self.data_root, x))
        self.dataset_df['video_name'] = self.dataset_df['video_path'].apply(
                                lambda x: os.path.basename(x).split('.')[0])
        self.dataset_df['feat_base_name'] = self.dataset_df['video_name'].apply(
                                lambda x: x + '.pkl')

        self.split_df = self.dataset_df[
                    self.dataset_df['split'] == self.mode].reset_index(drop=True)
        self._path_to_videos = []
        self.split_df['feat_path'] = self.split_df['feat_base_name'].apply(
                                lambda x: os.path.join(self.base_feature_path, x))
        original_len = len(self.split_df)
        self.split_df = self.split_df[
                    self.split_df['feat_path'].apply(os.path.exists)].reset_index(drop=True)
        new_len = len(self.split_df)
        assert new_len > 0.95 * original_len, "Some features are missing"
        self._make_final_lists()
