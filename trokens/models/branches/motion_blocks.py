"""Motion blocks"""
from collections import OrderedDict
import torch
import torch.nn as nn
from einops import rearrange

# pylint: disable=redefined-builtin
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, input: torch.Tensor):
        """Forward pass for LayerNorm"""
        orig_type = input.dtype
        ret = super().forward(input.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """QuickGELU activation function"""
    def forward(self, x: torch.Tensor):
        """Forward pass for QuickGELU"""
        return x * torch.sigmoid(1.702 * x)


def cross_patch_motion_v1_allneighbors(point_trajs_gt_coord, point_trajs_visibility_mask):
    """
    Computes cross-patch motion features for each spatio-temporal tokens.

    Args:
        point_trajs_gt_coord: Tensor of shape (B, M, T, 2) -> M trajectories
                                point coordinates, normalized between [-1, 1]
        point_trajs_visibility_mask: Tensor of shape (B, M, T) -> 1, Point visibility

    Returns:
        cross_path_motion_feature: Tensor of shape (B, T, M, D), D = M*2 for all neighbors

    """
    batch_size, num_points, temporal_len, _ = point_trajs_gt_coord.shape
    assert point_trajs_visibility_mask.shape == (batch_size, num_points, temporal_len)

    # Find indices where NaNs occur in point_trajs_gt_coord
    nan_indices = torch.isnan(point_trajs_gt_coord)

    # Set NaN values in point_trajs_gt_coord to 0
    point_trajs_gt_coord = torch.nan_to_num(point_trajs_gt_coord, nan=0.0)

    # Set corresponding entries in point_trajs_visibility_mask to 0
    # Only need to check one coordinate for NaN
    point_trajs_visibility_mask[nan_indices[..., 0]] = 0

    # (B, M, T, 2) -> (B, M, T, 1, 2)
    tensor_centers = point_trajs_gt_coord.unsqueeze(3)


    tensor_neighbors = point_trajs_gt_coord.permute(0, 2, 1, 3).unsqueeze(1)

    # (B, M, T, M, 2)
    distances_relative = tensor_centers - tensor_neighbors

    # Create a visibility mask for each pair
    vis_mask_centers = point_trajs_visibility_mask.unsqueeze(3)
    vis_mask_neighbors = point_trajs_visibility_mask.permute(0, 2, 1).unsqueeze(1)
    vis_mask_pair = vis_mask_centers * vis_mask_neighbors

    # Apply the mask to the relative distances
    # (B, M, T, M, 2)
    distances_relative = distances_relative * vis_mask_pair.unsqueeze(-1)

    # (B, M, T, M, 2) -> (B, M, T, D)
    feature_dim = num_points * 2
    cross_path_motion_fea = distances_relative.reshape(batch_size, num_points,
                                                        temporal_len, feature_dim)

    return cross_path_motion_fea

class CrossMotionModule(nn.Module):
    """Cross motion module"""
    def __init__(
        self,
        out_feature_dim =768,
        num_patches = 256,
        in_fea_dim_crossmotion = 256,

    ):
        super().__init__()
        self.num_patches = num_patches # we don't need cls token

        self.ln2 = LayerNorm(out_feature_dim)

        self.fc2 =nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(in_fea_dim_crossmotion, 4*out_feature_dim)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(4*out_feature_dim, out_feature_dim))
        ]))

        #self.fc2 =nn.Linear(in_fea_dim_crossmotion, out_feature_dim)


        print('verbose...CrossMotionModules')


    def forward(self, point_trajs_gt_coord, point_trajs_visibility_mask):
        '''
        Args:
        point_trajs_gt_coord: Tensor of shape (B, T, M, 2) -> M trajectories point coordinates,
                                normalized between [-1, 1]
        point_trajs_visibility_mask: Tensor of shape (B, T, M) -> Point visibility mask
        Returns:
            cross_path_motion_feature: Tensor of shape (B, T, M, D), D = M*2 for all neighbors

        '''

        point_trajs_gt_coord = rearrange(point_trajs_gt_coord, 'b t m d -> b m t d')
        batch_size, num_points, temporal_len, _ = point_trajs_gt_coord.shape
        point_trajs_visibility_mask = rearrange(point_trajs_visibility_mask, 'b t m -> b m t')
        assert point_trajs_visibility_mask.shape == (batch_size, num_points, temporal_len)

        #### range assertion
        # Check if all values are between 0 and 1, inclusive
        assert (point_trajs_visibility_mask.max() <= 1.0 and \
            point_trajs_visibility_mask.min() >= 0.0), "Tensor values outside the range [0, 1]."

        ######### handle NAN here
        # Find indices where NaNs occur in point_trajs_gt_coord
        nan_indices = torch.isnan(point_trajs_gt_coord)

        # Set NaN values in point_trajs_gt_coord to 0
        point_trajs_gt_coord = torch.nan_to_num(point_trajs_gt_coord, nan=0.0)

        # Set corresponding entries in point_trajs_visibility_mask to 0
        # Only need to check one coordinate (x or y) for NaN
        point_trajs_visibility_mask[nan_indices[..., 0]] = 0


      ########## self motion delta within a trajectory
        point_trajs_delta_coords_full = cross_patch_motion_v1_allneighbors(
            point_trajs_gt_coord, point_trajs_visibility_mask)
        selfmotion_feas = self.ln2(
                    self.fc2(point_trajs_delta_coords_full))
        motion_out_feas = rearrange(selfmotion_feas, 'b m t d -> b t m d')

        return motion_out_feas


class HODMotionModule(nn.Module):
    """HOD motion module"""
    def __init__(
        self,
        out_feature_dim =768,
        num_patches = 256,
        in_feature_dim = 32,
    ):
        super().__init__()
        self.num_patches = num_patches # we don't need cls token

        self.ln2 = LayerNorm(out_feature_dim)

        self.fc2 =nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(in_feature_dim, 4*out_feature_dim)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(4*out_feature_dim, out_feature_dim))
        ]))

        print('verbose...HODMotionModule')


    def forward(self, hod_feat):
        '''
        Args:
        hod_feat: (B, M, T, C), C = num_bins
        Returns:
            (B, T, M, D)

        '''


      ########## self motion delta within a trajectory
        hod_motion_feas = self.ln2(self.fc2(hod_feat)) #  # (B, M, T, D4)
        motion_out_feas = hod_motion_feas.permute(0, 2, 1, 3)  # (B, T, M, D4)

        return motion_out_feas
