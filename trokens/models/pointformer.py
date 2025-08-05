"""Pointformer model."""
import os
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from sympy import divisors
import numpy as np
from torch.nn.init import trunc_normal_

from trokens.models.attention import TrajectoryAttentionBlock
from trokens.models.branches.motion_blocks import (
    CrossMotionModule,
    HODMotionModule
)
from trokens.datasets.hod import get_orientation_hist
from .build import MODEL_REGISTRY

# pylint: disable=unused-argument,redefined-builtin

@MODEL_REGISTRY.register()
class Pointformer(nn.Module):
    """ Main model for point tracking based transformer model.
    """
    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.DATA.TRAIN_CROP_SIZE
        # self.patch_size = cfg.MF.PATCH_SIZE
        if cfg.MODEL.FEAT_EXTRACTOR == "dino":
            dino_config  = cfg.MODEL.DINO_CONFIG
            vit_mode = dino_config.split("_")[1]
            if 'vits' in vit_mode:
                vit_type = 'vits'
                self.embed_dim = self.dino_feat_size = 384
            elif 'vitb' in vit_mode:
                vit_type = 'vitb'
                self.embed_dim = self.dino_feat_size = 768
            elif 'vitl' in vit_mode:
                vit_type = 'vitl'
                self.embed_dim = self.dino_feat_size = 1024

            else:
                raise NotImplementedError("Only supports ViT-B and ViT-S for DINO")
            self.patch_size = int(vit_mode.replace(vit_type, ""))
        else:
            raise NotImplementedError('Feature extractor not implemented')


        self.in_chans = cfg.MF.CHANNELS
        if cfg.TRAIN.DATASET == "epickitchens" and cfg.TASK == 'classification':
            self.num_classes = [97, 300]
        else:
            self.num_classes = cfg.MODEL.NUM_CLASSES

        self.depth = cfg.MF.DEPTH
        self.num_heads = cfg.MF.NUM_HEADS
        self.mlp_ratio = cfg.MF.MLP_RATIO
        self.qkv_bias = cfg.MF.QKV_BIAS
        self.drop_rate = cfg.MF.DROP
        self.drop_path_rate = cfg.MF.DROP_PATH
        self.head_dropout = cfg.MF.HEAD_DROPOUT
        self.video_input = cfg.MF.VIDEO_INPUT
        self.temporal_resolution = cfg.DATA.NUM_FRAMES
        self.use_mlp = cfg.MF.USE_MLP
        self.num_features = self.embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.attn_drop_rate = cfg.MF.ATTN_DROPOUT
        self.head_act = cfg.MF.HEAD_ACT
        self.cfg = cfg
        self.num_patches = (224 // self.patch_size) ** 2
        if cfg.POINT_INFO.ENABLE:
            self.point_grid_size = self.get_point_grid_size()

        else:
            self.point_grid_size = int(self.num_patches ** 0.5)

        # CLS token
        if cfg.MODEL.USE_CLS_TOKEN:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(self.cls_token, std=.02)
        else:
            self.cls_token = nn.Identity()

        # # Positional embedding

        self.pos_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)

        dpr = [x.item() for x in torch.linspace(
            0, self.drop_path_rate, self.depth)]
        ##
        blocks = []
        for i in range(self.depth):
            # pt_attention is introduced, for now its just space-time attention
            _block = TrajectoryAttentionBlock(
                cfg = cfg,
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                pt_attention=cfg.MF.PT_ATTENTION,
                use_pt_visibility=cfg.MF.USE_PT_VISIBILITY or cfg.POINT_INFO.USE_PT_QUERY_MASK,
                num_mlp_layers=cfg.MF.NUM_MLP_LAYERS,
            )

            blocks.append(_block)
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(self.embed_dim)
        if self.cfg.MODEL.APPEARANCE_MODULE_DISABLE:
            assert (
            self.cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE or
            self.cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE), "One motion module must be enabled"

        # MLP head
        if self.use_mlp:
            hidden_dim = self.embed_dim
            if self.head_act == 'tanh':
                print("Using TanH activation in MLP")
                act = nn.Tanh()
            elif self.head_act == 'gelu':
                print("Using GELU activation in MLP")
                act = nn.GELU()
            else:
                print("Using ReLU activation in MLP")
                act = nn.ReLU()
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, hidden_dim)),
                ('act', act),
            ]))
        else:
            self.pre_logits = nn.Identity()
            self.agg_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(self.agg_cls_token, std=.02)


        # Classifier Head
        self.head_drop = nn.Dropout(p=self.head_dropout)
        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            for a, i in enumerate(range(len(self.num_classes))):
                setattr(self, f'head{a}', nn.Linear(
                                        self.embed_dim, self.num_classes[i]))
        else:
            self.head = (nn.Linear(self.embed_dim, self.num_classes)
                if self.num_classes > 0 else nn.Identity())
        self.dino_num_patch_side = 224 // self.patch_size
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.embed_dim, self.dino_num_patch_side,
                                                        self.dino_num_patch_side))

        trunc_normal_(self.spatial_pos_embed, std=.02)
        if cfg.MODEL.FEAT_EXTRACTOR == 'resnet':
            #TODO(pulkit): Remove hard coding
            self.space_pos_embed = nn.Parameter(torch.zeros(1,49, self.embed_dim))
        else:
            self.space_pos_embed = nn.Parameter(
                                torch.zeros(1,self.num_patches, self.embed_dim))

        self.time_pos_embed = nn.Parameter(
                        torch.zeros(1,self.cfg.DATA.NUM_FRAMES, self.embed_dim))
        trunc_normal_(self.space_pos_embed, std=.02)
        trunc_normal_(self.time_pos_embed, std=.02)
        self.space_pos_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)
        self.time_pos_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)



        self.spatial_pos_embed_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)
        self.layer_to_use = None

        # Initialize weights
        self.init_weights()
        self.apply(self._init_weights)
        if cfg.MODEL.FEAT_EXTRACTOR == "dino":
            dino_config  = cfg.MODEL.DINO_CONFIG
            local_path = os.path.join(os.environ['TORCH_HOME'], 'hub')
            if 'v2' in dino_config:
                local_path = os.path.join(local_path , 'facebookresearch_dinov2_main')
                self.dino = torch.hub.load(local_path, dino_config, source='local')
            else:
                local_path = os.path.join(local_path , 'facebookresearch_dino_main')
                self.dino = torch.hub.load(local_path, dino_config, source='local')
                self.dino.num_register_tokens = 0

            self.feat_dict = dict()
            #output of last norm to be taken.
            layer = self.dino.norm
            self.hook = layer.register_forward_hook(self.hook_fn(self.feat_dict, 'dino'))

            self.dino.cuda()
            # Set all DINO parameters to not require gradients
            for param in self.dino.parameters():
                param.requires_grad = False

        else:
            raise NotImplementedError('Feature extractor not implemented')

        if cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE:
            self.cross_motion_module = CrossMotionModule(
                out_feature_dim=self.embed_dim,
                num_patches=self.num_patches,
                in_fea_dim_crossmotion=2*self.cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE,
            )

        if cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE:
            if cfg.POINT_INFO.HOD.TEMPORAL_PYRAMID:
                #TODO(pulkit): make this dynamic
                assert cfg.POINT_INFO.HOD.TEMPORAL_PYRAMID_LEVELS == 3, 'hard coded for now'
                feat_dim = cfg.POINT_INFO.HOD.NUM_BINS * 7
            else:
                feat_dim = cfg.POINT_INFO.HOD.NUM_BINS
            self.hod_motion_module = HODMotionModule(
                in_feature_dim=feat_dim,
                out_feature_dim=self.embed_dim,
                num_patches=self.num_patches,
            )

    def hook_fn(self, feat_dict, layer_name):
        """Hook function to extract features of specific layers"""
        def hook(module, input, output):
            # Store the extracted features as an attribute of the model
            feat_dict[layer_name] = output
        return hook

    def backward_hook_fn(self, feat_dict, layer_name):
        """Backward hook function for extracting gradients"""
        def hook(module, grad_inputs, grad_outputs):
            # Store the extracted features as an attribute of the model
            grad_out_norm = np.mean([torch.norm(grad_output).item()
                                        for grad_output in grad_outputs])
            grad_in_norm = np.mean([torch.norm(grad_input).item()
                                        for grad_input in grad_inputs])
            feat_dict[layer_name] = {
                'grad_out': np.around(grad_out_norm, 3),
                'grad_in': np.around(grad_in_norm, 3)
            }
        return hook


    def init_weights(self):
        """Initialize weights"""
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """No weight decay params"""
        if self.cfg.MF.POS_EMBED == "joint":
            return {'pos_embed', 'cls_token', 'st_embed'}
        else:
            return {'pos_embed', 'cls_token', 'temp_embed'}

    def get_classifier(self):
        """Get classifier"""
        return self.head


    def get_point_grid_size(self):
        """Get point grid size"""
        all_divisors = divisors(self.cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE)
        return all_divisors[len(all_divisors) // 2]


    def get_dino_features(self, x):
        """ Get DINO features
        Args:
            x (torch.Tensor): Input features of shape [BS, T, C, H, W]

        Returns:
            torch.Tensor: DINO features of shape [BS, T, P, Q, D]
        """
        self.dino.eval()
        batch_size, num_frames, channel, height, width = x.shape
        x = x.view(-1, channel, height, width)
        if self.cfg.MODEL.TRAIN_BACKBONE:
            _ = self.dino(x)
        else:
            with torch.no_grad():
                _ = self.dino(x)
        #using hooks to get the patch tokens
        feat = self.feat_dict['dino'][:, self.dino.num_register_tokens + 1 :]
        feat_size = feat.shape[-1]
        #dino patch side is fine
        feat = feat.view(batch_size, num_frames, self.dino_num_patch_side,
                         self.dino_num_patch_side, feat_size)

        return feat


    def pt_forward(self, x, metadata):
        """ Forward pass for point tracking based transformer model.

        Args:
            x (torch.Tensor): Input features of shape [BS, T, N, D]
            metadata (dict): Metadata containing prediction masks

        Returns:
            torch.Tensor: Class token features
            torch.Tensor: Patch token features
        """
        if self.cfg.POINT_INFO.USE_PT_QUERY_MASK:
            pt_mask = metadata['pred_query_mask'] # [BS, T, N]
        else:
            pt_mask = metadata['pred_visibility'] # [BS, T, N]

        bs, temporal_dim, num_points, _ = x.shape
        # reshaping the input according to the attention block
        x = rearrange(x, 'b t n d -> b n t d')
        pt_mask = rearrange(pt_mask, 'b t n -> b n t')
        x = rearrange(x, 'b n t d -> b (n t) d')
        pt_mask = rearrange(pt_mask, 'b n t -> b (n t)')
        if self.cfg.MODEL.USE_CLS_TOKEN:
            cls_tokens = self.cls_token.expand(bs, -1, -1) # [BS, 1, dim]
            x = torch.cat((cls_tokens, x), dim=1) # [BS, N, dim]
            cls_token_mask = torch.ones(bs, 1).bool().to(x.device)
            pt_mask = torch.cat((cls_token_mask, pt_mask), dim=1) # [BS, N, dim]
        # Apply positional dropout
        x = self.pos_drop(x) # [BS, N, dim]
        # Encoding using transformer layers
        thw = [self.temporal_resolution, self.point_grid_size,
            int(num_points / self.point_grid_size)]
        for _, blk in enumerate(self.blocks):
            x, _ = blk(
                x,
                thw,
                pt_mask
            )
        if self.cfg.MODEL.ADAPOOLING.ENABLE:
            if self.cfg.MODEL.ADAPOOLING.TYPE == 'temporal_spatial':
                extra_cls_token =  self.agg_cls_token.expand(bs * num_points, -1, -1)
                cls_x, patch_x = self.adaptive_pooling(x, extra_cls_token)
                patch_x = rearrange(patch_x, 'b n d -> b 1 n d')

            elif self.cfg.MODEL.ADAPOOLING.TYPE == 'spatial_temporal':
                extra_cls_token = self.agg_cls_token.expand(bs * temporal_dim, -1, -1)
                cls_x, patch_x = self.adaptive_pooling(x, extra_cls_token)
                patch_x = rearrange(patch_x, 'b t d -> b t 1 d')

        else:
            x = self.norm(x)
            if self.cfg.MODEL.USE_CLS_TOKEN:
                cls_x, patch_x = x[:, 0], x[:, 1:]
                if self.cfg.MODEL.USE_PATCH_AS_CLS:
                    cls_x = patch_x.mean(dim=1)
            else:
                # If cls token is not ued, for now using global average pooling
                cls_x = x.mean(dim=1)
                patch_x = x

            cls_x = self.pre_logits(cls_x)
            # Taking the patch tokens back to the input shape
            patch_x = rearrange(patch_x, 'b (n t) d -> b t n d', t=temporal_dim)
        if not torch.isfinite(x).all():
            print("WARNING: nan in features out")
        return cls_x, patch_x

    def add_st_pos_embeddings(self, x):
        """ Add spatial and temporal positional embeddings to the input features.

        Args:
            x (torch.Tensor): Input features of shape [BS, T, P, Q, D]

        Returns:
            torch.Tensor: Output features of shape [BS, T, P, Q, D]
        """
        _, _, sp_dim_1, sp_dim_2, _ = x.shape
        # reshaping the input according to the attention block
        x = rearrange(x, 'b t p q d -> b t (p q) d')
        x = x + self.space_pos_embed.unsqueeze(0)
        x = self.space_pos_drop(x)
        x = rearrange(x, 'b t p d -> b p t d')
        x = x + self.time_pos_embed.unsqueeze(0)
        x = self.time_pos_drop(x)
        x = rearrange(x, 'b (p q) t d -> b t p q d', p=sp_dim_1, q=sp_dim_2)
        return x

    def forward(self, input_to_use):
        """Forward pass of the model"""
        x = input_to_use['video']
        metadata = input_to_use['metadata']

        if 'skip_feat_extractor' in input_to_use:
            skip_feat_extractor = input_to_use['skip_feat_extractor']
        else:
            skip_feat_extractor = False
        if not self.cfg.MODEL.APPEARANCE_MODULE_DISABLE:
            if skip_feat_extractor:
                embed_dim = self.embed_dim
                batch_size, num_frames = x.shape[:2]
                feat_to_use = torch.randn(batch_size, num_frames,
                                            int(self.num_patches**0.5),
                                            int(self.num_patches**0.5),
                                            embed_dim).to(x.device)
            else:
                if self.cfg.MODEL.FEAT_EXTRACTOR == "dino":
                    feat_to_use = self.get_dino_features(x)
                    if self.cfg.POINT_INFO.USE_CORRELATION:
                        # for ablation study without point tracking module
                        new_metadata = get_points_using_correlation(self.cfg, feat_to_use)
                        metadata.update(new_metadata)

                else:
                    raise NotImplementedError('Feature extractor not implemented')

            if self.cfg.MF.USE_BASE_POS_EMBED:
                feat_to_use = self.add_st_pos_embeddings(feat_to_use)


            if self.cfg.POINT_INFO.ENABLE:
                bs, num_frames = feat_to_use.shape[:2]
                feat_to_use = rearrange(feat_to_use, 'b t p q d -> (b t) p q d')
                feat_to_use = rearrange(feat_to_use, 'b p q d -> b d p q')
                num_x, num_y = feat_to_use.shape[-2:]
                assert self.num_patches == num_x * num_y, "Number of patches mismatch"
                pred_tracks = metadata['pred_tracks']
                pred_tracks = pred_tracks.view(bs * num_frames, -1,1,2)
                spatial_pos_embed = self.spatial_pos_embed.repeat(bs * num_frames, 1, 1, 1)
                sampled_feat = F.grid_sample(
                    feat_to_use,
                    pred_tracks,
                    align_corners=True,
                    mode=self.cfg.MODEL.FEAT_EXTRACT_MODE,
                )
                if (self.cfg.MF.USE_PT_SPACE_POS_EMBED and self.cfg.FEW_SHOT.USE_MODEL
                    and not self.cfg.MF.USE_BASE_POS_EMBED):
                    sample_pos_embedding = F.grid_sample(
                        spatial_pos_embed,
                        pred_tracks,
                        align_corners=True,
                        mode='bilinear',
                    )
                    sampled_feat = sampled_feat + sample_pos_embedding
                sampled_feat = rearrange(sampled_feat, 'b d p q -> b p q d')
                #Removing the extra added dim
                sampled_feat = sampled_feat.squeeze(-2)
                sampled_feat = rearrange(sampled_feat, '(b t) p d -> b t p d', t=num_frames)

            else:
                sampled_feat = rearrange(feat_to_use, 'b t p q d -> b t (p q) d')
                self.point_grid_size = int(sampled_feat.shape[2] ** 0.5)
        else:
            sampled_feat = 0

        if self.cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE:
            hod_motion_feat = self.hod_motion_module(metadata['hod_feat'].float())
            sampled_feat = sampled_feat + hod_motion_feat

        if self.cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE:
            cross_motion_feat = self.cross_motion_module(
                metadata['pred_tracks'], metadata['pred_visibility'])
            sampled_feat = sampled_feat + cross_motion_feat

        cls_x, patch_x = self.pt_forward(sampled_feat, metadata)
        # x = self.forward_features(x, metadata) # [BS, d]
        x = self.head_drop(cls_x)


        x = self.head(x)
        # previously there was a softmax here for validation which messed up the loss computation
        if self.cfg.TASK == 'few_shot':
            return x, patch_x
        return x


def compute_correlation_map(features):
    """
    Compute correlation map between consecutive frames
    Args:
        features: Tensor of shape [bs, num_frames, num_patches, d]
    Returns:
        correlation: Tensor of shape [bs, num_frames-1, num_patches, num_patches]
    """
    bs, num_frames, num_patches, d = features.shape

    # Get features for all frames except last one
    feat1 = features[:, :-1]  # [bs, num_frames-1, num_patches, d]

    # Get features for all frames except first one
    feat2 = features[:, 1:]   # [bs, num_frames-1, num_patches, d]

    # Normalize the features
    feat1_norm = F.normalize(feat1, p=2, dim=-1)
    feat2_norm = F.normalize(feat2, p=2, dim=-1)

    # Compute correlation through matrix multiplication
    # Reshape for batch matrix multiplication
    feat1_reshaped = feat1_norm.view(bs * (num_frames-1), num_patches, d)
    feat2_reshaped = feat2_norm.view(bs * (num_frames-1), num_patches, d)

    # Compute correlation [bs*(num_frames-1), num_patches, num_patches]
    correlation = torch.bmm(feat1_reshaped, feat2_reshaped.transpose(1, 2))

    # Reshape back to [bs, num_frames-1, num_patches, num_patches]
    correlation = correlation.view(bs, num_frames-1, num_patches, num_patches)

    return correlation


def get_points_using_correlation(cfg, features):
    """
    Get points using correlation
    Args:
        cfg (dict): Configuration dictionary
        features (torch.Tensor): Features of shape [BS, T, P, Q, D]

    Returns:
        dict: Metadata containing prediction tracks and visibility
    """
    features = rearrange(features, 'b t p q d -> b t (p q) d')
    bs, num_frames, num_patches, _ = features.shape

    # Compute basic correlation first
    correlation = compute_correlation_map(features)

    # Apply mutual matching
    corr_b = correlation.view(bs*(num_frames-1), num_patches, num_patches)
    corr_a = corr_b

    # Get max values
    corr_b_max, _ = torch.max(corr_b, dim=1, keepdim=True)
    corr_a_max, _ = torch.max(corr_a, dim=2, keepdim=True)

    # Normalize by max values
    eps = 1e-5
    corr_b = corr_b / (corr_b_max + eps)
    corr_a = corr_a / (corr_a_max + eps)

    # Compute mutual correlation
    mutual_correlation = correlation * (corr_a.view_as(correlation) * corr_b.view_as(correlation))
    _, max_indices = torch.max(mutual_correlation, dim=-1)
    extra_indices = torch.arange(max_indices.shape[-1]).to(max_indices.device)
    extra_indices = extra_indices.unsqueeze(0).unsqueeze(0).expand(max_indices.shape[0], -1, -1)
    max_indices = torch.cat([extra_indices, max_indices], dim=1)
    if cfg.MODEL.DINO_CONFIG=="dinov2_vitb14":
        grid_points = create_normalized_grid(image_size=224, grid_size=16)
    else:
        raise NotImplementedError(f'Grid points dim not set for {cfg.MODEL.DINO_CONFIG}')
    grid_points = grid_points.to(max_indices.device)
    grid_points = rearrange(grid_points, 'n d -> 1 1 n d')
    grid_points = repeat(grid_points, '1 1 n d -> b t n d', b=bs, t=num_frames)
    new_metadata = sample_grid_points_with_indices(cfg, grid_points, max_indices)

    return new_metadata

def create_normalized_grid(image_size=224, grid_size=16):
    """
    Create a normalized grid of points from an image
    Args:
        image_size (int): Size of the image (assuming square image)
        grid_size (int): Size of the grid (e.g., 16 for 16x16 grid)
    Returns:
        grid_points: Tensor of shape [grid_size*grid_size, 2] containing normalized x,y coordinates
    """
    # Create linear spaces for x and y coordinates
    grid_step = image_size // grid_size
    points = torch.arange(grid_step // 2, image_size, grid_step)

    # Create meshgrid
    y, x = torch.meshgrid(points, points, indexing='ij')

    # Reshape to [N, 2] where N = grid_size * grid_size
    grid_points = torch.stack([x, y], dim=-1).reshape(-1, 2)

    # Normalize to [-1, 1]
    grid_points = 2 * (grid_points / (image_size - 1)) - 1

    return grid_points


def sample_grid_points_with_indices(cfg, grid_points, max_indices):
    """
    Sample grid points using max correlation indices
    Args:
        grid_points: Tensor of shape [bs, num_frames, num_patches, 2]
        max_indices: Tensor of shape [bs, num_frames, num_patches]
    Returns:
        sampled_points: Tensor of shape [bs, num_frames, num_patches, 2]
    """
    bs, num_frames, num_patches, _ = grid_points.shape

    # Create batch indices for gathering
    batch_indices = torch.arange(bs).view(-1, 1, 1).expand(-1, num_frames, num_patches)
    frame_indices = torch.arange(num_frames).view(1, -1, 1).expand(bs, -1, num_patches)

    # Gather points using indices
    sampled_points = grid_points[batch_indices, frame_indices, max_indices]
    new_metadata = {}
    if cfg.POINT_INFO.HOD.GET_FEAT:
        sampled_points_for_hod = rearrange(sampled_points, 'b t n d -> b n t d')
        hod_feats = []
        for i in range(bs):
            hod_feat = torch.tensor(get_orientation_hist(
                                        sampled_points_for_hod[i].cpu().numpy(),
                                        cfg.POINT_INFO.HOD.NUM_BINS,
                                        preserve_temporal=True))
            hod_feats.append(hod_feat.unsqueeze(0))
        hod_feats = torch.cat(hod_feats, dim=0).to(grid_points.device)
        new_metadata['hod_feat'] = hod_feats
    new_metadata['pred_tracks'] = sampled_points
    new_metadata['pred_visibility'] = torch.ones_like(
                            sampled_points).bool()[...,0].to(grid_points.device)
    new_metadata['pred_query_mask'] = torch.ones_like(
                            sampled_points).bool()[...,0].to(grid_points.device)


    return new_metadata
