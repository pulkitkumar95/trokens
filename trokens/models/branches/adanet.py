import torch
import torch.nn as nn
from collections import OrderedDict
from einops import rearrange

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class CrossAttentionBlockGenral(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, cfg=None, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)

    def forward(self, query, key, value):
        return self.attn(self.ln_1(query), self.ln_1(key), self.ln_1(value), need_weights=False)[0]

class SpatialTemporalAdaPoolingNetwork(nn.Module):
    def __init__(self, cfg, out_feature_dim):
        super().__init__()
        mlp_ratio = 4
        # if cfg.MODIFY.PT_ATTN.TEMPORAL_AGG is None:
        # if cfg.MODIFY.PT_ATTN.TEMPORAL_AGG == 'None':
        #     self.num_frames = cfg.DATA.NUM_INPUT_FRAMES
        #     self.sparse_sample_alpha = 1
        # elif cfg.MODIFY.PT_ATTN.TEMPORAL_AGG == 'mean':
        #     self.num_frames = 1
        #     self.sparse_sample_alpha = 1

        self.num_frames = cfg.DATA.NUM_FRAMES
        self.sparse_sample_alpha = 1

        self.temporal_transformer = CrossAttentionBlockGenral(out_feature_dim, out_feature_dim // 64)
        self.positional_embedding = nn.Parameter(torch.empty(1, self.num_frames // self.sparse_sample_alpha, out_feature_dim))
        from timm.models.layers import trunc_normal_
        trunc_normal_(self.positional_embedding, std=0.02)
        self.output_map_cls_token = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(out_feature_dim, int(out_feature_dim * mlp_ratio))),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(int(out_feature_dim * mlp_ratio), out_feature_dim))
        ]))
        self.ln_out_temp_cls_token = LayerNorm(out_feature_dim)

        self.spatial_transformer = CrossAttentionBlockGenral(out_feature_dim, out_feature_dim // 64)
        self.output_map_spatial_cls_token = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(out_feature_dim, int(out_feature_dim * mlp_ratio))),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(int(out_feature_dim * mlp_ratio), out_feature_dim))
        ]))
        self.ln_out_spat_cls_token = LayerNorm(out_feature_dim)


    def forward(self, prev_feat, spatial_cls_token, num_entities=None):
        b = prev_feat.size(0)
        top_cls_token = prev_feat[:, 0, :].unsqueeze(0) # (1, b, d)
        prev_feat = prev_feat[:, 1:, :]
        num_entities = prev_feat.size(1) // self.num_frames
        prev_feat = rearrange(prev_feat, 'b (n t) d -> n (b t) d', t=self.num_frames)
        t = self.num_frames

        if hasattr(self, 'spatial_transformer'):
            aggregated_spatial_cls_token = self.spatial_transformer(spatial_cls_token, prev_feat, prev_feat)
            spatial_cls_token = spatial_cls_token + aggregated_spatial_cls_token
            spatial_cls_token = spatial_cls_token + self.output_map_spatial_cls_token(self.ln_out_spat_cls_token(spatial_cls_token))

            cls_token = spatial_cls_token[0, ...].reshape(b, t, spatial_cls_token.size(-1))
            spatial_cls_token = spatial_cls_token[0, ...].reshape(b, t, spatial_cls_token.size(-1))
            # aggregated_spatial_cls_token = self.output_map_spatial_cls_token(self.ln_out_spat_cls_token(aggregated_spatial_cls_token))
        else:
            cls_token = spatial_cls_token[0, ...].reshape(b, t, spatial_cls_token.size(-1))
            aggregated_spatial_cls_token = 0.0

        if hasattr(self, 'temporal_transformer'):
            if hasattr(self, 'positional_embedding'):
                cls_token = (cls_token + self.positional_embedding.to(prev_feat.dtype)).permute(1, 0, 2)
            else:
                cls_token = cls_token.permute(1, 0, 2)
            aggregated_cls_token = self.temporal_transformer(top_cls_token, cls_token, cls_token)
            top_cls_token = top_cls_token + aggregated_cls_token
            top_cls_token = top_cls_token + self.output_map_cls_token(self.ln_out_temp_cls_token(top_cls_token))

        top_cls_token = top_cls_token.squeeze(0) # (1, )
        return top_cls_token, spatial_cls_token


class TemporalSpatialAdaPoolingNetwork(nn.Module):
    def __init__(self, cfg, out_feature_dim):
        super().__init__()
        mlp_ratio = 4

        self.num_frames = cfg.DATA.NUM_FRAMES
        self.sparse_sample_alpha = 1

        self.temporal_transformer = CrossAttentionBlockGenral(out_feature_dim, out_feature_dim // 64)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_frames // self.sparse_sample_alpha, 1, out_feature_dim))
        from timm.models.layers import trunc_normal_
        trunc_normal_(self.positional_embedding, std=0.02)
        self.output_map_cls_token = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(out_feature_dim, int(out_feature_dim * mlp_ratio))),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(int(out_feature_dim * mlp_ratio), out_feature_dim))
        ]))
        self.ln_out_temp_cls_token = LayerNorm(out_feature_dim)

        self.spatial_transformer = CrossAttentionBlockGenral(out_feature_dim, out_feature_dim // 64)
        self.output_map_temporal_cls_token = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(out_feature_dim, int(out_feature_dim * mlp_ratio))),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(int(out_feature_dim * mlp_ratio), out_feature_dim))
        ]))
        self.ln_out_spat_cls_token = LayerNorm(out_feature_dim)


    def forward(self, prev_feat, temporal_cls_token):
        #prev_feat is of shape (B, (1 + nt), d)
        #temporal_cls_token is of shape (B, 1, d)
        b = prev_feat.size(0)
        top_cls_token = prev_feat[:, 0, :].unsqueeze(0) # (1, b, d)
        prev_feat = prev_feat[:, 1:, :]
        num_entities = prev_feat.size(1) // self.num_frames
        prev_feat = rearrange(prev_feat, 'b (n t) d -> t (b n) d', t=self.num_frames)

        if hasattr(self, 'temporal_transformer'):
            if hasattr(self, 'positional_embedding'):
                prev_feat = prev_feat + self.positional_embedding.to(prev_feat.dtype)
            aggregated_temporal_cls_token = self.temporal_transformer(temporal_cls_token, prev_feat, prev_feat)
            temporal_cls_token = temporal_cls_token + aggregated_temporal_cls_token
            temporal_cls_token = temporal_cls_token + self.output_map_temporal_cls_token(self.ln_out_spat_cls_token(temporal_cls_token))
            cls_token = temporal_cls_token[0, ...].reshape(b, num_entities, temporal_cls_token.size(-1))
            temporal_cls_token = temporal_cls_token[0, ...].reshape(b, num_entities, temporal_cls_token.size(-1))

            # aggregated_spatial_cls_token = self.output_map_spatial_cls_token(self.ln_out_spat_cls_token(aggregated_spatial_cls_token))
        else:
            cls_token = temporal_cls_token[0, ...].reshape(b, num_entities, temporal_cls_token.size(-1))
            aggregated_temporal_cls_token = 0.0

        if hasattr(self, 'spatial_transformer'):
            cls_token = cls_token.permute(1, 0, 2) # (b, num_entities, d) -> (num_entities, b, d)
            aggregated_cls_token = self.spatial_transformer(top_cls_token, cls_token, cls_token)
            top_cls_token = top_cls_token + aggregated_cls_token
            top_cls_token = top_cls_token + self.output_map_cls_token(self.ln_out_temp_cls_token(top_cls_token))

        top_cls_token = top_cls_token.squeeze(0) # (1, )
        return top_cls_token, temporal_cls_token
