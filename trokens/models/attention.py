"""Attention modules"""
import torch
import torch.nn as nn
from einops import rearrange
from trokens.models.common import DropPath, Mlp

class Attention(nn.Module):
    """Attention module"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                attn_drop=0., proj_drop=0., with_qkv=True, use_mask=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.use_mask = use_mask

    def forward(self, x, mask=None):
        """Forward pass"""
        batch_size, seq_len, feat_dim = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads,
                                    feat_dim // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(batch_size, seq_len, self.num_heads,
                            feat_dim // self.num_heads).permute(0, 2, 1, 3)
            q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.use_mask:
            if len(mask.shape) == 2:
                # extra_dim for heads and dim on which attention is not applied
                attn = attn.masked_fill(torch.logical_not(mask[:,None,None,:]), float('-inf'))
            else:
                # extra_dim for heads
                # mask -> bs x num_spacetime_tokens x num_spacetime_tokens
                attn = attn.masked_fill(torch.logical_not(mask[:,None,:,:]), float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, feat_dim)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """Block module"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                            qk_scale=qk_scale, attn_drop=attn_drop,
                                            proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                        act_layer=act_layer, drop=drop)


    def forward(self, x, batch_size, num_frames, width):
        """Forward pass"""
        num_spatial_tokens = (x.size(1) - 1) // num_frames
        height = num_spatial_tokens // width

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',
                            b=batch_size, h=height, w=width, t=num_frames)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',
                                    b=batch_size, h=height, w=width, t=num_frames)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, num_frames, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) 1 m',b=batch_size,t=num_frames)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',
                            b=batch_size, h=height, w=width, t=num_frames)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m', b=batch_size, t=num_frames)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',
                                    b=batch_size, h=height, w=width, t=num_frames)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class CrossAttention(nn.Module):
    """
    Cross attention module where queries come from one source and keys/values from another.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in q, k, v projections
        qk_scale: Scale factor for attention (if None, uses head_dim ** -0.5)
        attn_drop: Dropout rate for attention matrix
        proj_drop: Dropout rate for projection output
        use_mask: Whether to use attention masking
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                attn_drop=0., proj_drop=0., use_mask=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Separate projections for query and key-value
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_mask = use_mask

    def forward(self, kv, query, mask=None):
        """
        Args:
            query: Query tensor of shape (B, nq, C)
            kv: Key-value tensor of shape (B, nkv, C)
            mask: Optional attention mask
        """
        batch_size, nq, feat_dim = query.shape
        _, nkv, _ = kv.shape

        # Project query and key-value
        q = self.q_proj(query).reshape(batch_size, nq, self.num_heads,
                                        feat_dim // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(kv).reshape(batch_size, nkv, 2, self.num_heads,
                                        feat_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.use_mask and mask is not None:
            if len(mask.shape) == 2:
                attn = attn.masked_fill(torch.logical_not(mask[:,None,None,:]), float('-inf'))
            else:
                attn = attn.masked_fill(torch.logical_not(mask[:,None,:,:]), float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, nq, feat_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class TrajectoryAttentionBlock(nn.Module):
    """Trajectory attention block"""
    def __init__(
            self, cfg=None, dim=768, num_heads=12,
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            pt_attention='divided_space_time', use_pt_visibility=False,
            num_mlp_layers=1,
        ):
        super().__init__()
        self.cfg = cfg
        self.pt_attention = pt_attention
        self.norm1 = norm_layer(dim)
        if pt_attention in 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            temp_mask_flag = True if pt_attention == 'temp_patch' else False
            self.temporal_attn = Attention(dim, num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop,
                                    proj_drop=drop, use_mask=(use_pt_visibility or temp_mask_flag))
            self.temporal_fc = nn.Linear(dim, dim)
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  attn_drop=attn_drop, proj_drop=drop,
                                  use_mask=use_pt_visibility)
        else:
            raise NotImplementedError(f"Unsupported attention type {pt_attention}")
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = []
        for _ in range(num_mlp_layers):
            mlp_hidden_dim = int(dim * mlp_ratio)
            mlp_ratio = max(mlp_ratio // 2, 1)
            self.mlp.append(Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                                act_layer=act_layer, drop=drop))
        self.mlp = nn.Sequential(*self.mlp)

        self.use_pt_visibility = use_pt_visibility

    def forward(self, x, thw, pt_mask, with_cls_token=True):
        """Forward pass"""
        if self.pt_attention == 'divided_space_time':
            if self.cfg.MODEL.USE_CLS_TOKEN:
                xt = x[:,1:,:]
                temporal_mask = pt_mask[:,1:].clone()
            else:
                xt = x
                temporal_mask = pt_mask.clone()
            num_frames, height, width = thw
            batch_size = x.shape[0]
            spatial_mask = None

            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',
                            b=batch_size, h=height, w=width, t=num_frames)
            if self.use_pt_visibility:
                temporal_mask = rearrange(temporal_mask, 'b (h w t) -> (b h w) t',
                                            b=batch_size, h=height, w=width, t=num_frames)
            res_temporal = self.drop_path(
                            self.temporal_attn(self.temporal_norm1(xt), temporal_mask))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',
                                    b=batch_size, h=height, w=width, t=num_frames)
            res_temporal = self.temporal_fc(res_temporal)
                # res_temporal = res_temporal * temporal_mask.unsqueeze(-1)


            if self.cfg.MODEL.USE_CLS_TOKEN:
                xt = x[:,1:,:] + res_temporal
                init_cls_token = x[:,0,:].unsqueeze(1)
                cls_token = init_cls_token.repeat(1, num_frames, 1)
                if self.use_pt_visibility:
                    intit_cls_pt_mask = pt_mask[:,0].unsqueeze(1)
                    spatial_mask = pt_mask[:,1:].clone()
                    intit_cls_pt_mask = intit_cls_pt_mask.repeat(1, num_frames)

                cls_token = rearrange(cls_token, 'b t m -> (b t) 1 m',
                                            b=batch_size, t=num_frames)
            else:
                xt = x + res_temporal
                if self.use_pt_visibility:
                    spatial_mask = pt_mask.clone()
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',
                                b=batch_size, h=height, w=width, t=num_frames)
            if self.cfg.MODEL.USE_CLS_TOKEN:
                xs = torch.cat((cls_token, xs), 1)
                if self.use_pt_visibility:
                    intit_cls_pt_mask = rearrange(intit_cls_pt_mask, 'b t -> (b t) 1',
                                                b=batch_size,t=num_frames)
                    spatial_mask = rearrange(spatial_mask, 'b (h w t) -> (b t) (h w)',
                                        b=batch_size,h=height,w=width,t=num_frames)
                    spatial_mask = torch.cat((intit_cls_pt_mask, spatial_mask), 1)
            else:
                if self.use_pt_visibility:
                    spatial_mask = rearrange(spatial_mask, 'b (h w t) -> (b t) (h w)',
                                            b=batch_size,h=height,w=width,t=num_frames)


            res_spatial = self.drop_path(self.attn(self.norm1(xs), spatial_mask))
            if self.cfg.MODEL.USE_CLS_TOKEN:
                ### Taking care of CLS token
                cls_token = res_spatial[:,0,:]
                cls_token = rearrange(cls_token, '(b t) m -> b t m',b=batch_size,t=num_frames)
                cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
                res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',
                                    b=batch_size,h=height,w=width,t=num_frames)
            res = res_spatial
            x = xt

            ## Mlp
            if self.cfg.MODEL.USE_CLS_TOKEN:
                x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            else:
                x = x + res

        elif self.pt_attention == 'trajectory':
            x  = x + self.drop_path(
                self.attn(
                    self.norm1(x),
                    thw,
                    with_cls_token=with_cls_token,
                )[0]
            )
        else:
            raise NotImplementedError(f"Unsupported attention type {self.pt_attention}")


        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, thw
