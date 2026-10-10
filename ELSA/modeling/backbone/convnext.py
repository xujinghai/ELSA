# ConvNeXt backbone (Liu et al., 2022), adapted for detectron2.
# Tiny config: depths [3,3,9,3], dims [96,192,384,768].
# Module naming follows the official facebookresearch/ConvNeXt so that
# pretrained weights load correctly after key remapping (adding "backbone." prefix).
import torch
import torch.nn as nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import Backbone, BACKBONE_REGISTRY


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (per convNeXt paper)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Stochastic Depth per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    """LayerNorm supporting channels-first format (B, C, H, W).

    Normalizes across the channel dimension, matching the official ConvNeXt
    implementation so weight/bias shapes are (dim,) and load correctly.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: B x C x H x W
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Block(nn.Module):
    """ConvNeXt Block: DWConv -> LN -> 1x1 expand(4x) -> GELU -> 1x1 project -> LS + Stochastic Depth."""

    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # inverted bottleneck expand
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)  # project back
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim)
        ) if layer_scale_init > 0 else None
        self.drop = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC for Linear
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = identity + self.drop(x)
        return x


class ConvNeXt(Backbone):
    """ConvNeXt backbone with multi-scale feature outputs (res2-res5)."""

    def __init__(self, cfg, input_shape):
        super().__init__()
        depths = cfg.MODEL.CONVNEXT.DEPTHS
        dims = cfg.MODEL.CONVNEXT.DIMS
        drop_path_rate = cfg.MODEL.CONVNEXT.DROP_PATH_RATE
        self._out_features = cfg.MODEL.CONVNEXT.OUT_FEATURES
        in_chans = input_shape.channels if isinstance(input_shape, ShapeSpec) else 3

        # Stochastic depth rates: linearly scale from 0 -> drop_path_rate across all blocks
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # downsample_layers[0] = stem (4x4 stride 4), [1..3] = stage transitions (LN + 2x2 stride 2)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6),
        )
        self.downsample_layers.append(stem)
        for i in range(len(depths) - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        # stages: each stage is a sequence of ConvNeXt Blocks
        self.stages = nn.ModuleList()
        block_idx = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[
                    Block(dims[i], drop_path=dpr[block_idx + j], layer_scale_init=1e-6)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            block_idx += depths[i]

        # final norm (kept for weight-loading compatibility; not used in feature outputs)
        self.norm = LayerNorm(dims[-1], eps=1e-6)

        # Feature output mapping: res2=stage0(stride4), res3=stage1(stride8), ...
        self._out_feature_strides = {
            "res2": 4, "res3": 8, "res4": 16, "res5": 32,
        }
        self._out_feature_channels = {
            "res2": dims[0], "res3": dims[1], "res4": dims[2], "res5": dims[3],
        }

    def forward(self, x):
        outs = {}
        x = self.downsample_layers[0](x)  # stem -> stride 4
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if f"res{i + 2}" in self._out_features:
                outs[f"res{i + 2}"] = x
            if i < len(self.stages) - 1:
                x = self.downsample_layers[i + 1](x)
        return outs

    @property
    def size_divisibility(self) -> int:
        return 32


@BACKBONE_REGISTRY.register()
def build_convnext_backbone(cfg, input_shape):
    """Build a ConvNeXt backbone registered with detectron2."""
    return ConvNeXt(cfg, input_shape)
