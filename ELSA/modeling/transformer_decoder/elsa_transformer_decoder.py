# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry

from .position_encoding import PositionEmbeddingSine

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in MaskFormer.
"""

def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class LocalRepresentation(nn.Module):
    """
    Local Representation module for generating feature vectors from input features.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors (default: 256).

    Attributes:
        to_query_3x3 (nn.Conv2d): 3x3 depth-wise convolutional layer for local feature extraction.
        bn (nn.BatchNorm2d): Batch normalization layer.
        out (nn.Linear): Linear transformation layer.
        d_model (int): The dimensionality of the input and output feature vectors.

    Methods:
        forward(self, x): Forward pass through the LocalRepresentation module.
    """
    def __init__(self, d_model=256):
        super().__init__()

        self.to_query_3x3 = nn.Conv2d(d_model, d_model, 3, groups=d_model, padding=1)
        #self.bn = nn.SyncBatchNorm(d_model)
        self.bn = nn.BatchNorm2d(d_model)
        self.out = nn.Linear(d_model, d_model)

        self.d_model = d_model

    def forward(self, x):
        # Retrieve input tensor shape
        B, C, H, W = x.shape

        # Apply pre-normalisation followed by 3x3 local convolution to extract local features
        x = self.bn(x)
        x_3x3 = self.to_query_3x3(x)

        # Reshape the local features and permute dimensions for linear transformation
        return self.out(x_3x3.view(B, self.d_model, H*W).permute(0, 2, 1))


class LinearCrossAttention(nn.Module):
    """
    Linear Cross-Attention (LCA) module with softplus kernel.

    Replaces PBCA's hard argmax prototype selection with softplus-kernel linear attention,
    achieving O(N*d^2) complexity soft cross-attention.

    Design (per lda-elsa-improvement.md V2):
      - V/K projection: LocalRepresentation (3x3 DWConv + BN + Linear), preserves local spatial bias
      - Q projection: LayerNorm + Linear
      - Kernel: softplus (non-negative, smooth)
      - Mask: global soft gating (Plan A) - average pred_mask across queries to weight K and V
      - Output: out_proj + learnable gamma (per-channel) + residual
      - A+ per-query bias: mask-pooled V added to linear attention output (post-norm),
        only for N <= 4096 (s=3/s=4); s=2 (N=16384) skipped to avoid O(NQd) cost.

    Args:
        d_model (int): feature dimension (default: 256)
        nhead (int): number of heads (default: 8)

    Forward signature:
        tgt:        Q x B x C    query features
        memory:     B x C x H x W  spatial features (from src[level_index])
        memory_mask: B x Q x H x W  per-query pred_mask logits (from forward_prediction_heads)
        pos:        B x C x H x W  positional encoding for memory
        query_pos:  Q x B x C    positional encoding for query
    Returns:
        Q x B x C
    """

    def __init__(self, d_model=256, nhead=8):
        super().__init__()

        # Independent V/K projections via LocalRepresentation (3x3 DWConv + BN + Linear)
        # Preserves local spatial aggregation and position inductive bias.
        # Linear attention is permutation-invariant, so DWConv is essential for spatial structure.
        self.v_proj = LocalRepresentation(d_model)
        self.k_proj = LocalRepresentation(d_model)

        # Q projection: LayerNorm + Linear
        self.q_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # LayerNorm before out_proj: bounds the per-query output magnitude.
        # The KV matrix accumulates over N spatial positions; without this bound a large
        # V magnitude would propagate to pred_masks logits and saturate sigmoid.
        self.out_norm = nn.LayerNorm(d_model)

        # Learnable per-channel residual scaling (init to identity)
        self.gamma = nn.Parameter(torch.ones(1, 1, d_model))

        # A+ per-query output bias (A_Plus_Design.md V2):
        # Adds mask-pooled V (per-query) to the linear attention output (post-normalization).
        # Only active for N <= 4096 (s=3/s=4); s=2 (N=16384) skipped to avoid O(NQd) cost.
        # eta=0.1 init: global attention dominates (90%), per-query correction is 10%.
        # Scale-matched: both linear_out and q_bias are O(1), so eta directly controls the ratio.
        self.eta = nn.Parameter(torch.tensor(0.1))

        self.num_heads = nhead
        self.head_dim = d_model // nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, memory_mask, pos, query_pos):
        # tgt: QxBxC, memory: BxCxHxW, memory_mask: BxQxHxW, pos: BxCxHxW, query_pos: QxBxC
        res = tgt

        # Disable AMP for the entire LCA forward: the KV matrix accumulates over
        # N=16384 spatial positions, and fp16 overflows (>65504) even with internal
        # .float() casts because the upstream projections (k_proj/v_proj) and the
        # output projection still run in fp16 under autocast. Running the whole
        # module in fp32 is the standard fix for numerically sensitive attention.
        with torch.cuda.amp.autocast(enabled=False):
            # Cast all inputs to fp32 (no-op if already fp32)
            tgt = tgt.float()
            memory = memory.float()
            if memory_mask is not None:
                memory_mask = memory_mask.float()
            if pos is not None:
                pos = pos.float()
            if query_pos is not None:
                query_pos = query_pos.float()

            # Inject positional encoding into memory and query (same as PBCA)
            # K and V use the same pos-augmented memory so their feature spaces are consistent.
            memory = self.with_pos_embed(memory, pos)
            tgt = self.with_pos_embed(tgt, query_pos)

            # Project: memory (BxCxHxW) -> BxNxC via LocalRepresentation
            #          tgt (QxBxC) -> BxQxC via q_proj
            K = self.k_proj(memory)                                   # B x N x C
            V = self.v_proj(memory)                                   # B x N x C
            Q = self.q_proj(tgt.permute(1, 0, 2))                     # B x Q x C

            # A+ V2: save reference to original V (before global gating).
            # q_bias uses ungated V so per-query features are not double-suppressed.
            # This is a reference (no copy); V * fg_weight below creates a new tensor.
            V_original = V  # B x N x C

            B, N, C = K.shape
            Q_len = Q.shape[1]

            # Global soft gating (Plan A): average per-query pred_mask -> global foreground map
            # Note: when memory_mask is near uniform early in training, the gate degenerates
            # to ~0.5 everywhere, which is acceptable (no foreground bias is enforced).
            if memory_mask is not None:
                # memory_mask: BxQxHxW (logits, can be negative)
                mask_prob = torch.sigmoid(memory_mask)                # BxQxHxW -> [0, 1]
                fg_weight = mask_prob.mean(dim=1, keepdim=True)       # Bx1xHxW (global foreground prob)
                fg_weight = fg_weight.flatten(2).permute(0, 2, 1)     # BxNx1
                K = K * fg_weight                                     # broadcast over channel
                V = V * fg_weight

            # Multi-head reshape: (B, *, C) -> (B, h, *, d)
            Q = Q.view(B, Q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B x h x Q x d
            K = K.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)      # B x h x N x d
            V = V.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)      # B x h x N x d

            # A+ V2: multi-head reshape for original V (used by q_bias)
            if memory_mask is not None and N <= 4096:
                V_orig = V_original.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B x h x N x d

            # Apply non-negative softplus kernel to Q and K (V is left untouched as the "value")
            Q_sp = F.softplus(Q)
            K_sp = F.softplus(K)

            # KV matrix: sum over spatial positions -> B x h x d x d
            # This is the linear-attention factorization: phi(K)^T @ V aggregated over N.
            KV = torch.einsum('bhnd,bhnv->bhdv', K_sp, V)

            # Numerator: phi(Q) @ KV -> B x h x Q x d
            numerator = torch.einsum('bhqd,bhdv->bhqv', Q_sp, KV)

            # Denominator: phi(Q) @ sum_n phi(K_n) -> B x h x Q x 1
            # Use + eps (not clamp) to keep the function differentiable everywhere.
            K_sum = K_sp.sum(dim=2)                                   # B x h x d
            denominator = torch.einsum('bhqd,bhd->bhq', Q_sp, K_sum).unsqueeze(-1)  # B x h x Q x 1
            linear_out = numerator / (denominator + 1e-4)           # B x h x Q x d  ← O(1) scale

            # A+ V2: per-query output bias (added post-normalization, scale-matched).
            # q_bias = mask-weighted average of ungated V for each query.
            # Only for N <= 4096 (s=3/s=4); s=2 (N=16384) skipped (avoid O(NQd) cost).
            # clamp(min=1.0): empty mask → q_bias=0/1.0=0 (safe, no contribution).
            if memory_mask is not None and N <= 4096:
                mask_flat = mask_prob.flatten(2)                                  # B x Q x N
                mask_h = mask_flat.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # B x h x Q x N
                q_bias = torch.einsum('bhqn,bhnd->bhqd', mask_h, V_orig)         # B x h x Q x d
                mask_sum = mask_h.sum(dim=3, keepdim=True).clamp(min=1.0)        # B x h x Q x 1
                q_bias = q_bias / mask_sum                                       # B x h x Q x d  ← O(1) scale
                out = linear_out + self.eta * q_bias                             # O(1) + O(1) ← eta=0.1 = 10%
            else:
                out = linear_out

            # Merge heads: B x h x Q x d -> B x Q x C
            out = out.permute(0, 2, 1, 3).reshape(B, Q_len, C)

            # Output projection + learnable per-channel gamma + residual
            out = self.out_norm(out)                              # bound magnitude per query
            out = self.out_proj(out) * self.gamma                 # B x Q x C

        # Cast output back to input dtype (fp16 if AMP is enabled)
        return out.permute(1, 0, 2).to(res.dtype) + res            # Q x B x C (matches input tgt shape)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                LinearCrossAttention(d_model=hidden_dim, nhead=nheads)
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src, pos, size_list = [], [], []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None))
            src.append((self.input_proj[i](x[i]) + self.level_embed.weight[i][None, :, None, None]))

        bs = src[0].shape[0]

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).expand(-1, bs, -1)
        output = self.query_feat.weight.unsqueeze(1).expand(-1, bs, -1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # attention: cross-attention first

            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                pos=pos[level_index],
                query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.permute(1, 0, 2)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
