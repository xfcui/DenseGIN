import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from jax.ops import segment_sum

# dataset.py feature dimensions
# 10 atom features, 6/2/3/4 hop bond features, 17 node continuous features
from dataset import (
    NODE_FEAT_VOCAB_SIZES,
    NODE_FEAT_TOTAL_VOCAB,
    EDGE_FEAT_VOCAB_SIZES,
    EDGE_FEAT_TOTAL_VOCAB,
)

DROPOUT         = 0.1
EPSILON         = 1e-6
MINMAX_RATIO    = 20**.5
WIDTH_ACT_SCALE = 4

EMBED_POS  = 12    # RWPE12 only (ignore coord/en/geom auxiliaries)
EDGE_SUFFIXES = list(EDGE_FEAT_VOCAB_SIZES.keys())
EDGE_DIMS_PER_HOP = [
    (EDGE_FEAT_TOTAL_VOCAB[suffix], len(EDGE_FEAT_VOCAB_SIZES[suffix]))
    for suffix in EDGE_SUFFIXES
]


def _split_or_none(key, num):
    """Return a list of subkeys, or ``None`` if no key is provided."""
    if num == 0:
        return []
    if key is None:
        return [None] * num
    return list(jax.random.split(key, num))

def _clip_with_grad(x, min, max):
    """Clamp with gradients that flow through unclipped regions."""
    return jax.lax.stop_gradient(x.clip(min, max) - x) + x

def _count_params(model: eqx.Module) -> int:
    """Count the number of parameters in an Equinox module."""
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


class EmbedLayer(eqx.Module):
    """Memory-efficient multi-feature embedding unit."""
    embeddings: jnp.ndarray

    def __init__(self, total_vocab, num_features, width, key):
        # Concatenate all embeddings into one large array to reduce overhead
        total_dim = int(total_vocab)
        self.embeddings = jax.random.normal(key, (total_dim, width)) / np.sqrt(num_features)

    def __call__(self, x):
        """Sum embedding lookups over the feature dimension."""
        return jnp.sum(self.embeddings[x], axis=-2)

# ReZero: https://arxiv.org/abs/2003.04887
# LayerScale: https://arxiv.org/abs/2103.17239
class ScaleLayer(eqx.Module):
    """Learnable per-channel log-scale; forward multiplies by ``exp(scale)``."""
    scale: jnp.ndarray

    def __init__(self, width, scale_init=1.0):
        self.scale = jnp.full((width,), np.log(scale_init), dtype=jnp.float32)

    def __call__(self, x):
        """Scale each channel by a positive trainable coefficient."""
        scale = _clip_with_grad(jnp.exp(self.scale), 1e-2, 1)
        return scale * x

# PNA: https://arxiv.org/abs/2004.05718
# Graphormer: https://arxiv.org/abs/2106.05234
class DegreeLayer(eqx.Module):
    """Degree-weighted scaling: ``deg^power * x``."""
    num_head: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    power: jnp.ndarray

    def __init__(self, num_head, dim_head, power_init=-.5):
        self.num_head = num_head
        self.dim_head = dim_head
        self.power = jnp.full((num_head,), power_init, dtype=jnp.float32)

    def __call__(self, x, deg):
        """Apply per-head degree exponents."""
        power = jnp.repeat(_clip_with_grad(self.power, None, 0), self.dim_head)
        return jnp.pow(deg[:, None], power) * x

class ActLayer(eqx.Module):
    bias: jnp.ndarray

    def __init__(self, width, *, key=None):
        self.bias = jnp.zeros((width,), dtype=jnp.float32)

    def __call__(self, x, key=None):
        """Softplus activation with optional dropout at training time."""
        xx = jax.nn.softplus(x + self.bias)
        xx = _clip_with_grad(xx, 1/MINMAX_RATIO, MINMAX_RATIO)
        if key is None or DROPOUT <= 0.0:
            return xx

        keep = 1.0 - DROPOUT
        mask = jax.random.bernoulli(key, p=keep, shape=xx.shape)
        return xx * mask.astype(xx.dtype) / keep

class LinearLayer(eqx.Module):
    kernel: jnp.ndarray

    def __init__(self, width_in, width_out, key):
        self.kernel = jax.random.normal(key, (width_in, width_out)) / np.sqrt(width_in)

    def __call__(self, x):
        return x @ self.kernel


class GroupLinearBlock(eqx.Module):
    """Groups-parallel linear: weight ``(num_head, d_in, d_out)``, no bias."""
    num_head: int
    dim_head: int
    linear: LinearLayer
    kernel: jnp.ndarray

    def __init__(self, width_in, width_out, num_head, dim_head, key):
        width_norm = num_head * dim_head
        keys = _split_or_none(key, 2)

        self.num_head  = num_head
        self.dim_head = dim_head
        self.linear = LinearLayer(width_in, width_norm, keys[0])
        assert width_out % num_head == 0
        self.kernel = jax.random.normal(keys[1], (num_head, dim_head, width_out // num_head)) / np.sqrt(dim_head)

    def __call__(self, x, norm_bias=None):
        """Project per-head with optional grouped normalization bias."""
        xx = self.linear(x)
        xx = xx.reshape(*x.shape[:-1], self.num_head, -1)
        xx = xx / jnp.sqrt(jnp.mean(jnp.square(xx), axis=-1, keepdims=True) + EPSILON)
        if norm_bias is not None:
            xx = (xx + norm_bias.reshape(xx.shape)) / 2**.5
        xx = jnp.einsum("...hd,hdf->...hf", xx, self.kernel)
        xx = xx.reshape(*x.shape[:-1], -1)
        return xx

# GLU: https://arxiv.org/abs/1612.08083
# GLU-variants: https://arxiv.org/abs/2002.05202
class GatedLinearBlock(eqx.Module):
    """Gated linear unit with group-linear and optional post-projection."""
    num_head: int
    dim_head: int
    gate:   GroupLinearBlock
    value:  GroupLinearBlock
    act:    ActLayer
    kernel: jnp.ndarray
    linear: LinearLayer | None

    def __init__(self, width_in, width_out, num_head, dim_head, keep_groups=False, key=None):
        width_norm = num_head * dim_head
        width_act  = num_head * dim_head * WIDTH_ACT_SCALE
        keys = _split_or_none(key, 4)

        self.num_head = num_head
        self.dim_head = dim_head
        self.gate  = GroupLinearBlock(width_in, width_act, num_head, dim_head, keys[0])
        self.value = GroupLinearBlock(width_in, width_act, num_head, dim_head, keys[1])
        self.act = ActLayer(width_act)
        if keep_groups:
            assert width_out % num_head == 0
            self.kernel = jax.random.normal(keys[2], (num_head, dim_head * WIDTH_ACT_SCALE, width_out // num_head)) / np.sqrt(dim_head * WIDTH_ACT_SCALE)
            self.linear = None
        else:
            self.kernel = jax.random.normal(keys[2], (num_head, dim_head * WIDTH_ACT_SCALE, dim_head)) / np.sqrt(dim_head * WIDTH_ACT_SCALE)
            self.linear = LinearLayer(width_norm, width_out, keys[3])

    def __call__(self, x, y=None, gate_bias=None, value_bias=None, key=None):
        """Run gate/value paths and combine them by elementwise product."""
        keys = _split_or_none(key, 1)

        gg = self.gate(x, gate_bias)
        if y is None:
            vv = self.value(x, value_bias)
        else:
            vv = self.value(y, value_bias)
        xx = self.act(gg, keys[0]) * vv
        xx = xx.reshape(*x.shape[:-1], self.num_head, -1)
        xx = jnp.einsum("...hd,hdf->...hf", xx, self.kernel)
        xx = xx.reshape(*x.shape[:-1], -1)
        if self.linear is not None:
            xx = self.linear(xx)
        return xx

# MetaFormer: https://arxiv.org/abs/2210.13452
class MetaFormerBlock(eqx.Module):
    sca_pre:  ScaleLayer
    glu_pre:  GatedLinearBlock
    glu_post: GatedLinearBlock

    def __init__(self, width, num_head, dim_head, key=None):
        keys = _split_or_none(key, 2)

        self.sca_pre  = ScaleLayer(width, scale_init=1.0)
        self.glu_pre  = GatedLinearBlock(width, width, num_head, dim_head, key=keys[0])
        self.glu_post = GatedLinearBlock(width, width, num_head, dim_head, key=keys[1])

        print(f"##params[meta]:", _count_params(self))

    def __call__(self, x, msg, key=None):
        """Normalize then run two-step MetaFormer update."""
        keys = _split_or_none(key, 2)
        xx = self.sca_pre(x) + self.glu_pre(x, value_bias=msg, key=keys[0])
        xx = self.glu_post(xx, key=keys[1])
        return xx


# VoVNet: https://arxiv.org/abs/1904.09730
# GNN-AK: https://openreview.net/forum?id=Mspk_WYKoEH
class ConvKernel(eqx.Module):
    """Bond-aware graph convolution with degree normalisation."""
    scale:      jnp.ndarray
    embed_edge: EmbedLayer
    embed_deg:  EmbedLayer
    lin_pre:    LinearLayer
    glu_post:   GatedLinearBlock

    def __init__(self, width, num_head, dim_head, edge_total_vocab, edge_num_features, key=None):
        width_norm = num_head * dim_head
        keys = _split_or_none(key, 4)

        self.scale      = jnp.zeros((1, width_norm), dtype=jnp.float32) + np.log(np.expm1(2e-2))
        self.embed_edge = EmbedLayer(edge_total_vocab, edge_num_features, width, keys[0])
        self.embed_deg  = EmbedLayer(6, 1, width_norm, keys[1])
        self.lin_pre    = LinearLayer(width, width_norm, keys[2])
        self.glu_post   = GatedLinearBlock(width_norm, width_norm, num_head, dim_head, keep_groups=True, key=keys[3])

        print(f"##params[conv]:", _count_params(self), edge_total_vocab, edge_num_features)

    def __call__(self, x, deg, edge_idx, edge_attr, key=None):
        """
        edge_idx:  ``(2, E_pad)`` global node indices
        edge_attr: ``(E_pad, num_bond_features)`` int32 edge features
        """
        msg = self.lin_pre(x)[edge_idx[0]] \
            + jnp.sum(x[edge_idx[0]] * self.embed_edge(edge_attr), axis=-1, keepdims=True) \
            @ _clip_with_grad(jax.nn.softplus(self.scale), 1e-2, 1)
        msg = segment_sum(msg, edge_idx[1], len(x))
        msg = self.glu_post(msg, gate_bias=self.embed_deg(deg[:, None]), key=key)
        return msg

# GIN-virtual: https://arxiv.org/abs/2103.09430
class VirtKernel(eqx.Module):
    """Virtual node aggregation for global information exchange."""
    num_head: int
    lin_pre:  LinearLayer
    glu_post: GatedLinearBlock

    def __init__(self, width, num_head, dim_head, key=None):
        width_norm = num_head * dim_head
        keys = _split_or_none(key, 2)

        self.num_head = num_head
        self.lin_pre = LinearLayer(width, width_norm, keys[0])
        self.glu_post = GatedLinearBlock(width_norm, width_norm, num_head, dim_head, keep_groups=True, key=keys[1])

        print("##params[virt]:", _count_params(self))

    def __call__(self, x, virt, batch, batch_size, key=None):
        """Aggregate graph-level messages and optionally merge with virtual node."""
        msg = self.lin_pre(x)
        msg = segment_sum(msg, batch, batch_size)
        msg = msg / jnp.sqrt(jnp.mean(jnp.square(msg), axis=-1, keepdims=True) + EPSILON)
        if virt is None:
            virt = msg
        else:
            msg = virt = msg + virt
        msg = self.glu_post(msg, key=key)[batch]
        return msg, virt

class MixerKernel(eqx.Module):
    """Mixes 1-hop, 2-hop, 3-hop convolutions and virtual node information."""
    num_head: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    conv: tuple
    virt: VirtKernel
    scale: jnp.ndarray

    def __init__(self, width, num_head, dim_head, edge_dims_per_hop, key=None):
        keys = _split_or_none(key, len(edge_dims_per_hop) + 2)
        self.num_head = num_head
        self.dim_head = dim_head
        self.conv = tuple(
            ConvKernel(
                width,
                num_head,
                dim_head,
                edge_dims_per_hop[i][0],
                edge_dims_per_hop[i][1],
                key=keys[i],
            )
            for i in range(len(edge_dims_per_hop))
        )
        self.virt = VirtKernel(width, num_head, dim_head, key=keys[-2])
        # scale shape: (num_messages, num_head)
        self.scale = jnp.zeros((len(edge_dims_per_hop) + 1, num_head), dtype=jnp.float32)

    def __call__(self, x, virt, edges, batch, batch_size, key=None):
        """Run multi-hop convolutions, then mix with virtual-node message."""
        keys = _split_or_none(key, len(self.conv) + 1)

        scale = jax.nn.softplus(self.scale)
        scale = _clip_with_grad(scale, 1/MINMAX_RATIO, MINMAX_RATIO)
        scale = scale / jnp.sqrt(jnp.sum(jnp.square(scale), axis=0, keepdims=True))
        scale = jnp.repeat(scale, self.dim_head, axis=-1)
        
        msg = [conv(x, deg, idx, attr, key=keys[i]) for i, (conv, (idx, attr, deg)) in enumerate(zip(self.conv, edges))]
        ext, virt = self.virt(x, virt, batch, batch_size, key=keys[-1])
        msg = sum(s * m for s, m in zip(scale, msg + [ext]))
        return msg, virt

class HeadKernel(eqx.Module):
    """Readout head: virtual + node pooling → scalar prediction."""
    num_head: int
    lin_pre:  LinearLayer
    glu_post: GatedLinearBlock

    def __init__(self, width, num_head, dim_head, key=None):
        width_norm = num_head * dim_head
        keys = _split_or_none(key, 2)

        self.num_head = num_head
        self.lin_pre  = LinearLayer(width, width_norm, keys[0])
        self.glu_post = GatedLinearBlock(width_norm, 1, num_head*2, dim_head*2, key=keys[1])

        print("##params[head]:", _count_params(self))

    def __call__(self, x, virt, batch, batch_size, key=None):
        """Readout head: sum pool, normalize, fuse virtual node, and project."""
        msg = self.lin_pre(x)
        msg = segment_sum(msg, batch, batch_size)
        msg = msg / jnp.sqrt(jnp.mean(jnp.square(msg), axis=-1, keepdims=True) + EPSILON) + virt
        msg = self.glu_post(msg, key=key)
        return msg * 1.162127 + 5.689452


# GIN: https://openreview.net/forum?id=ryGs6iA5Km
# DenseNet: https://arxiv.org/abs/1608.06993
# AttnRes: https://arxiv.org/abs/2603.15031
class DenseGIN(eqx.Module):
    """DenseGIN for PCQM4Mv2: 19-feature atoms, 6-feature bonds, 8-step RWPE."""
    depth: int
    width: int
    num_head: int
    dim_head: int

    # Multi-feature atom encoder: one embedding table per atom feature dimension
    atom_embed: EmbedLayer
    atom_pos:   GatedLinearBlock

    mixs: tuple  # depth MixerKernels
    meta: tuple  # depth MetaFormerBlocks wrapping middle mixers

    head: HeadKernel

    def __init__(self, depth, width, num_head, dim_head, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        self.depth = depth
        self.width = width
        self.num_head = num_head
        self.dim_head = dim_head
        print(
            f"#model={self.__class__.__name__}, "
            f"depth={self.depth}, width={self.width}, "
            f"num_head={self.num_head}, dim_head={self.dim_head}"
        )
        
        # keys: atom_embed (1), atom_pos (1), mixers (depth + 2), meta (depth), glu (2), head (1)
        total_keys = 1 + 1 + (depth + 2) + depth + 2 + 1
        keys = _split_or_none(key, total_keys)
        
        curr = 0
        self.atom_embed = EmbedLayer(NODE_FEAT_TOTAL_VOCAB, len(NODE_FEAT_VOCAB_SIZES), width, keys[curr]); curr += 1
        self.atom_pos = GatedLinearBlock(EMBED_POS, width, num_head // 2, dim_head, 1, key=keys[curr]); curr += 1

        mixs = []
        meta = []
        for i in range(depth):
            mixs.append(MixerKernel(width, num_head, dim_head, EDGE_DIMS_PER_HOP, key=keys[curr]))
            curr += 1
            meta.append(MetaFormerBlock(width, num_head, dim_head, key=keys[curr]))
            curr += 1
        self.mixs = tuple(mixs)
        self.meta = tuple(meta)

        self.head = HeadKernel(width, num_head, dim_head, key=keys[curr])

        print("#params:", _count_params(self))
        print()

    def __call__(self, batch, training=False, key=None):
        """
        batch: flat dict from dataloader with keys:
            node_feat (N_pad, 10), node_embd (N_pad, 17),
            edgeX_feat (E_pad, 6/2/3/4), edgeX_index (2, E_pad),
            edgeX_batch (E_pad,), node_batch (N_pad,),
            batch_n_graphs (scalar)
            where X in {"", "_2hop", "_3hop", "_4hop"} and graph index 0 is null.
        """
        node_feat  = batch['node_feat']      # (N_pad, 10) int32
        node_embd  = batch['node_embd'][..., :EMBED_POS]      # (N_pad, 12)
        graph_id   = batch['node_batch']     # (N_pad,)
        batch_size = batch['batch_n_graphs'] + 1
        edges = self._get_edge(batch)
        keys = _split_or_none(key if training else None, (self.depth + 2) * 2)
        print("nodes={} | {}".format(
            node_feat.shape[0],
            ", ".join(
                "{}_edges={}".format(
                    "1hop" if suffix == "" else suffix[1:],
                    batch[f"edge{suffix}_index"].shape[1],
                )
                for suffix in EDGE_SUFFIXES
            ),
        ))

        x, virt = self.atom_embed(node_feat) + jax.vmap(self.atom_pos)(node_embd), None
        for i in range(self.depth):
            msg, virt = self.mixs[i](x, virt, edges, graph_id, batch_size, key=keys[2*i])
            x = self.meta[i](x, msg, key=keys[2*i+1])
        y = self.head(x, virt, graph_id, batch_size)[1:]
        return y

    def _get_edge(self, batch):
        """Build edge tuples ``(edge_index, edge_attr, degree)`` per hop."""
        num_nodes = batch['node_feat'].shape[0]
        edges = []
        for suffix in EDGE_SUFFIXES:
            edge_index = batch[f'edge{suffix}_index']   # (2, E_pad)
            edge_attr  = batch[f'edge{suffix}_feat']    # (E_pad, num_bond_features)
            deg = segment_sum(jnp.ones((batch[f'edge{suffix}_batch'].shape[0], 1), dtype=edge_index.dtype), edge_index[1], num_nodes).squeeze(-1).clip(1, None)
            edges.append((edge_index, edge_attr, deg))
        return edges


def get_model(key):
    """Create the default DenseGIN model (depth=5, width=256, heads=16)."""
    return DenseGIN(depth=5, width=256, num_head=16, dim_head=16, key=key)

