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

DROPOUT    = 0.1

EMBED_POS  = 12    # RWPE12 only (ignore coord/en/geom auxiliaries)
EDGE_SUFFIXES = list(EDGE_FEAT_VOCAB_SIZES.keys())
EDGE_DIMS_PER_HOP = [
    (EDGE_FEAT_TOTAL_VOCAB[suffix], len(EDGE_FEAT_VOCAB_SIZES[suffix]))
    for suffix in EDGE_SUFFIXES
]
SCALE_GRAPH, SCALE_NODE, SCALE_EDGE = 4, 3, 2


def _split_or_none(key, num):
    if num == 0:
        return []
    if key is None:
        return [None] * num
    return list(jax.random.split(key, num))

def _count_params(model: eqx.Module) -> int:
    """Count the number of parameters in an Equinox module."""
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


def _dropout_act(x, key):
    xx = jax.nn.softplus(x)
    if key is None or DROPOUT <= 0.0: return xx

    keep = 1.0 - DROPOUT
    mask = jax.random.bernoulli(key, p=keep, shape=xx.shape)
    xx = xx * mask.astype(xx.dtype) / keep
    return xx


class EmbedLayer(eqx.Module):
    """Memory-efficient multi-feature embedding unit."""
    embeddings: jnp.ndarray

    def __init__(self, total_vocab, num_features, width, key):
        # Concatenate all embeddings into one large array to reduce overhead
        total_dim = int(total_vocab)
        scale = 1.0 / np.sqrt(num_features)
        self.embeddings = jax.random.normal(key, (total_dim, width)) * scale

    def __call__(self, x):
        # x: (N, num_features)
        return jnp.sum(self.embeddings[x], axis=-2)

# ReZero: https://arxiv.org/abs/2003.04887
# LayerScale: https://arxiv.org/abs/2103.17239
class ScaleLayer(eqx.Module):
    """Learnable per-channel log-scale; forward multiplies by exp(scale)."""
    scale: jnp.ndarray

    def __init__(self, width, scale_init=1.0):
        self.scale = jnp.full((width,), np.log(scale_init), dtype=jnp.float32)

    def __call__(self, x):
        return jnp.exp(self.scale.clip(None, 0)) * x

# PNA: https://arxiv.org/abs/2004.05718
# Graphormer: https://arxiv.org/abs/2106.05234
class DegreeLayer(eqx.Module):
    """Degree-weighted scaling: deg^degree * x."""
    num_head: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    power: jnp.ndarray

    def __init__(self, num_head, dim_head, power_init=-.5):
        self.num_head = num_head
        self.dim_head = dim_head
        self.power = jnp.full((num_head,), power_init, dtype=jnp.float32)

    def __call__(self, x, deg):
        # x: (N, num_head * dim_head)
        # deg: (N,)
        power = jnp.repeat(self.power.clip(None, 0), self.dim_head)
        return jnp.pow(deg[:, None], power) * x

class LinearLayer(eqx.Module):
    kernel: jnp.ndarray

    def __init__(self, width_in, width_out, key):
        scale = 1.0 / np.sqrt(width_in)
        self.kernel = jax.random.normal(key, (width_in, width_out)) * scale

    def __call__(self, x):
        return x @ self.kernel


class GroupLinearBlock(eqx.Module):
    """Groups-parallel linear: weight (num_head, d_in, d_out), no bias."""
    num_head: int
    dim_head: int
    linear: LinearLayer
    kernel: jnp.ndarray

    def __init__(self, width_in, width_out, num_head, dim_head, key):
        width_norm = num_head * dim_head
        keys = _split_or_none(key, 2)
        scale = 1.0 / np.sqrt(dim_head)

        self.num_head  = num_head
        self.dim_head = dim_head
        self.linear = LinearLayer(width_in, width_norm, keys[0])
        assert width_out % num_head == 0
        self.kernel = jax.random.normal(keys[1], (num_head, dim_head, width_out // num_head)) * scale

    def __call__(self, x, norm_bias=None, eps=1e-5):
        xx = self.linear(x)
        xx = xx.reshape(*x.shape[:-1], self.num_head, -1)
        xx = xx / jnp.sqrt(jnp.mean(jnp.square(xx), axis=-1, keepdims=True) + eps)
        if norm_bias is not None:
            xx = (xx + norm_bias.reshape(xx.shape)) / 2**.5
        xx = jnp.einsum("...hd,hdf->...hf", xx, self.kernel)
        xx = xx.reshape(*x.shape[:-1], -1)
        return xx

# GLU: https://arxiv.org/abs/1612.08083
# GLU-variants: https://arxiv.org/abs/2002.05202
class GatedLinearBlock(eqx.Module):
    """Gated linear unit with group-linear and linear post-projection."""
    num_head: int
    dim_head: int
    gate:  GroupLinearBlock
    value: GroupLinearBlock
    kernel: jnp.ndarray
    linear: LinearLayer | None

    def __init__(self, width_in, width_out, num_head, dim_head, scale_act, keep_groups=False, key=None):
        width_norm = num_head * dim_head
        width_act  = num_head * dim_head * scale_act
        keys = _split_or_none(key, 4)
        scale = 1.0 / np.sqrt(dim_head * scale_act)

        self.num_head = num_head
        self.dim_head = dim_head
        self.gate  = GroupLinearBlock(width_in, width_act, num_head, dim_head, keys[0])
        self.value = GroupLinearBlock(width_in, width_act, num_head, dim_head, keys[1])
        if keep_groups:
            assert width_out % num_head == 0
            self.kernel = jax.random.normal(keys[2], (num_head, dim_head * scale_act, width_out // num_head)) * scale
            self.linear = None
        else:
            self.kernel = jax.random.normal(keys[2], (num_head, dim_head * scale_act, dim_head)) * scale
            self.linear = LinearLayer(width_norm, width_out, keys[3])

    def __call__(self, x, y=None, gate_bias=None, value_bias=None, key=None):
        gg = self.gate(x, gate_bias)
        if y is None:
            vv = self.value(x, value_bias)
        else:
            vv = self.value(y, value_bias)
        
        # Split key for dropout if provided
        keys = _split_or_none(key, 1)

        xx = _dropout_act(gg, keys[0]) * vv
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

    def __init__(self, width, num_head, dim_head, scale_act, key=None):
        keys = _split_or_none(key, 2)

        self.sca_pre  = ScaleLayer(width, scale_init=1.0)
        self.glu_pre  = GatedLinearBlock(width, width, num_head, dim_head, scale_act, key=keys[0])
        self.glu_post = GatedLinearBlock(width, width, num_head, dim_head, scale_act, key=keys[1])

        print(f"##params[meta]:", _count_params(self))

    def __call__(self, x, msg, key=None):
        keys = _split_or_none(key, 2)
        xx = self.sca_pre(x) + self.glu_pre(x, value_bias=msg, key=keys[0])
        xx = self.glu_post(xx, key=keys[1])
        return xx


# VoVNet: https://arxiv.org/abs/1904.09730
# GNN-AK: https://openreview.net/forum?id=Mspk_WYKoEH
class ConvKernel(eqx.Module):
    """Bond-aware graph convolution with degree normalisation."""
    embed: EmbedLayer
    glu: GatedLinearBlock
    deg: DegreeLayer

    def __init__(self, width, num_head, dim_head, edge_total_vocab, edge_num_features, key=None):
        width_norm = num_head * dim_head
        keys = _split_or_none(key, 2)

        self.embed = EmbedLayer(edge_total_vocab, edge_num_features, width_norm, keys[0])
        self.glu = GatedLinearBlock(width, width_norm, num_head, dim_head, SCALE_EDGE, keep_groups=True, key=keys[1])
        self.deg = DegreeLayer(num_head, dim_head)

        print(f"##params[conv]:", _count_params(self), edge_total_vocab, edge_num_features)

    def __call__(self, x, deg, edge_idx, edge_attr, edge_mask, key=None):
        """
        edge_idx:  (2, E_pad) global node indices
        edge_attr: (E_pad, num_bond_features) int32
        edge_mask: (E_pad,) bool — False for padding entries
        """
        # Sum embeddings across all bond feature dimensions for the gate bias
        msg = self.embed(edge_attr)
        msg = self.glu(x[edge_idx[1]], x[edge_idx[0]], gate_bias=msg, key=key)
        msg = segment_sum(msg, edge_idx[1], len(x))
        msg = self.deg(msg, deg)
        return msg

# GIN-virtual: https://arxiv.org/abs/2103.09430
class VirtKernel(eqx.Module):
    """Virtual node aggregation for global information exchange."""
    num_head: int
    glu_pre:  GatedLinearBlock
    glu_post: GatedLinearBlock

    def __init__(self, width, num_head, dim_head, key=None):
        width_norm = num_head * dim_head
        keys = _split_or_none(key, 2)

        self.num_head = num_head
        self.glu_pre  = GatedLinearBlock(width, width_norm, num_head, dim_head, SCALE_EDGE, keep_groups=True, key=keys[0])
        self.glu_post = GatedLinearBlock(width_norm, width_norm, num_head, dim_head, SCALE_GRAPH, keep_groups=True, key=keys[1])

        print("##params[virt]:", _count_params(self))

    def __call__(self, x, virt, batch, batch_size, key=None):
        xx = self.glu_pre(x, key=key)
        msg = segment_sum(xx, batch, batch_size)
        if virt is None:
            virt = msg
        else:
            msg = virt = msg + virt
        msg = self.glu_post(msg, key=key)
        return msg[batch], virt

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
        # edges is a list of (edge_index, edge_attr, deg, edge_mask) for each hop
        keys = _split_or_none(key, len(self.conv) + 1)
        msgs = [conv(x, deg, idx, attr, mask, key=keys[i]) for i, (conv, (idx, attr, deg, mask)) in enumerate(zip(self.conv, edges))]
        xx, virt = self.virt(x, virt, batch, batch_size, key=keys[len(self.conv)])
        msgs.append(xx)
        
        # msgs: list of (N, num_head * dim_head)
        # scale: (num_messages, num_head)
        scale = jax.nn.softplus(self.scale).clip(1/10, 10)
        # scale_norm: (num_messages, num_head)
        scale_norm = scale / jnp.sqrt(jnp.sum(jnp.square(scale), axis=0, keepdims=True))
        # Apply per-head scaling: (num_messages, num_head) -> (num_messages, num_head * dim_head)
        scale_expanded = jnp.repeat(scale_norm, self.dim_head, axis=-1)
        
        return sum(s * m for s, m in zip(scale_expanded, msgs)), virt

class HeadKernel(eqx.Module):
    """Readout head: virtual + node pooling → scalar prediction."""
    num_head: int
    glu_pre:  GatedLinearBlock
    glu_post: GatedLinearBlock

    def __init__(self, width, num_head, dim_head, key=None):
        width_norm = num_head * dim_head
        keys = _split_or_none(key, 2)

        self.num_head = num_head
        self.glu_pre  = GatedLinearBlock(width, width_norm, num_head, dim_head, SCALE_EDGE, keep_groups=True, key=keys[0])
        self.glu_post = GatedLinearBlock(width_norm, 1, num_head*2, dim_head, SCALE_GRAPH, key=keys[1])

        print("##params[head]:", _count_params(self))

    def __call__(self, x, virt, batch, batch_size, key=None):
        xx = self.glu_pre(x, key=key)
        yy = segment_sum(xx, batch, batch_size) + virt
        yy = self.glu_post(yy, key=key) * 1.162127 + 5.689452
        return yy


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
    atom_pos: GatedLinearBlock

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
            meta.append(MetaFormerBlock(width, num_head, dim_head, SCALE_NODE, key=keys[curr]))
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

        # Initial node features from atom embeddings + RWPE
        h = self.atom_embed(node_feat)
        h = h + jax.vmap(self.atom_pos)(node_embd)

        # Kernels 1 to self.depth: Mixer wrapped by MetaFormer
        virt = None
        for i in range(self.depth):
            msg, virt = self.mixs[i](h, virt, edges, graph_id, batch_size, key=keys[2*i])
            h = self.meta[i](h, msg, key=keys[2*i+1])

        return self.head(h, virt, graph_id, batch_size)[1:]

    def _get_edge(self, batch):
        """Build edge tuple list for hops 1, 2, 3, and 4."""
        num_nodes = batch['node_feat'].shape[0]
        edges = []
        for suffix in EDGE_SUFFIXES:
            edge_index = batch[f'edge{suffix}_index']   # (2, E_pad)
            edge_attr  = batch[f'edge{suffix}_feat']    # (E_pad, num_bond_features)
            edge_mask  = batch[f'edge{suffix}_batch'] > 0    # (E_pad,)
            deg = segment_sum(jnp.ones((edge_mask.shape[0], 1), dtype=edge_index.dtype), edge_index[1], num_nodes).squeeze(-1).clip(1, None)
            edges.append((edge_index, edge_attr, deg, edge_mask))
        return edges


def get_model(key):
    return DenseGIN(depth=6, width=256, num_head=16, dim_head=16, key=key)
