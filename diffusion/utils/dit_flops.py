import jax.numpy as jnp
from diffusion.models.dit import (
    FinalLayer, DiTBlock, TimestepEmbedder, LabelEmbedder, PatchEmbed, ConvNetEmbed, SimpleDiffusionEmbedder, SimpleDiffusionUnpacify)
from diffusion.models.uvit import (
    DownSample, UpSample, Stage
)
from diffusion.models.utils import Identity
from diffusion.models.layers.s4d_jax import S4DAttention
from diffusion.models.layers.attention import (
    MTTTAttention, TorchAttention, LinearAttention, JaxAttention, DiffuserAttention
)
from diffusion.models.layers.ffn import Mlp
from diffusion.models.layers.moe_mlp import SwitchFeedForward
from diffusion.models.layers.mttt_lm import TTTLMBiDirAttention
from diffusion.utils.mt3_flops import mt3_attn_flops, mt3_lm_bdar_flops
from diffusion.utils.vae_flops import convnet_emb_flops
from diffusion.utils.ssm_flops import s4d_attn_flops
from diffusion.utils.moe_mlp_flops import switch_ff_flops
from diffusion.utils.simpdiff_emb_flops import simpdiff_emb_flops, simpdiff_unpatchify_flops
from diffusion.utils.uvit_flops import uvit_downsample_flops, uvit_upsample_flops
from diffusion.utils.flops_utils import * 
from typing import Union


def dit_attn_flops(x_shape, layer, backward=False, unit=1):
    flops = 0 
    if isinstance(layer, TorchAttention):
        x_shape, flops, block_attn_flops_dict = self_attn_flops(
            x_shape, layer, backward=backward, unit=unit)
    elif isinstance(layer, JaxAttention) or isinstance(layer, DiffuserAttention):
        x_shape, flops, block_attn_flops_dict = jax_attn_flops(
            x_shape, layer, backward=backward, unit=unit)
    elif isinstance(layer, MTTTAttention):
        x_shape, flops, block_attn_flops_dict = mt3_attn_flops(
            x_shape, layer, backward=backward, unit=unit)
    elif isinstance(layer, LinearAttention):
        x_shape, flops, block_attn_flops_dict = linear_attn_flops(
            x_shape, layer, backward=backward, unit=unit)
    elif isinstance(layer, S4DAttention):
        x_shape, flops, block_attn_flops_dict = s4d_attn_flops(
            x_shape, layer, backward=backward, unit=unit)
    elif isinstance(layer, TTTLMBiDirAttention):
        x_shape, flops, block_attn_flops_dict = mt3_lm_bdar_flops(
            x_shape, layer, backward=backward, unit=unit)
    else:
        # pass
        raise ValueError(f"Undefined attention type for flops counting: {type(layer)}")

    return x_shape, flops, block_attn_flops_dict


def dit_block_flops(x_shape, c_shape, layer: DiTBlock, backward=False, unit=1):
    assert isinstance(layer, DiTBlock), f"Expected DiTBlock, got {type(layer)}"
    # NOTE: assuming that the input dimension dictate the FLOPS
    flops = 0 
    block_flops_dict = {'DiTBlockConditionFlops': 0, 'DiTBlockAttnFlops': 0, 'DiTBlockMlpFlops': 0}
    # 2 normalization each 2x prod(shape) for stats, 1xprod(shape) for norm
    # 2 modulation, each time x * scale + shift
    x_shape, mod_flops = ln_flops(
        x_shape, layer=None, backward=backward, unit=unit)
    flops += mod_flops * 2
    block_flops_dict['DiTBlockConditionFlops'] += mod_flops * 2

    # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c)
    _, flops_ = dense_flops(
        c_shape, 
        layer=layer.adaLN_mlp,
        backward=backward, 
        unit=unit)
    flops += flops_
    block_flops_dict['DiTBlockConditionFlops'] += flops_
    
    # activation
    c_act_flops = jnp.prod(c_shape)
    if backward:
        c_act_flops += c_act_flops * 2
    flops += c_act_flops / unit
    block_flops_dict['DiTBlockConditionFlops'] += c_act_flops / unit
    
    # Attention
    x_shape, flops_, block_attn_flops_dict = dit_attn_flops(
        x_shape, layer.attn, backward=backward, unit=unit)
    flops += flops_
    block_flops_dict['DiTBlockAttnFlops'] += flops_
    block_flops_dict['DiTBlockAttnFlopsBreakdown'] = block_attn_flops_dict

    # Also MLP flops
    if isinstance(layer.mlp, Mlp):
        x_shape, flops_ = mlp_flops(
            x_shape, layer.mlp, backward=backward, unit=unit)
    elif isinstance(layer.mlp, SwitchFeedForward):
        x_shape, flops_ = switch_ff_flops(
            x_shape, layer.mlp, backward=backward, unit=unit)
        
    flops += flops_
    block_flops_dict['DiTBlockMlpFlops'] += flops_
    
    return (x_shape, c_shape), flops, block_flops_dict


def final_layer_flops(x_shape, c_shape, layer: FinalLayer, backward=False, 
                      unit=1):
    flops = 0
    
    # 1 normalization each 2x prod(shape) for stats, 1xprod(shape) for norm
    # 1 modulation, each time x * scale + shift
    x_shape, flops_ = ln_flops(
        x_shape, layer=None, backward=backward, unit=unit)
    flops += flops_
    
    # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c)
    if hasattr(layer, 'adaLN_mlp'):
        c_shape, flops_ = dense_flops(
            c_shape, layer=layer.adaLN_mlp, backward=backward, unit=unit)
        flops += flops_
    
    c_act_flops = jnp.prod(c_shape)
    if backward:
        c_act_flops += c_act_flops * 2
    flops += c_act_flops / unit
    
    if hasattr(layer, 'linear'):
        x_shape, flops_ = dense_flops(
            x_shape, layer=layer.linear, backward=backward, unit=unit)
        flops += flops 
    
    return (x_shape, c_shape), flops

def x_emb_flops(x_shape, c_shape, layer: PatchEmbed, backward=False, unit=1):
    ps = layer.patch_size
    B, C, H, W = x_shape
    H, W = H // ps, W // ps
    D = layer.embed_dim
    flops = (B * H * W) * (C * D * ps ** 2) / unit
    if backward:
        flops *= 3
    x_shape = jnp.array([B, H * W, D], dtype=jnp.float32)
    return x_shape, flops
    
def t_emb_flops(t_shape, layer: TimestepEmbedder, backward=False, unit=1):
    flops = 0
    B = t_shape[0]
    D = layer.frequency_embedding_size
    t_shape = jnp.array([B, D])
    t_shape, flops_ = dense_flops(
        t_shape, layer=layer.dense1, backward=backward, unit=unit)
    flops += flops_
    
    # SILU FLOPs!!! CAN BE MORE
    flops += (jnp.prod(t_shape) * 3 if backward else jnp.prod(t_shape)) / unit
    
    t_shape, flops_ = dense_flops(
        t_shape, layer=layer.dense2, backward=backward, unit=unit)
    flops += flops_
    
    return t_shape, flops
    
def y_emb_flops(y_shape, layer: LabelEmbedder, backward=False, unit=1):
    B = y_shape[0]
    y_shape = jnp.array([B, layer.hidden_size])
    return y_shape, 0 / unit


def dit_flops(x_shape, y_shape, t_shape, dit, backward=False, unit=1):
    x_shape = jnp.array(x_shape, dtype=jnp.float32)
    y_shape = jnp.array(y_shape, dtype=jnp.float32)
    t_shape = jnp.array(t_shape, dtype=jnp.float32)
    flops = 0
    print(f"Initial x_shape={x_shape}, y_shape={y_shape}, t_shape={t_shape}")
    
    # Compute temb flops 
    t_shape, flops_ = t_emb_flops(
        t_shape, dit.t_embedder, backward=backward, unit=unit)
    flops += flops_
    print(f't emb flops: {flops_}, {flops} \t t_shape={t_shape}')
    
    # Compute yemb flops 
    c_shape, flops_ = y_emb_flops(
        y_shape, dit.y_embedder, backward=backward, unit=unit)
    flops += flops_
    print(f'y emb flops: {flops_}, {flops} \t c_shape={c_shape}')
    assert list(t_shape) == list(c_shape), f"t_shape={t_shape} c_shape={c_shape}"
    print(f'x_shape={x_shape} c_shape=(=t_shape){c_shape}')
    
    # Compute xemb flops 
    if isinstance(dit.x_embedder, ConvNetEmbed):
        x_shape, flops_ = convnet_emb_flops(
            x_shape, c_shape, dit.x_embedder, backward=backward, unit=unit)
    elif isinstance(dit.x_embedder, PatchEmbed):
        x_shape, flops_ = x_emb_flops(
            x_shape, c_shape, dit.x_embedder, backward=backward, unit=unit)
    elif isinstance(dit.x_embedder, SimpleDiffusionEmbedder):
        x_shape, flops_ = simpdiff_emb_flops(
            x_shape, c_shape, dit.x_embedder, backward=backward, unit=unit)
    else:
        raise ValueError(f"Undefined x_embedder type for flops counting: {type(dit.x_embedder)}")
    flops += flops_
    print(f'x emb flops: {flops_}, {flops} \t x_shape={x_shape}')
    
    if not hasattr(dit, 'blocks'):
        assert dit.package_name=='uvit', f"Expected uvit package"

        n_stage = len(dit.stages_frd)
        assert n_stage == len(dit.stages_bck), f"Forward and backward stages should be equal"

        # Forward stage
        skip_lst_shape = {}
        for lid, stage in enumerate(dit.stages_frd):
            skip_lst_shape[lid]=x_shape if lid in dit.skip_idxs else None
            for block in stage.blocks:
                (x_shape, c_shape), flops_, block_flops_dict = dit_block_flops(
                    x_shape, c_shape, block, backward=backward, unit=unit)
                flops += flops_
            
            if isinstance(dit.down_layers[lid], Identity):
                pass
            elif isinstance(dit.down_layers[lid], DownSample):
                x_shape, flops_ = uvit_downsample_flops(
                    x_shape, dit.down_layers[lid], backward=backward, unit=unit)
                flops += flops_
            else:
                raise ValueError(f"Undefined down_layer type for flops counting: {type(dit.down_layers[lid])}")
        
            print(f"\tDownStage-{lid} {flops_} {flops}\tx_shape={x_shape} c_shape={c_shape}")
        
        # Center stage
        stage = dit.center_stage
        for block in stage.blocks:
            (x_shape, c_shape), flops_, block_flops_dict = dit_block_flops(
                x_shape, c_shape, block, backward=backward, unit=unit)
            flops += flops_
        print(f"\tCenterStage {flops_} {flops}\tx_shape={x_shape} c_shape={c_shape}")

        # Backward stage
        for lid, stage in enumerate(dit.stages_bck):
            if isinstance(dit.up_layers[lid], Identity):
                pass
            elif isinstance(dit.up_layers[lid], UpSample):
                x_shape, flops_ = uvit_upsample_flops(
                    x_shape, dit.up_layers[lid], backward=backward, unit=unit)
                flops += flops_
            else:
                raise ValueError(f"Undefined up_layer type for flops counting: {type(dit.up_layers[lid])}")
            
            skip_shape = skip_lst_shape[n_stage-1-lid]
            if skip_shape!=None:
                assert (x_shape[:-1]==skip_shape[:-1]).all(), f"x_shape: {x_shape[:-1]} skip_shape: {skip_shape[:-1]}"
                x_shape = jnp.array([*x_shape[:-1], x_shape[-1] + skip_shape[-1]])

            for block in stage.blocks:
                (x_shape, c_shape), flops_, block_flops_dict = dit_block_flops(
                    x_shape, c_shape, block, backward=backward, unit=unit)
                flops += flops_
            
            print(f"\tUpStage-{lid} {flops_} {flops}\tx_shape={x_shape} c_shape={c_shape}")


    elif hasattr(dit.blocks, '__iter__'):
        for i, block in enumerate(dit.blocks):
            (x_shape, c_shape), flops_, block_flops_dict = dit_block_flops(
                x_shape, c_shape, block, backward=backward, unit=unit)
            flops += flops_

            skip_layer_flops = 0
            if i in dit.skip_layers:
                _, skip_layer_flops = dense_flops(
                    jnp.array([x_shape[0], x_shape[1], 2 * x_shape[2]]), dit.skip_blocks[i], backward=backward, unit=unit)
                skip_layer_flops += jnp.prod(x_shape) / unit * (3 if backward else 1)
                flops += skip_layer_flops
            print(f"\tDIT-{i}: {flops_+skip_layer_flops} {flops} x_shape={x_shape} c_shape={c_shape}")

    else:
        if isinstance(dit.blocks, DiTBlock):
            block = dit.blocks
        else:
            import jax
            block = DiTBlock(**dit.blocks.block_args)
            params = block.init(jax.random.PRNGKey(0),
                        jnp.ones(jnp.array([1,*(x_shape[1:])], dtype=jnp.int32)),
                        jnp.ones(jnp.array([1,*(c_shape[1:])], dtype=jnp.int32)),
                        True,
                        False)
            block = block.bind(params)
        
        for i in range(dit.depth):
            (x_shape, c_shape), flops_, block_flops_dict = dit_block_flops(
                x_shape, c_shape, block, backward=backward, unit=unit)
            flops += flops_

            skip_layer_flops = 0
            if i in dit.skip_layers:
                _, skip_layer_flops = dense_flops(
                    jnp.array([x_shape[0], x_shape[1], 2 * x_shape[2]]), dit.skip_blocks[i], backward=backward, unit=unit)
                skip_layer_flops += jnp.prod(x_shape) / unit * (3 if backward else 1)
                flops += skip_layer_flops

            print(f"\tDIT-{i}: {flops_+skip_layer_flops} {flops} x_shape={x_shape} c_shape={c_shape}")
    

    (x_shape, c_shape), flops_ = final_layer_flops(
        x_shape, c_shape, dit.final_layer, backward=backward, unit=unit)
    flops += flops_
    
    if isinstance(dit.unpatchify, SimpleDiffusionUnpacify):
        x_shape, flops_ = simpdiff_unpatchify_flops(
                x_shape, c_shape, dit.unpatchify, backward=backward, unit=unit)
        flops += flops_

    print(f'final layer flops: {flops_}, {flops} \t x_shape={x_shape} c_shape={c_shape}')
    
    # x_shape: 256, 256, 32
    # c_shape: 256, 768
    
    return flops, block_flops_dict