import numpy as np
from diffusion.models.uvit import (
    DownSample, UpSample, Stage
)
from diffusion.utils.flops_utils import *

def uvit_downsample_flops(x_shape, layer: DownSample, backward=False, unit=1):
    assert isinstance(layer, DownSample), f"Expected DownSample, got {type(layer)}"
    B, L, C = x_shape
    H = W = int(np.sqrt(L))
    assert H*W==L
    x_shape = jnp.array([B, H//2, W//2, C])
    x_shape, flops = dense_flops(x_shape, nn.Dense(layer.out_dim), backward=backward, unit=unit)
    x_shape = jnp.array([B, L//4, layer.out_dim])
    
    return x_shape, flops

def uvit_upsample_flops(x_shape, layer: UpSample, backward=False, unit=1):
    assert isinstance(layer, UpSample), f"Expected Upsample, got {type(layer)}"
    B, L, C = x_shape
    H = W = int(np.sqrt(L))
    assert H*W==L
    x_shape = jnp.array([B, H*2, W*2, C])
    x_shape, flops = dense_flops(x_shape, nn.Dense(layer.out_dim), backward=backward, unit=unit)
    x_shape = jnp.array([B, L*4, layer.out_dim])
    
    return x_shape, flops