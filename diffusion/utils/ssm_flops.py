import jax.numpy as jnp
import flax.linen as nn
from diffusion.models.layers.s4d_jax import S4DAttention, S4D, S4DKernel 
from diffusion.utils.flops_utils import *

def gelu_flops(shape, layer: nn.gelu, backward=False, unit=1):
    flops = float(jnp.prod(shape)) / unit
    if backward:
        flops *= 3
    return flops

def s4d_attn_flops(shape, layer: S4DAttention, backward=False, unit=1):
    assert isinstance(layer, S4DAttention), f"{type(layer)}"
    flops = 0
    block_attn_flops_dict = {'DiTBlockAttn-kqvFlops': 0, 'DiTBlockAttn-attnFlops': 0, 'DiTBlockAttn-outFlops': 0}

    ## forward_u = self.activation(self.pre_fdense(u))
    forward_shape, flops_ = dense_flops(
        shape, layer=layer.pre_fdense, backward=backward, unit=unit)
    flops += flops_
    gelu_flops_ = gelu_flops(forward_shape, layer.activation, backward=backward, unit=unit)
    flops += gelu_flops_
    block_attn_flops_dict['DiTBlockAttn-kqvFlops'] += flops_ + gelu_flops_

    ## backward_u = self.activation(self.pre_bdense(jnp.flip(u, axis=1)))
    backward_shape, flops_ = dense_flops(
        shape, layer=layer.pre_bdense, backward=backward, unit=unit)
    flops += flops_
    gelu_flops_ = gelu_flops(backward_shape, layer.activation, backward=backward, unit=unit)
    flops += gelu_flops_
    block_attn_flops_dict['DiTBlockAttn-kqvFlops'] += flops_ + gelu_flops_

    ## forward_u, _ = self.fssm(forward_u)
    forward_shape, flops_ = s4d_flops(
        forward_shape, layer.fssm, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_

    ## backward_u, _ = self.bssm(backward_u)
    backward_shape, flops_ = s4d_flops(
        backward_shape, layer.bssm, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_

    ## forward_u = self.post_fdense(forward_u)
    forward_shape, flops_ = dense_flops(
        forward_shape, layer=layer.post_fdense, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_

    ## backward_u = jnp.flip(self.post_bdense(backward_u), axis=1)
    backward_shape, flops_ = dense_flops(
        backward_shape, layer=layer.post_bdense, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_

    ## y = jnp.multiply(forward_u, backward_u)
    assert (forward_shape == backward_shape).all(), f"{forward_shape} {backward_shape}"
    shape = forward_shape
    flops += float(jnp.prod(shape)) / unit
    block_attn_flops_dict['DiTBlockAttn-outFlops'] += float(jnp.prod(shape)) / unit
    if backward:
        flops += 2*float(jnp.prod(shape)) / unit
        block_attn_flops_dict['DiTBlockAttn-outFlops'] += 2*float(jnp.prod(shape)) / unit

    ## y = self.post_dense(y)
    shape, flops_ = dense_flops(
        shape, layer=layer.post_dense, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-outFlops'] += flops_
    
    return shape, flops, block_attn_flops_dict
    
def s4d_flops(shape, layer: S4D, backward=False, unit=1):
    assert isinstance(layer, S4D), f"{type(layer)}"
    alpha = 2.5
    flops = 0

    if not layer.transposed:
        shape_ = jnp.array([shape[0], shape[2], shape[1]])
    L = shape_[-1]

    ## k = self.kernel(L)
    try:
        flops += s4d_kernel_flops(L, layer=layer.kernel, backward=backward, unit=unit)
    except:
        flops += s4d_kernel_flops(L, layer=(layer.h, layer.n), backward=backward, unit=unit)
        
    
    ## Convolution
    D = layer.h
    if hasattr(layer, 'expansion'):
        D *= layer.expansion
    fft_flops = 2*alpha*L*jnp.log(2*L)*D/unit
    fft_flops += 2*alpha*L*jnp.log(2*L)*D/unit
    fft_flops += 2*alpha*L*jnp.log(2*L)*D/unit
    if backward:
        fft_flops *= 3
    flops += fft_flops

    ## y = y + u * self.D[:, None]
    flops += float(jnp.prod(shape_))*2 / unit
    if backward:
        flops += 2*float(jnp.prod(shape_))*2 / unit

    if not layer.transposed:
        shape_ = jnp.array([shape_[0], shape_[2], shape_[1]])

    assert (shape_ == shape).all(), f"{shape_} {shape}"

    return shape_, flops

def s4d_kernel_flops(L, layer: S4DKernel, backward=False, unit=1):
    assert isinstance(layer, S4DKernel) or isinstance(layer, tuple), f"{type(layer)}"
    flops = 0
    if isinstance(layer, S4DKernel):
        flops += layer.d_model / unit
        A_shape = jnp.array(layer.log_A_real.shape)
    else:
        H, N = layer
        flops += H / unit
        A_shape = jnp.array([H, N//2])
    H, N = A_shape

    flops += (float(jnp.prod(A_shape)) + float(jnp.prod(A_shape))) / unit
    
    flops += float(H*N) / unit
    flops += float(H*N*L) / unit
    flops += float(H*N*2) / unit
    flops += float(H*N*L + H*L*(2*N-1) + H*L) / unit

    if backward:
        flops *= 3

    return flops