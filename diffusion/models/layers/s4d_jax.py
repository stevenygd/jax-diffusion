import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import repeat
import math
from typing import Dict, Any
from dataclasses import field
import numpy as np
from diffusion.models.utils import precision_str_to_type

def _c2r(x):
    return jnp.concatenate([x.real, x.imag], axis=-1)

def _r2c(x):
    return x[..., :x.shape[-1]//2] + 1j * x[..., x.shape[-1]//2:]

class S4DKernel(nn.Module):
    d_model: int
    N: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1
    lr: float = 5e-5
    seed: int = 0

    def setup(self):
        H = self.d_model
        N = self.N

        # Use numpy for initialization to match PyTorch
        rng = np.random.RandomState(self.seed)

        # Initialize log_dt
        log_dt = rng.uniform(0, 1, (H,)) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
        self.log_dt = self.param('log_dt', lambda _: jnp.array(log_dt))

        # Initialize C
        C_real = rng.randn(H, N // 2)
        C_imag = rng.randn(H, N // 2)
        C = C_real + 1j * C_imag
        self.C = self.param('C', lambda _: _c2r(jnp.array(C)))

        # Initialize A
        log_A_real = jnp.log(0.5 * jnp.ones((H, N//2)))
        A_imag = math.pi * repeat(jnp.arange(N//2), 'n -> h n', h=H)
        
        self.log_A_real = self.param('log_A_real', lambda _: log_A_real)
        self.A_imag = self.param('A_imag', lambda _: A_imag)

    def __call__(self, L):
        # Materialize parameters
        dt = jnp.exp(self.log_dt)
        C = _r2c(self.C)
        A = -jnp.exp(self.log_A_real) + 1j * self.A_imag

        # Vandermonde multiplication
        dtA = A * dt[:, None]
        K = dtA[:, :, None] * jnp.arange(L)
        C = C * (jnp.exp(dtA) - 1.) / A
        K = 2 * jnp.einsum('hn,hnl->hl', C, jnp.exp(K)).real

        return K

class S4D(nn.Module):
    d_model: int
    d_state: int = 64
    dropout: float = 0.0
    transposed: bool = False
    kernel_args: Dict[str, Any] = field(default_factory=dict)

    def setup(self):
        self.h = self.d_model
        self.n = self.d_state
        self.d_output = self.h
        self.D = self.param('D', jax.random.normal, (self.h,))
        self.kernel = S4DKernel(self.h, N=self.n, **self.kernel_args)

    def __call__(self, u, train: bool = True):
        if not self.transposed:
            u = jnp.transpose(u, (0, 2, 1))
        L = u.shape[-1]

        # Compute SSM Kernel
        k = self.kernel(L)

        # Convolution
        k_f = jnp.fft.rfft(k, n=2*L)
        u_f = jnp.fft.rfft(u, n=2*L)
        y = jnp.fft.irfft(u_f*k_f, n=2*L)[..., :L]

        # Compute D term in state space equation
        y = y + u * self.D[:, None]
        
        if not self.transposed:
            y = jnp.transpose(y, (0, 2, 1))
        return y, None

class S4DAttention(nn.Module):
    # Hidden dimension
    dim: int
    
    # Placeholder
    num_heads: int
    qkv_bias: bool = False 
   
    # SSM configs
    d_expansion: int = 1
    # d_state: int = 64
    # rnn_dim=128 outside, and d_state=self.rnn_dim
    d_state: int = 128
    dropout_rate: float = 0.0
    transposed: bool = False
    kernel_args: Dict[str, Any] = field(default_factory=dict)
    
    # Computation precision
    dtype: str = "float32"

    def setup(self):
        self.h = self.dim
        self.i = self.h * self.d_expansion

        self.pre_bdense = nn.Dense(
            self.i, 
            dtype=precision_str_to_type(self.dtype),
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), 
                ("qkv/inp", "qkv/out")),
            bias_init=nn.initializers.zeros_init(),
        )
        self.pre_fdense = nn.Dense(
            self.i, dtype=precision_str_to_type(self.dtype),
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), 
                ("qkv/inp", "qkv/out")),
            bias_init=nn.initializers.zeros_init(),
        ) 
        self.post_bdense = nn.Dense(
            self.i, dtype=precision_str_to_type(self.dtype),
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), 
                ("ssm/post_dense/inp", "ssm/post_dense/out")),
            bias_init=nn.initializers.zeros_init()
        ) 
        self.post_fdense = nn.Dense(
            self.i, dtype=precision_str_to_type(self.dtype),
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), 
                ("ssm/post_dense/inp", "ssm/post_dense/out")),
            bias_init=nn.initializers.zeros_init()
        )
        self.post_dense = nn.Dense(
            self.i, dtype=precision_str_to_type(self.dtype),
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), ("proj/inp", "proj/out")),
            bias_init=nn.initializers.zeros_init(),
        )

        self.bssm = S4D(d_model=self.i, **self.kernel_args)
        self.fssm = S4D(d_model=self.i, **self.kernel_args)

        self.dropout = nn.Dropout(rate=self.dropout_rate)
        # self.activation = nn.gelu

    def activation(self, x):
        return nn.gelu(x.astype(jnp.float32))

    def __call__(self, u, training: bool = False, return_aux: bool = False):
        forward_u = self.activation(self.pre_fdense(u))
        backward_u = self.activation(self.pre_bdense(jnp.flip(u, axis=1)))

        forward_u, _ = self.fssm(forward_u)
        backward_u, _ = self.bssm(backward_u)

        forward_u = self.post_fdense(forward_u)
        backward_u = jnp.flip(self.post_bdense(backward_u), axis=1)

        y = jnp.multiply(forward_u, backward_u)
        y = self.post_dense(y)

        y = self.dropout(y, deterministic=not training)

        if return_aux:
            return y, {}

        return y