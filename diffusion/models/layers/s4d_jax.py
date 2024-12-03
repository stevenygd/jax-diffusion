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
