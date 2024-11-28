"""
Adapted from:
https://github.com/test-time-training/ttt-lm-jax/blob/main/ttt/models/ttt_layer.py
"""

import math
import numpy as np

from functools import partial
from typing import Any, Union, Sequence, Optional, Tuple

import jax
import jax.numpy as jnp
import flax
from jax import vmap
from jax.tree_util import tree_map
from jax.sharding import PartitionSpec as PS
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from diffusion.models.utils import precision_str_to_type 


Axes = Union[int, Sequence[int]]


def get_gradient_checkpoint_policy(name):
    return {
        "everything_saveable": jax.checkpoint_policies.everything_saveable,
        "nothing_saveable": jax.checkpoint_policies.nothing_saveable,
        "checkpoint_dots": jax.checkpoint_policies.checkpoint_dots,
        "checkpoint_dots_with_no_batch_dims": jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]


def scan_remat_every_n_iterations_scan(f, n, carry, x):
    """
    Remat every n mini batches.
    """
    if n > 1:
        x_grouped = tree_map(lambda x: x.reshape((-1, n, *x.shape[1:])), x)
        carry, y_grouped = jax.lax.scan(jax.remat(partial(jax.lax.scan, f), prevent_cse=False), carry, x_grouped)
    elif n < 0:
        x_grouped = tree_map(lambda x: x.reshape((-1, *x.shape[1:])), x)
        carry, y_grouped = jax.lax.scan(f, carry, x_grouped)
    else:
        assert n == 1
        x_grouped = tree_map(lambda x: x.reshape((-1, *x.shape[1:])), x)
        carry, y_grouped = jax.remat(
            partial(jax.lax.scan, f), prevent_cse=False)(carry, x_grouped)
    y = tree_map(lambda x: x.reshape((-1, *x.shape[2:])), y_grouped)
    return carry, y


def get_multi_head_params(self, params, param_dtype, kernel_init="normal", std=0.02):
    flat_params = flax.traverse_util.flatten_dict(params, sep="/")
    for k in flat_params.keys():
        new_shape = (self.num_heads, *flat_params[k].shape)
        if "scale" in k:
            p = self.param(k, jax.nn.initializers.ones, new_shape, param_dtype)
        elif "kernel" in k:
            if kernel_init == "normal":
                initializer = nn.initializers.normal(std)
            elif kernel_init == "zeros":
                initializer = nn.initializers.zeros
            elif kernel_init == "ones":
                initializer = nn.initializers.ones
            else:
                raise NotImplementedError("Initializer %s Not Implemented." % (kernel_init))
            p = self.param(k, initializer, new_shape, param_dtype)
        else:
            p = self.param(k, jax.nn.initializers.zeros, new_shape, param_dtype)
        flat_params[k] = p
    params_init = flax.traverse_util.unflatten_dict(flat_params, sep="/")
    return params_init


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = np.arange(end)
    freqs = np.outer(t, freqs).astype(dtype)
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def apply_rotary_emb(
    xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray, dtype: jnp.dtype = jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype), xk_out.astype(dtype)


def diff_gelu(x):
    tanh_out = jnp.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff


class LinearLayerTemplate(nn.Module):
    width: int
    use_bias: bool
    name: str
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.width, use_bias=self.use_bias, name=self.name, dtype=self.dtype, param_dtype=self.param_dtype
        )(x)
        return x


class LayerNormTemplate(nn.Module):
    name: str
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(name=self.name, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return x


class TTTBase(nn.Module):
    width: int
    num_heads: int
    config: Any = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None
    set_up_qkvo: bool = True
    learn_token_idx: bool = True
    sigmoid_learnable_token_idx: bool = False
    use_ar_mask: bool = True

    def setup(self):
        # self.width = self.config.hidden_size
        # self.num_heads = self.config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.mini_batch_size = self.config.mini_batch_size
        self.n_mini_batch = self.config.max_sequence_length // self.mini_batch_size
        self.seq_shape = (self.n_mini_batch, self.mini_batch_size)
        self.freqs_cis = precompute_freqs_cis(
            self.head_dim, self.mini_batch_size * 2, 
            theta=self.config.rope_theta, dtype=self.dtype
        )

        if self.set_up_qkvo:
            self.setup_qkvo()
        self.setup_token_idx()
        self.setup_ttt_lr_gate()

        self.ttt_norm = LayerNormTemplate(dtype=self.dtype, param_dtype=self.param_dtype)
        ttt_norm_params = self.ttt_norm.init(
            jax.random.PRNGKey(0), jnp.ones([1, self.head_dim]))["params"]
        self.ttt_norm_params = get_multi_head_params(
            self, ttt_norm_params, param_dtype=self.param_dtype, kernel_init="layer_norm"
        )
        self.post_norm = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)

        self.ttt_params = ()

    def setup_qkvo(self):
        self.wq = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wk = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wv = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wo = nn.Dense(
            self.width,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

    def setup_token_idx(self):
        if self.use_ar_mask:
            self.token_idx = 1.0 / jnp.arange(
                1, self.mini_batch_size + 1, dtype=jnp.float32)
        else:
            self.token_idx = jnp.ones((1,)) / float(self.mini_batch_size + 1)
        if self.learn_token_idx:
            self.learnable_token_idx = self.param(
                "learnable_token_idx", nn.initializers.zeros, 
                (self.mini_batch_size,), jnp.float32
            )

    def setup_ttt_lr_gate(self):
        self.learnable_ttt_lr = LinearLayerTemplate(
            width=1, use_bias=True, name="learnable_ttt_lr", dtype=self.dtype, param_dtype=self.param_dtype
        )
        learnable_ttt_lr_params = self.learnable_ttt_lr.init(
            jax.random.PRNGKey(0), jnp.ones([1, self.width]))["params"]
        self.learnable_ttt_lr_params = get_multi_head_params(
            self,
            learnable_ttt_lr_params,
            param_dtype=self.param_dtype,
            kernel_init="normal",
            std=self.config.initializer_range,
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _split_mini_batches(self, hidden_states):
        B, N, num_head, head_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(
            B, *self.seq_shape, self.num_heads, self.head_dim).transpose(
            0, 3, 1, 2, 4
        )
        return hidden_states

    def get_qkv_projections(self, batch):
        XQ, XK, XV = self.wq(batch), self.wk(batch), self.wv(batch)
        return XQ, XK, XV

    def get_eta(self, X):
        learnable_ttt_lr = vmap(
            lambda x, p: self.learnable_ttt_lr.apply({"params": p}, x), 
            axis_name="head", in_axes=[None, 0], out_axes=1
        )(X, self.learnable_ttt_lr_params)
        learnable_ttt_lr = nn.sigmoid(learnable_ttt_lr)
        learnable_ttt_lr = learnable_ttt_lr.transpose(0, 1, 2, 4, 3)
        
        if self.learn_token_idx:
            if self.sigmoid_learnable_token_idx:
                # The 2x is not stable
                # E.g. https://wandb.ai/xnf/mt3_res256/groups/group-ttt%2Fpixels%2Fps2-TTT-S-lmbd-mlp-2024-Sep-05-16-24-40/workspace?nw=nwusergy46
                # token_idx = 2 * nn.sigmoid(self.learnable_token_idx) * self.token_idx
                token_idx = nn.sigmoid(self.learnable_token_idx) * self.token_idx
            else: 
                token_idx = self.learnable_token_idx + self.token_idx
        else:
            token_idx = self.token_idx
        token_idx = jnp.clip(token_idx, a_min=0.0)

        eta = (
            (self.config.ttt_base_lr * token_idx).reshape(1, 1, 1, token_idx.shape[0], -1)
            * learnable_ttt_lr
            / self.head_dim
        )
        return eta

    def get_ttt_inputs(self, batch, position_ids):
        B, N, F = batch.shape
        n_mini_batch = N // self.mini_batch_size
        X = batch.reshape(B, *self.seq_shape, self.width)

        XQ, XK, XV = self.get_qkv_projections(batch)

        if self.config.output_ttt_stats:
            XV_last_in_mini_batch = XV[:, :: self.mini_batch_size, ...].reshape(
                B, n_mini_batch, self.num_heads, self.head_dim
            )
            XK_last_in_mini_batch = XK[:, :: self.mini_batch_size, ...].reshape(
                B, n_mini_batch, self.num_heads, self.head_dim
            )
            ssl_tgt_last_in_mini_batch = XV_last_in_mini_batch - XK_last_in_mini_batch
            ssl_tgt_mean = (XV - XK).mean(axis=1, keepdims=True).reshape(
                B, 1, self.num_heads, self.head_dim)
            ssl_tgt_last_in_mini_batch_from_mean_mse = (
                (ssl_tgt_last_in_mini_batch - ssl_tgt_mean) ** 2).mean(
                axis=(0, 2, 3)
            )
        else:
            ssl_tgt_last_in_mini_batch_from_mean_mse = None

        XQ = nn.with_sharding_constraint(XQ, PS(("dp", "fsdp"), None, "tp"))
        XK = nn.with_sharding_constraint(XK, PS(("dp", "fsdp"), None, "tp"))
        XV = nn.with_sharding_constraint(XV, PS(("dp", "fsdp"), None, "tp"))

        XQ = self._split_heads(XQ)
        XK = self._split_heads(XK)
        XV = self._split_heads(XV)
        
        if position_ids is not None:
            freqs_cis = jnp.take(self.freqs_cis, position_ids % self.mini_batch_size, axis=0)
            XQ, XK = apply_rotary_emb(XQ, XK, freqs_cis=freqs_cis, dtype=self.dtype)

        XQ = self._split_mini_batches(XQ)
        XK = self._split_mini_batches(XK)
        XV = self._split_mini_batches(XV)

        eta = self.get_eta(X)

        return (XQ, XK, XV, eta, (ssl_tgt_last_in_mini_batch_from_mean_mse,))

    def apply_gate(self, hidden_states, ttt_output):
        return ttt_output

    def project_ttt_outputs(self, XQW_batch):
        z_batch = self.wo(XQW_batch)
        return z_batch

    def process_mini_batch(
        self,
        XQ_mini_batch,
        XK_mini_batch,
        XV_mini_batch,
        eta_mini_batch,
        ttt_params_init,
        ttt_params_mini_batch_init,
        ttt_norm_params,
    ):
        raise NotImplementedError

    def ttt(self, XQ, XK, XV, eta):
        B, N = XV.shape[0], XV.shape[2] * XV.shape[3]

        @partial(vmap, axis_name="batch")
        def update_embed(XQ, XK, XV, eta):
            @partial(vmap, axis_name="head")
            def parallelize_over_heads(XQ, XK, XV, eta, ttt_params_init, ttt_norm_params):
                def compute_mini_batch(ttt_params_mini_batch_init, inputs):
                    XQ_mini_batch = inputs["XQ"]
                    XK_mini_batch = inputs["XK"]
                    XV_mini_batch = inputs["XV"]
                    eta_mini_batch = inputs["eta"]

                    ttt_params_last_in_mini_batch, outputs = self.process_mini_batch(
                        XQ_mini_batch,
                        XK_mini_batch,
                        XV_mini_batch,
                        eta_mini_batch,
                        ttt_params_init,
                        ttt_params_mini_batch_init,
                        ttt_norm_params,
                    )
                    return ttt_params_last_in_mini_batch, outputs

                inputs = {"XQ": XQ, "XK": XK, "XV": XV, "eta": eta}

                _, outputs = scan_remat_every_n_iterations_scan(
                    compute_mini_batch, self.config.remat_mini_batch_group_size, 
                    ttt_params_init, inputs
                )
                Z, ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1 = outputs
                return (Z.reshape(-1, self.head_dim), ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1)

            outputs = parallelize_over_heads(XQ, XK, XV, eta, self.ttt_params, self.ttt_norm_params)
            return outputs

        outputs = update_embed(XQ, XK, XV, eta)
        Z, ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1 = outputs
        Z = Z.transpose(0, 2, 1, 3).reshape(B, N, -1)

        if self.config.output_ttt_stats:
            ttt_loss_mse_init = ttt_loss_mse_init.mean(axis=(0, 1))
            ttt_loss_mse_step_0 = ttt_loss_mse_step_0.mean(axis=(0, 1))
            ttt_loss_mse_step_1 = ttt_loss_mse_step_1.mean(axis=(0, 1))

        return Z, {
            "ttt_loss_mse_init": ttt_loss_mse_init, 
            "ttt_loss_mse_step0": ttt_loss_mse_step_0, 
            "ttt_loss_mse_step1": ttt_loss_mse_step_1
        }

    def __call__(
        self,
        hidden_states,
        position_ids=None,
        deterministic: bool = True,
        output_ttt_stats: bool = False,
        ttt_lr_mult=1.0,
    ):
        self.config.output_ttt_stats = output_ttt_stats
        del deterministic
        XQ, XK, XV, eta, precompute_stats = self.get_ttt_inputs(
            hidden_states, position_ids=position_ids)
        eta *= ttt_lr_mult
        Z, ttt_stats = self.ttt(XQ, XK, XV, eta)
        Z = self.post_norm(Z)
        Z = self.apply_gate(hidden_states, Z)
        ttt_output = self.project_ttt_outputs(Z)
        return ttt_output, {
            "precompute_stats": precompute_stats, 
            "ttt_stats": ttt_stats
        }


class TTTLinearBase(TTTBase):
    def setup(self):
        super().setup()
        self.W1 = self.param(
            "ttt_dense_0",
            nn.initializers.normal(self.config.initializer_range),
            (self.num_heads, self.head_dim, self.head_dim),
            self.param_dtype,
        )
        self.b1 = self.param("ttt_bias_0", nn.initializers.zeros, (self.num_heads, 1, self.head_dim), self.param_dtype)
        self.ttt_params = (self.W1, self.b1)

    def process_mini_batch(
        self,
        XQ_mini_batch,
        XK_mini_batch,
        XV_mini_batch,
        eta_mini_batch,
        ttt_params_init,
        ttt_params_mini_batch_init,
        ttt_norm_params,
    ):

        W1_init, b1_init = ttt_params_mini_batch_init
        square_eta_mini_batch = eta_mini_batch[: self.mini_batch_size]
        last_eta_in_mini_batch = eta_mini_batch[-1][:, None]

        X1 = XK_mini_batch
        Z1 = X1 @ W1_init + b1_init
        ttt_norm_out, ttt_norm_vjp = jax.vjp(lambda z: self.ttt_norm.apply({"params": ttt_norm_params}, z), Z1)
        ssl_target = XV_mini_batch - XK_mini_batch
        grad_l_wrt_ttt_norm_out = ttt_norm_out - ssl_target
        grad_l_wrt_Z1 = ttt_norm_vjp(grad_l_wrt_ttt_norm_out)[0]

        # Calculate TTT loss using W_init of the current mini-batch
        if self.config.output_ttt_stats:
            ttt_loss_mse_step_0 = (grad_l_wrt_ttt_norm_out[-1] ** 2).mean()
        else:
            ttt_loss_mse_step_0 = None

        # Calculate TTT loss using W_init of the entire sequence
        if self.config.output_ttt_stats:
            W1_0, b1_0 = ttt_params_init
            Z1_0 = X1 @ W1_0 + b1_0
            ttt_norm_out_0 = self.ttt_norm.apply({"params": ttt_norm_params}, Z1_0)
            ttt_loss_mse_init = ((ttt_norm_out_0 - ssl_target)[-1] ** 2).mean()
        else:
            ttt_loss_mse_init = None

        X1_bar = XQ_mini_batch
        if self.use_ar_mask:
            Attn1 = jnp.tril(X1_bar @ X1.transpose(1, 0))
            b1_bar = b1_init - (square_eta_mini_batch * jnp.tril(jnp.ones_like(Attn1))) @ grad_l_wrt_Z1
        else:    
            Attn1 = X1_bar @ X1.transpose(1, 0)
            b1_bar = b1_init - square_eta_mini_batch @ grad_l_wrt_Z1
        Z1_bar = X1_bar @ W1_init - (square_eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
        ttt_norm_out_bar = self.ttt_norm.apply({"params": ttt_norm_params}, Z1_bar)

        output_mini_batch = X1_bar + ttt_norm_out_bar

        W1_bar_last = W1_init - (last_eta_in_mini_batch * X1).transpose(1, 0) @ grad_l_wrt_Z1
        b1_bar_last = b1_init - jnp.sum(last_eta_in_mini_batch * grad_l_wrt_Z1, axis=0, keepdims=True)

        # Calculate ttt loss using the updated W_init by the current mini-batch
        if self.config.output_ttt_stats:
            X1_last_fwd_new = X1[-1:] @ W1_bar_last + b1_bar_last
            X1_last_fwd_new = self.ttt_norm.apply({"params": ttt_norm_params}, X1_last_fwd_new)
            ttt_loss_mse_step_1 = ((X1_last_fwd_new - ssl_target[-1:]) ** 2).mean()
        else:
            ttt_loss_mse_step_1 = None

        ttt_params_mini_batch_new = (W1_bar_last, b1_bar_last)

        return (
            ttt_params_mini_batch_new,
            (output_mini_batch, ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1),
        )


class TTTLinear(TTTLinearBase):
    def setup(self):
        super().setup()
        self.wg = nn.Dense(
            self.width,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

    def setup_qkvo(self):
        self.wq = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        if self.config.remat_conv != "":
            conv_module = nn_partitioning.remat(
                nn.Conv, policy=get_gradient_checkpoint_policy(self.config.remat_conv), prevent_cse=True
            )
        else:
            conv_module = nn.Conv
        self.conv_q = conv_module(
            self.config.hidden_size,
            (self.config.conv_width,),
            padding="CAUSAL",
            feature_group_count=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.conv_k = conv_module(
            self.config.hidden_size,
            (self.config.conv_width,),
            padding="CAUSAL",
            feature_group_count=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.wv = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wo = nn.Dense(
            self.width,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

    def get_qkv_projections(self, batch):
        xqk, XV = self.wq(batch), self.wv(batch)
        XQ = self.conv_q(xqk)
        XK = self.conv_k(xqk)
        return XQ, XK, XV

    def apply_gate(self, hidden_states, ttt_output):
        y = self.wg(hidden_states)
        y = nn.gelu(y)
        output = y * ttt_output
        return output


class TTTMLPBase(TTTBase):
    def setup(self):
        super().setup()
        self.W1 = self.param(
            "ttt_dense_0",
            nn.initializers.normal(self.config.initializer_range),
            (self.num_heads, self.head_dim, 4 * self.head_dim),
            self.param_dtype,
        )
        self.b1 = self.param(
            "ttt_bias_0", nn.initializers.zeros, (self.num_heads, 1, 4 * self.head_dim), self.param_dtype
        )
        self.W2 = self.param(
            "ttt_dense_1",
            nn.initializers.normal(self.config.initializer_range),
            (self.num_heads, 4 * self.head_dim, self.head_dim),
            self.param_dtype,
        )
        self.b2 = self.param("ttt_bias_1", nn.initializers.zeros, (self.num_heads, 1, self.head_dim), self.param_dtype)
        self.ttt_params = (self.W1, self.W2, self.b1, self.b2)

    def process_mini_batch(
        self,
        XQ_mini_batch,
        XK_mini_batch,
        XV_mini_batch,
        eta_mini_batch,
        ttt_params_init,
        ttt_params_mini_batch_init,
        ttt_norm_params,
    ):

        W1_init, W2_init, b1_init, b2_init = ttt_params_mini_batch_init
        square_eta_mini_batch = eta_mini_batch[: self.mini_batch_size]
        last_eta_in_mini_batch = eta_mini_batch[-1][:, None]

        X1 = XK_mini_batch
        Z1 = X1 @ W1_init + b1_init
        X2 = nn.gelu(Z1)
        Z2 = X2 @ W2_init + b2_init
        ttt_norm_out, ttt_norm_vjp = jax.vjp(lambda z: self.ttt_norm.apply({"params": ttt_norm_params}, z), Z2)

        ssl_target = XV_mini_batch - X1
        grad_l_wrt_ttt_norm_out = ttt_norm_out - ssl_target
        grad_l_wrt_Z2 = ttt_norm_vjp(grad_l_wrt_ttt_norm_out)[0]
        grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(1, 0) * diff_gelu(Z1)

        if self.config.output_ttt_stats:
            ttt_loss_mse_step_0 = (grad_l_wrt_ttt_norm_out[-1] ** 2).mean()
        else:
            ttt_loss_mse_step_0 = None

        # Calculate ttt loss using W_init of the entire sequence
        if self.config.output_ttt_stats:
            W1_0, W2_0, b1_0, b2_0 = ttt_params_init
            Z1_0 = X1 @ W1_0 + b1_0
            X2_0 = nn.gelu(Z1_0)
            Z2_0 = X2_0 @ W2_0 + b2_0
            ttt_norm_out_0 = self.ttt_norm.apply({"params": ttt_norm_params}, Z2_0)
            ttt_loss_mse_init = ((ttt_norm_out_0 - ssl_target)[-1] ** 2).mean()
        else:
            ttt_loss_mse_init = None

        X1_bar = XQ_mini_batch
        if self.use_ar_mask:
            Attn1 = jnp.tril(X1_bar @ X1.transpose(1, 0))
            b1_bar = b1_init - (square_eta_mini_batch * jnp.tril(jnp.ones_like(Attn1))) @ grad_l_wrt_Z1
        else:
            Attn1 = X1_bar @ X1.transpose(1, 0)
            b1_bar = b1_init - square_eta_mini_batch @ grad_l_wrt_Z1
        Z1_bar = X1_bar @ W1_init - (square_eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar

        X2_bar = nn.gelu(Z1_bar)
        if self.use_ar_mask:
            Attn2 = jnp.tril(X2_bar @ X2.transpose(1, 0))
            b2_bar = b2_init - (square_eta_mini_batch * jnp.tril(jnp.ones_like(Attn2))) @ grad_l_wrt_Z2
        else:
            Attn2 = X2_bar @ X2.transpose(1, 0)
            b2_bar = b2_init - square_eta_mini_batch @ grad_l_wrt_Z2
        Z2_bar = X2_bar @ W2_init - (square_eta_mini_batch * Attn2) @ grad_l_wrt_Z2 + b2_bar
        ttt_norm_out_bar = self.ttt_norm.apply({"params": ttt_norm_params}, Z2_bar)

        output_mini_batch = X1_bar + ttt_norm_out_bar

        W1_bar_last = W1_init - (last_eta_in_mini_batch * X1).transpose(1, 0) @ grad_l_wrt_Z1
        W2_bar_last = W2_init - (last_eta_in_mini_batch * X2).transpose(1, 0) @ grad_l_wrt_Z2
        b1_bar_last = b1_init - jnp.sum(last_eta_in_mini_batch * grad_l_wrt_Z1, axis=0, keepdims=True)
        b2_bar_last = b2_init - jnp.sum(last_eta_in_mini_batch * grad_l_wrt_Z2, axis=0, keepdims=True)

        if self.config.output_ttt_stats:
            X1_last_fwd_new = nn.gelu((X1[-1:] @ W1_bar_last) + b1_bar_last) @ W2_bar_last + b2_bar_last
            X1_last_fwd_new = self.ttt_norm.apply({"params": ttt_norm_params}, X1_last_fwd_new)
            ttt_loss_mse_step_1 = ((X1_last_fwd_new - ssl_target[-1:]) ** 2).mean()
        else:
            ttt_loss_mse_step_1 = None

        ttt_params_mini_batch_new = (W1_bar_last, W2_bar_last, b1_bar_last, b2_bar_last)

        return (
            ttt_params_mini_batch_new,
            (output_mini_batch, ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1),
        )


class TTTMLP(TTTMLPBase):
    def setup(self):
        super().setup()
        self.wg = nn.Dense(
            self.width,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

    def setup_qkvo(self):
        self.wq = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        if self.config.remat_conv != "":
            conv_module = nn_partitioning.remat(
                nn.Conv, policy=get_gradient_checkpoint_policy(self.config.remat_conv), prevent_cse=True
            )
        else:
            conv_module = nn.Conv
        self.conv_q = conv_module(
            self.config.hidden_size,
            (self.config.conv_width,),
            padding="CAUSAL",
            feature_group_count=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.conv_k = conv_module(
            self.config.hidden_size,
            (self.config.conv_width,),
            padding="CAUSAL",
            feature_group_count=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.wv = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wo = nn.Dense(
            self.width,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

    def get_qkv_projections(self, batch):
        xqk, XV = self.wq(batch), self.wv(batch)
        XQ = self.conv_q(xqk)
        XK = self.conv_k(xqk)
        return XQ, XK, XV

    def apply_gate(self, hidden_states, ttt_output):
        y = self.wg(hidden_states)
        y = nn.gelu(y)
        output = y * ttt_output
        return output
    
 
class TTTLMAttention(nn.Module):
    """Interface with DiT attention layer."""
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_drop: float = 0.
    proj_drop: float = 0.
    mttt_type: str = "linear"
    mttt_kwargs: Any = None

    def setup(self):
        self.mttt_kwargs.hidden_size = self.dim
        self.mttt_kwargs.num_attention_heads = self.num_heads
        match self.mttt_type:
            case "linear":
                ttt_class = TTTLinear
            case "linear_base":
                ttt_class = TTTLinearBase
            case "mlp":
                ttt_class = TTTMLP
            case "mlp_base":
                ttt_class = TTTMLPBase
            case _:
                raise NotImplementedError(
                    f"TTT type {self.mttt_type} not implemented.")
        self.ttt = ttt_class(
            width=self.dim, num_heads=self.num_heads, 
            config=self.mttt_kwargs, dtype=jnp.float32)
    
    def __call__(self, x, training: bool, return_aux: bool = False):
        x, aux = self.ttt(x, position_ids=None, deterministic=not training)
        if return_aux:
            return x, aux
        return x
    
    
class TTTLMBiDirAttention(nn.Module):
    """Interface with DiT attention layer."""
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_drop: float = 0.
    proj_drop: float = 0.
    mttt_type: str = "linear"
    mttt_kwargs: Any = None
    separate_qkv: bool = False
    grad_checkpoint_qkv: bool = False
    grad_checkpoint_out: bool = False
    ttt_out_norm: bool = False # whether add normalization after ttt (bf mul)
    proj_norm: bool = False # whether add normalization before projection
    mult_norm: bool = True  # whether add normalization before multiplication
    apply_gate: bool = True # whether apply gating after ttt
    learn_token_idx: bool = True # whether learn the token idx
    sigmoid_learnable_token_idx: bool = False # whether apply sigmoid to learnable token idx
    use_ar_mask: bool = True # whether AR within mini-batch
    
    # Mixed precision
    qkv_dtype: str = "float32"
    qkv_ptype: str = "float32"
    out_dtype: str = "float32"
    out_ptype: str = "float32"

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.mttt_kwargs.hidden_size = self.dim
        self.mttt_kwargs.num_attention_heads = self.num_heads
        match self.mttt_type:
            case "linear":
                ttt_class = TTTLinear
            case "linear_base":
                ttt_class = TTTLinearBase
            case "mlp":
                ttt_class = TTTMLP
            case "mlp_base":
                ttt_class = TTTMLPBase
            case _:
                raise NotImplementedError(
                    f"TTT type {self.mttt_type} not implemented.")
       
        common_args = {
            "width": self.dim, "num_heads": self.num_heads, 
            "config": self.mttt_kwargs, "dtype": jnp.float32, 
            "set_up_qkvo": False,
            "sigmoid_learnable_token_idx": self.sigmoid_learnable_token_idx,
            "learn_token_idx": self.learn_token_idx,
            "use_ar_mask": self.use_ar_mask
        } 
        self.ttt_forward = ttt_class(**common_args)
        self.ttt_backward = ttt_class(**common_args)
      
        qkv_dense = nn.Dense 
        if self.grad_checkpoint_qkv: 
            qkv_dense = nn.remat(
                nn.Dense, policy=jax.checkpoint_policies.nothing_saveable)
        self.qkv = qkv_dense(
            self.dim * (6 if self.separate_qkv else 3), 
            use_bias=self.qkv_bias,
            # kernel_init=nn.initializers.xavier_uniform(),
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), 
                ("qkv/inp", "qkv/out")),
            bias_init=nn.initializers.zeros_init(),
            # bias_init=nn.with_logical_partitioning(
            #     nn.initializers.zeros_init(), ("qkv/bias")),
            dtype=precision_str_to_type(self.qkv_dtype),
            param_dtype=precision_str_to_type(self.qkv_ptype),
        )
        
        out_dense = nn.Dense
        if self.grad_checkpoint_out:
            out_dense = nn.remat(
                nn.Dense, policy=jax.checkpoint_policies.nothing_saveable)
        self.proj = out_dense(
            self.dim,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), ("proj/inp", "proj/out")),
            bias_init=nn.initializers.zeros_init(),
            dtype=precision_str_to_type(self.out_dtype),
            param_dtype=precision_str_to_type(self.out_ptype),
        )
        if self.proj_drop > 0.:
            self.proj_dropout = nn.Dropout(self.proj_drop) 
        else:
            self.proj_dropout = lambda x, **kwargs: x 
        if self.proj_norm:
            self.proj_ln = nn.LayerNorm()
        else:
            self.proj_ln = lambda x: x
        
    def get_ttt_inputs(self, ttt, batch, XQ, XK, XV, position_ids):
        B, N, F = batch.shape
        n_mini_batch = N // ttt.mini_batch_size
        X = batch.reshape(B, *ttt.seq_shape, ttt.width)

        if ttt.config.output_ttt_stats:
            XV_last_in_mini_batch = XV[:, :: ttt.mini_batch_size, ...].reshape(
                B, n_mini_batch, ttt.num_heads, ttt.head_dim
            )
            XK_last_in_mini_batch = XK[:, :: ttt.mini_batch_size, ...].reshape(
                B, n_mini_batch, ttt.num_heads, ttt.head_dim
            )
            ssl_tgt_last_in_mini_batch = XV_last_in_mini_batch - XK_last_in_mini_batch
            ssl_tgt_mean = (XV - XK).mean(axis=1, keepdims=True).reshape(
                B, 1, ttt.num_heads, ttt.head_dim)
            ssl_tgt_last_in_mini_batch_from_mean_mse = (
                (ssl_tgt_last_in_mini_batch - ssl_tgt_mean) ** 2).mean(
                axis=(0, 2, 3)
            )
        else:
            ssl_tgt_last_in_mini_batch_from_mean_mse = None

        XQ = nn.with_logical_constraint(XQ, ("B", "N", "D"))
        XK = nn.with_logical_constraint(XK, ("B", "N", "D"))
        XV = nn.with_logical_constraint(XV, ("B", "N", "D"))

        XQ = ttt._split_heads(XQ)
        XK = ttt._split_heads(XK)
        XV = ttt._split_heads(XV)
        
        if position_ids is not None:
            freqs_cis = jnp.take(
                ttt.freqs_cis, position_ids % ttt.mini_batch_size, axis=0)
            XQ, XK = apply_rotary_emb(XQ, XK, freqs_cis=freqs_cis, dtype=ttt.dtype)

        XQ = ttt._split_mini_batches(XQ)
        XK = ttt._split_mini_batches(XK)
        XV = ttt._split_mini_batches(XV)

        eta = ttt.get_eta(X)

        return (XQ, XK, XV, eta, {
            "ssl_tgt_last_in_mini_batch_from_mean_mse": ssl_tgt_last_in_mini_batch_from_mean_mse,
            "eta": eta.mean(),
            "eta_per_step": eta.mean(axis=(0, 1, -2, -1)),
        })

    def ttt(self, ttt_layer, x, q, k, v, pos_ids):
        q_, k_, v_, eta, inp_stat = self.get_ttt_inputs(
            ttt_layer, x, q, k, v, position_ids=pos_ids) 
        z, ttt_aux = ttt_layer.ttt(q_, k_, v_, eta)
        if self.mult_norm:
            z = ttt_layer.post_norm(z)
        if self.ttt_out_norm:
            z = ttt_layer.apply_gate(x, z)
        return z, {
            "input_stats": inp_stat,
            "output_stats": ttt_aux
        }
    
    def __call__(self, x, training: bool, return_aux: bool = False):
        """[x] B, N, C"""
        B, N, C = x.shape
        
        # Transform K, Q, V
        x = x.astype(precision_str_to_type(self.qkv_dtype)) 
        qkv = self.qkv(x)
        qkv = qkv.astype(jnp.float32) # the output of qkv will cast to flaot32
        if self.separate_qkv:
            qkv = qkv.reshape(B, N, 6, -1)
            # shape=(B, N, 6, dim) -> (6, B, N, dim) -> 6 x (B, N, dim)
            q_f, k_f, v_f, q_b, k_b, v_b = qkv.transpose((2, 0, 1, 3)) 
        else:
            qkv = qkv.reshape(B, N, 3, -1)
            # shape=(B, N, 3, dim) -> (3, B, N, dim) -> (B, N, dim)
            q_f, k_f, v_f = qkv.transpose((2, 0, 1, 3)) 
            q_b, k_b, v_b = q_f, k_f, v_f
         
        # Forward 
        position_ids = jnp.arange(N)[None, :]
        x_f, aux_f = self.ttt(self.ttt_forward, x, q_f, k_f, v_f, position_ids)
        
        # Backward 
        x_flip = jnp.flip(x, axis=-2)
        q_flip = jnp.flip(q_b, axis=-2)
        k_flip = jnp.flip(k_b, axis=-2)
        v_flip = jnp.flip(v_b, axis=-2)
        # [TODO] position_ids should we flip?
        x_b, aux_b = self.ttt(
            self.ttt_backward, x_flip, q_flip, k_flip, v_flip, position_ids)
        
        # Combine forward and backward 
        x = jnp.multiply(x_f, jnp.flip(x_b, axis=-2))
        
        # Output layer 
        x = self.proj_ln(x)
        x = x.astype(precision_str_to_type(self.out_dtype))
        x = self.proj(x)
        x = self.proj_dropout(x, deterministic=not training)
        x = x.astype(jnp.float32) # next layer will take float32
        if return_aux:
            return x, {"forward": aux_f, "backward": aux_b}
        return x
    
    
class TTTLMBiDirAttentionV2(TTTLMBiDirAttention):
    """Interface with DiT attention layer."""

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.mttt_kwargs.hidden_size = self.dim
        self.mttt_kwargs.num_attention_heads = self.num_heads
        match self.mttt_type:
            case "linear":
                ttt_class = TTTLinear
            case "linear_base":
                ttt_class = TTTLinearBase
            case "mlp":
                ttt_class = TTTMLP
            case "mlp_base":
                ttt_class = TTTMLPBase
            case _:
                raise NotImplementedError(
                    f"TTT type {self.mttt_type} not implemented.")
       
        common_args = {
            "width": self.dim * 2, "num_heads": self.num_heads * 2, 
            "config": self.mttt_kwargs, "dtype": jnp.float32, 
            "set_up_qkvo": False,
            "sigmoid_learnable_token_idx": self.sigmoid_learnable_token_idx,
            "learn_token_idx": self.learn_token_idx,
            "use_ar_mask": self.use_ar_mask
        } 
        self.ttt_layer = ttt_class(**common_args)
      
        qkv_dense = nn.Dense 
        if self.grad_checkpoint_qkv: 
            qkv_dense = nn.remat(
                nn.Dense, policy=jax.checkpoint_policies.nothing_saveable)
        self.qkv = qkv_dense(
            self.dim * (6 if self.separate_qkv else 3), 
            use_bias=self.qkv_bias,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), 
                ("qkv/inp", "qkv/out")),
            bias_init=nn.initializers.zeros_init(),
            dtype=precision_str_to_type(self.qkv_dtype),
            param_dtype=precision_str_to_type(self.qkv_ptype),
        )
        
        out_dense = nn.Dense
        if self.grad_checkpoint_out:
            out_dense = nn.remat(
                nn.Dense, policy=jax.checkpoint_policies.nothing_saveable)
        self.proj = out_dense(
            self.dim,
            # kernel_init=nn.initializers.xavier_uniform(),
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), ("proj/inp", "proj/out")),
            bias_init=nn.initializers.zeros_init(),
            dtype=precision_str_to_type(self.out_dtype),
            param_dtype=precision_str_to_type(self.out_ptype),
        )
        if self.proj_drop > 0.:
            self.proj_dropout = nn.Dropout(self.proj_drop) 
        else:
            self.proj_dropout = lambda x, **kwargs: x 
        if self.proj_norm:
            self.proj_ln = nn.LayerNorm()
        else:
            self.proj_ln = lambda x: x
        
    def __call__(self, x, training: bool, return_aux: bool = False):
        """[x] B, N, C"""
        B, N, C = x.shape
        
        # Transform K, Q, V
        x = x.astype(precision_str_to_type(self.qkv_dtype)) 
        qkv = self.qkv(x)
        qkv = qkv.astype(jnp.float32) # the output of qkv will cast to flaot32
        if self.separate_qkv:
            qkv = qkv.reshape(B, N, 6, -1)
            # shape=(B, N, 6, dim) -> (6, B, N, dim) -> 6 x (B, N, dim)
            q_f, k_f, v_f, q_b, k_b, v_b = qkv.transpose((2, 0, 1, 3)) 
        else:
            qkv = qkv.reshape(B, N, 3, -1)
            # shape=(B, N, 3, dim) -> (3, B, N, dim) -> (B, N, dim)
            q_f, k_f, v_f = qkv.transpose((2, 0, 1, 3)) 
            q_b, k_b, v_b = q_f, k_f, v_f
         
        position_ids = jnp.arange(N)[None, :]
        x_flip = jnp.flip(x, axis=-2)
        xx = jnp.concat([x, x_flip], axis=-1)
        
        q_flip = jnp.flip(q_b, axis=-2)
        qq = jnp.concat([q_f, q_flip], axis=-1)
        
        k_flip = jnp.flip(k_b, axis=-2)
        kk = jnp.concat([k_f, k_flip], axis=-1)
        
        v_flip = jnp.flip(v_b, axis=-2)
        vv = jnp.concat([v_f, v_flip], axis=-1)
        
        x, aux = self.ttt(self.ttt_layer, xx, qq, kk, vv, position_ids)
        
        # Combine forward and backward 
        xf, xb = jnp.split(x, 2, axis=-1)
        x = jnp.multiply(xf, jnp.flip(xb, axis=-2))
        
        # Output layer 
        x = self.proj_ln(x)
        x = x.astype(precision_str_to_type(self.out_dtype))
        x = self.proj(x)
        x = self.proj_dropout(x, deterministic=not training)
        x = x.astype(jnp.float32) # next layer will take float32
        if return_aux:
            return x, {"forward": aux, "backward": aux}
        return x
    
    
if __name__ == "__main__":
    import lovely_jax
    from time import time
    lovely_jax.monkey_patch()
    from omegaconf import OmegaConf
    
    rng = jax.random.PRNGKey(1)
    D = 6 * 64
    num_heads = 6
    head_dim = D // num_heads
    B, N = 2, (256//2)**2
    ttt_layer = TTTLMAttention(
        dim=D, num_heads=num_heads, 
        mttt_type="linear", 
        mttt_kwargs=OmegaConf.create({
            "hidden_size": None,                
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/model.py#L76 
            "conv_width": 4,                    
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/model.py#L64C9-L64C35 
            "initializer_range": 0.02,
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/model.py#L72C9-L72C31
            "mini_batch_size": 16,
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/model.py#L63
            "max_sequence_length": N, 
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/model.py#L74C9-L74C31
            "rope_theta": 10000.0,
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/model.py#L69C9-L69C28
            "ttt_base_lr": 1.0,
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/model.py#L73C1-L74C1
            "remat_mini_batch_group_size": 4,
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/model.py#L213
            "remat_conv": "",
            "output_ttt_stats": True,      
      })
    )
    print(ttt_layer)
    ttt_params = ttt_layer.init(rng, jnp.ones([B, N, D]), True)
    print("TTT params:", ttt_params)
    ttt_out, _ = ttt_layer.apply(ttt_params, jnp.ones([B, N, D]), True)
    print(ttt_out)