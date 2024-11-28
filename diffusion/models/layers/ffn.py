import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from typing import Optional, Any


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks 
        NOTE: I change the initialization to have DiT init 
    """
    in_features: int
    hidden_features: Optional[int]  = None
    out_features: Optional[int]     = None
    act_layer: str                  = "gelu"
    bias: bool                      = True
    drop: float                     = 0.
   
    # Training precision 
    dtype: Any = jnp.float32
    ptype: Any = jnp.float32 
    
    def setup(self):
        self.act = {
            "gelu": partial(nn.gelu, approximate=False),
            "gelu_approx": partial(nn.gelu, approximate=True),
        }[self.act_layer]
        
        hidden_features = self.hidden_features or self.in_features
        self.fc1 = nn.Dense(
            hidden_features, use_bias=self.bias,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(),
                ("mlp/feat", "mlp/hidden")
            ),
            bias_init=nn.initializers.zeros_init(),
            param_dtype=self.ptype,
            dtype=self.dtype
        )
        self.drop1 = nn.Dropout(self.drop)
        
        out_features = self.out_features or self.in_features
        self.fc2 = nn.Dense(
            out_features, use_bias=self.bias,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(),
                ("mlp/hidden", "mlp/feat")
            ),
            bias_init=nn.initializers.zeros_init(),
            param_dtype=self.ptype,
            dtype=self.dtype
        )
        self.drop2 = nn.Dropout(self.drop)

    def __call__(self, x, training: bool):
        # Input cast to dtype
        x = nn.with_logical_constraint(x, ("B", "N", "D"))
        x = x.astype(self.dtype)
        x = self.fc1(x)
        # Activation in float32
        x = x.astype(jnp.float32)
        x = self.act(x)
        x = x.astype(self.dtype)
        x = self.drop1(x, deterministic=not training)
        x = self.fc2(x)
        x = self.drop2(x, deterministic=not training)
        # Output in float32
        x = x.astype(jnp.float32)
        x = nn.with_logical_constraint(x, ("B", "N", "D"))
        return x
    
   
if __name__ == "__main__":
    # Test for mixed precision
    import lovely_jax as lj
    lj.monkey_patch()

    B = 32
    N = 16384
    D = 1024
    mlp = Mlp(
        D, 4 * D, D, act_layer="gelu", drop=0.0, 
        dtype=jnp.bfloat16, # B16, 8 its/s, 49GB A6000 GPU, OOM at B=64
        # dtype=jnp.float32,  # B16, 4 its/s, 49GB A6000 GPU, OOM at B=32
        ptype=jnp.float32)
    x = jnp.ones((B, N, D)) 
    params = mlp.init(jax.random.PRNGKey(0), x, training=True)
    print(params["params"]["fc1"]["kernel"].dtype)
    
    def loss(params, rng):
        rng1, rng2 = jax.random.split(rng)
        x = jax.random.uniform(rng1, (B, N, D), dtype=jnp.float32)
        # out = mlp.apply(params, x, training=True, rngs={"dropout": rng2})
        out = mlp.apply(params, x, training=True)
        return jnp.square(out - 1).mean()
    
    @jax.jit 
    def step(params, rng, lr=1e-4):
        loss_val, grads = jax.value_and_grad(loss)(params, rng)
        params = jax.tree_util.tree_map(
            lambda p, g: p - lr * g, params, grads)
        return params, loss_val
    
    # Jit the step function
    rng = jax.random.PRNGKey(0)
    spl, rng = jax.random.split(rng)
    params, loss_val = step(params, spl) 
    print(loss_val)
    
    # Timing
    import time
    import tqdm
    start = time.time()
    total_steps = 100
    pbar = tqdm.tqdm(range(total_steps))
    for _ in pbar:
        spl, rng = jax.random.split(rng)
        params, loss_val = step(params, spl)
        pbar.set_description(f"Loss: {loss_val:.4f}") 
    duration = (time.time() - start) / float(total_steps)
    print("Time", duration)
    print(params["params"]["fc1"]["kernel"].dtype)