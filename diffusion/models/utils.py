import jax.numpy as jnp
import flax.linen as nn

def precision_str_to_type(precision_str: str):
    match precision_str:
        case "bfloat16":
            return jnp.bfloat16
        case "float16":
            return jnp.float16
        case "float32":
            return jnp.float32
        case "float64":
            return jnp.float64
        case _:
            raise ValueError
        

class Identity(nn.Module):
    def __call__(self, x, *args, **kwargs):
        return x