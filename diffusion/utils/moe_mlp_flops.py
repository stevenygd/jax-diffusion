from diffusion.models.layers.moe_mlp import SwitchFeedForward
from diffusion.utils.flops_utils import *

def switch_ff_flops(x_shape, layer: SwitchFeedForward, backward=False, unit=1):
    assert isinstance(layer, SwitchFeedForward), f"{type(layer)}"
    flops = 0
    original_shape = x_shape

    # x = x.reshape(-1, self.hidden_size)
    x_shape = jnp.array([jnp.prod(x_shape)/layer.hidden_size, layer.hidden_size]).astype(jnp.int32)
    num_tokens, hidden_size = x_shape
    n_experts = layer.n_experts

    # gate_logits = self.router(x)
    gate_logits_shape, flops_ = dense_flops(
        x_shape, layer.router, backward=backward, unit=unit)
    flops += flops_

    # route_prob = jax.nn.softmax(gate_logits, axis=-1)
    _, flops_ = softmax_flops(gate_logits_shape, backward=backward, unit=unit)
    flops += flops_

    # expert_counts = jnp.sum(expert_mask, axis=0)
    expert_mask_shape = jnp.array([num_tokens, n_experts])
    flops += jnp.prod(expert_mask_shape) *(1 if backward else 3) / unit

    # expert_inputs = jnp.einsum('te,td->etd', expert_mask, x)
    expert_inputs_shape = jnp.array([n_experts, num_tokens, hidden_size])
    flops += jnp.prod(expert_inputs_shape) * 2 *(1 if backward else 3) / unit

    # Apply experts
    expert_input_shape = jnp.array([expert_inputs_shape[1], expert_inputs_shape[2]])
    # for expert in layer.experts:
    _, flops_ = mlp_flops(
        expert_input_shape, layer.experts[0], backward=backward, unit=unit)
    flops += flops_

    # Combine expert outputs
    # combined_outputs = jnp.einsum('te,etd->td', expert_mask, expert_outputs)
    expert_outputs_shape = expert_inputs_shape
    combined_outputs_shape = jnp.array([num_tokens, hidden_size])
    flops += jnp.prod(expert_outputs_shape) * 2 *(1 if backward else 3) / unit
    return original_shape, flops

# def moe_mlp_flops(x_shape, layer: MoEMlp, backward=False, unit=1):
#     assert isinstance(layer, MoEMlp), f"{type(layer)}"
#     flops = 0

#     # x = self.fc1(x)
#     x_shape, flops_ = dense_flops(
#         x_shape, layer.fc1, backward=backward, unit=unit)
#     flops += flops_

#     # activation
#     flops += jnp.prod(x_shape) *(1 if backward else 3) / unit

#     # x = self.fc2(x)
#     x_shape, flops_ = dense_flops(
#         x_shape, layer.fc2, backward=backward, unit=unit)
#     flops += flops_

#     return x_shape, flops