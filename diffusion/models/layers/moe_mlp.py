import jax
import jax.numpy as jnp
from jax import lax
from jax import random
from flax import linen as nn
from functools import partial
from typing import Optional, Callable, Any
from diffusion.models.layers.ffn import Mlp

# class MoEMlp(Mlp):
# class MoEMlp(nn.Module):
    # def __call__(self, x, training: bool):
        # return super().__call__(x, training)
    
    # """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    # in_features: int
    # hidden_features: Optional[int] = None
    # out_features: Optional[int] = None
    # act_layer: str = "gelu"
    # bias: bool = True
    # drop: float = 0.

    # def setup(self):
    #     hidden_features = self.hidden_features or self.in_features
    #     out_features = self.out_features or self.in_features

    #     self.fc1 = nn.Dense(
    #         hidden_features, 
    #         use_bias=self.bias,
    #         kernel_init=nn.initializers.xavier_uniform(),
    #         bias_init=nn.initializers.zeros
    #     )
    #     self.fc2 = nn.Dense(
    #         out_features, 
    #         use_bias=self.bias,
    #         kernel_init=nn.initializers.xavier_uniform(),
    #         bias_init=nn.initializers.zeros
    #     )
    #     
    #     self.act = {
    #         "gelu": nn.gelu,
    #         "gelu_approx": partial(nn.gelu, approximate=True),
    #     }[self.act_layer]
    #     
    #     self.dropout = nn.Dropout(self.drop)

    # def __call__(self, x, deterministic: bool = False):
    #     x = self.fc1(x)
    #     x = self.act(x)
    #     x = self.dropout(x, deterministic=deterministic)
    #     x = self.fc2(x)
    #     x = self.dropout(x, deterministic=deterministic)
    #     return x

class SwitchFeedForward(nn.Module):
    hidden_size: int
    intermediate_size: int
    n_experts: int
    capacity_factor: float = 1.0
    drop_tokens: bool = True
    is_scale_prob: bool = True
    act_layer: str = "gelu"
    bias: bool = True
    drop: float = 0.

    # Mixed precision
    dtype: Any = jnp.float32
    ptype: Any = jnp.float32

    def setup(self):
        # Initialize the routing layer
        # Router: Linear projection to compute logits for expert assignment
        self.router = nn.Dense(
            features=self.n_experts,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=1.0 / self.hidden_size**0.5)
        )
        
        # using vmapped Mlp to create n_experts copies of the same MLP
        self.experts = [Mlp(
            in_features=self.hidden_size,
            hidden_features=self.intermediate_size,
            out_features=self.hidden_size,
            act_layer=self.act_layer,
            bias=self.bias,
            drop=self.drop,
            ptype=self.ptype, dtype=self.dtype
        ) for _ in range(self.n_experts)]

    # def __call__(self, x: jnp.ndarray, deterministic: bool = False, rng: Optional[jax.random.PRNGKey] = None):
    def __call__(self, x: jnp.ndarray, training: bool = False, rng: Optional[jax.random.PRNGKey] = None):
        deterministic = not training
        original_shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        num_tokens = x.shape[0]

        # Compute routing probabilities
        # gate_logits shape: (num_tokens, n_experts)
        # Compute gate logits: h(x) = Wx + b
        gate_logits = self.router(x)

        # route_prob shape: (num_tokens, n_experts)
        # p_i(x) = exp(h(x)_i) / sum_j(exp(h(x)_j))        
        route_prob = jax.nn.softmax(gate_logits, axis=-1)

        # route_prob_max shape: (num_tokens,)
        # Get the maximum routing probability and the corresponding expert for each token
        route_prob_max = jnp.max(route_prob, axis=-1)

        # routes shape: (num_tokens,)
        routes = jnp.argmax(route_prob, axis=-1)

        # Calculate expert capacity: tokens per expert * capacity_factor
        # Ensures load balancing by limiting the number of tokens per expert
        capacity = jnp.floor(self.capacity_factor * num_tokens / self.n_experts).astype(jnp.int32)

        expert_mask = jax.nn.one_hot(routes, self.n_experts)

        # Token dropping logic
        if self.drop_tokens and not deterministic and rng is not None:
            # Split the RNG for token dropping and potential dropout use
            token_rng, dropout_rng = jax.random.split(rng)
            # Create separate RNGs for each expert
            token_rngs = jax.random.split(token_rng, self.n_experts)

            for i in range(self.n_experts):
                # Get the mask for the current expert
                mask = expert_mask[:, i]
                # Count how many tokens are assigned to this expert
                expert_count = jnp.sum(mask).astype(jnp.int32)

                # If the expert is over capacity, we need to drop some tokens
                if expert_count > capacity:
                    # Generate random priorities for all tokens
                    random_priorities = jax.random.uniform(token_rngs[i], shape=(num_tokens,))
                    # Get indices of tokens assigned to this expert
                    expert_indices = jnp.where(mask)[0]
                    # Sort the expert's tokens by their random priorities and select the top 'capacity' tokens
                    prioritized_indices = jnp.argsort(random_priorities[expert_indices])[-capacity:]
                    selected_indices = expert_indices[prioritized_indices]

                    # Create a new mask with only the selected tokens
                    new_mask = jnp.zeros_like(mask)
                    new_mask = new_mask.at[selected_indices].set(1)
                    # Update the expert mask with the new selection
                    expert_mask = expert_mask.at[:, i].set(new_mask)
        else:
            # If we're not dropping tokens, we don't need a dropout RNG
            dropout_rng = None

        # Convert the expert mask to float for further computations
        expert_mask = expert_mask.astype(jnp.float32)
        # Count how many tokens are assigned to each expert after potential dropping
        expert_counts = jnp.sum(expert_mask, axis=0)

        # Compute inputs for each expert
        # select the tokens for each expert and arranges them
        # into a (n_experts, tokens_per_expert, hidden_size) tensor
        expert_inputs = jnp.einsum('te,td->etd', expert_mask, x)

        # Apply experts
        # expert_outputs = [expert(expert_inputs[i], deterministic=deterministic) for i, expert in enumerate(self.experts)]
        expert_outputs = [expert(expert_inputs[i], training=training) for i, expert in enumerate(self.experts)]
        expert_outputs = jnp.stack(expert_outputs)

        # Combine expert outputs
        combined_outputs = jnp.einsum('te,etd->td', expert_mask, expert_outputs)

        if self.is_scale_prob:
            final_output = combined_outputs * route_prob_max[:, None]
        else:
            final_output = combined_outputs * (route_prob_max / jax.lax.stop_gradient(route_prob_max))[:, None]

        final_output = final_output.reshape(original_shape)
        gate_logits = gate_logits.reshape(original_shape[:-1] + (-1,))

        return final_output, expert_counts, jnp.sum(route_prob, axis=0), gate_logits
    

def compute_switch_loss(moe_info: dict,
                        n_experts: int,
                        use_z_loss: bool = False) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """
    Compute the switch loss for Mixture of Experts.
    
    Args:
    moe_info: A dictionary containing:
        - 'counts': jnp.ndarray of shape (n_experts,)
        - 'route_prob_sums': jnp.ndarray of shape (n_experts,)
        - 'gate_logits': jnp.ndarray of shape (batch_size, seq_len, n_experts)
    n_experts: int, number of experts
    use_z_loss: bool, whether to compute z_loss
    
    Returns:
    tuple of (load_balancing_loss, z_loss)
    """
    counts = moe_info['counts']
    route_prob_sums = moe_info['route_prob_sums']
    gate_logits = moe_info['gate_logits']
    
    # Compute load balancing loss
    total_tokens = jnp.sum(counts)
    fraction_expert_capacity = counts / total_tokens
    route_fraction = route_prob_sums / total_tokens
    
    load_balancing_loss = n_experts * jnp.sum(fraction_expert_capacity * route_fraction)  

    # Compute z_loss if required
    z_loss = None
    if use_z_loss:
        z_loss = jnp.mean(jnp.square(jax.nn.logsumexp(gate_logits, axis=-1)))

    return load_balancing_loss, z_loss
