"""
This code is adapted from
https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py
"""

from typing import Optional, Any 
import jax
import numpy as np
import flax.linen as nn
import jax.ad_checkpoint
import jax.numpy as jnp
from dataclasses import field
from diffusion.models.layers.ffn import Mlp
from diffusion.models.layers.attention import (
    JaxAttention, LinearAttention, S4DAttention, TTTAttention)
from diffusion.models.layers.patch_embed import (
    PatchEmbed, UnPatchify,
    SimpleDiffusionEmbedder, SimpleDiffusionUnpacify
)
from diffusion.models.layers.bpt import blockwise_ffn 
from diffusion.models.utils import precision_str_to_type
from jax.sharding import PartitionSpec as PS
from diffusion.utils.sharding import with_sharding_constraint
from diffusion.models.posenc import get_2d_sincos_pos_embed


################################################################################
#                                 Core DiT Model                               #
################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale[:, None, ...]) + shift[:, None, ...]


class DiTBlock(nn.Module):
    """ A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning. """
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    # rnn_dim: int = 128
    # ssm_expansion: int = 2
    attn_kwargs: Optional[dict] = field(default_factory=dict)
    attn_type: str = "torch"
    n_experts: int = 8 
    grad_checkpoint_attn: bool = False
    grad_checkpoint_mlp: bool = False
    grad_checkpoint_adamlp: bool = False
    
    # BPT MLP parameters
    blockwise_mlp: bool = False # whether use blockwise remat for mlp
    blockwise_mlp_chunk_size: int = 2048 # default from bpt
    
    # Mixed Precision training
    mlp_dtype: str = "float32"
    mlp_ptype: str = "float32"
    adaln_mlp_dtype: str = "float32"
    adaln_mlp_ptype: str = "float32"
    
    def setup(self):
        match self.attn_type:
            case "jax" :
                attn_class = JaxAttention
            case "linear":
                attn_class = LinearAttention
            case "ssm":
                attn_class = S4DAttention
            case "ttt":
                attn_class = TTTAttention 
            case _:
                raise ValueError
            
        if self.grad_checkpoint_attn:
            attn_class = nn.remat(
                attn_class, static_argnums=(2, 3),
                policy=jax.checkpoint_policies.nothing_saveable
            )
        self.attn = attn_class(
            dim=self.hidden_size,
            num_heads=self.num_heads, 
            qkv_bias=True, **self.attn_kwargs)
       
        self.norm1 = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=1e-6)
        self.norm2 = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=1e-6)
        
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        mlp_class = Mlp
        if self.grad_checkpoint_mlp:
            mlp_class = nn.remat(
                mlp_class, static_argnums=(2,),
                policy=jax.checkpoint_policies.nothing_saveable
            ) 
       
        mlp_dtype = precision_str_to_type(self.mlp_dtype) 
        mlp_ptype = precision_str_to_type(self.mlp_ptype) 
        self.mlp = mlp_class(
            in_features=self.hidden_size, 
            hidden_features=mlp_hidden_dim, 
            act_layer="gelu_approx", drop=0,
            ptype=mlp_ptype, dtype=mlp_dtype
        )
           
        if self.grad_checkpoint_adamlp:
            ada_dense_class = nn.remat(nn.Dense)
        else:
            ada_dense_class = nn.Dense
        adaln_mlp_dtype = precision_str_to_type(self.adaln_mlp_dtype)
        adaln_mlp_ptype = precision_str_to_type(self.adaln_mlp_ptype)
        self.adaLN_mlp = ada_dense_class(
            6 * self.hidden_size, use_bias=True,
            # zero initial AdaLN prediction
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.zeros_init(),
                ("adaln/inp", "adaln/out")
            ),
            bias_init=nn.initializers.zeros_init(),
            param_dtype=adaln_mlp_ptype, dtype=adaln_mlp_dtype)
        
    def adaLN(self, c): 
        c = nn.activation.silu(c)
        c = self.adaLN_mlp(c)
        c = c.astype(jnp.float32) # default to float32
        # Original pytorch code
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = x.chunk(6, dim=1)
        return jnp.array_split(c, 6, axis=-1)
   
    def concatMLP(self, c):
        c = nn.activation.silu(c)
        c = self.concat_mlp(c)
        return jnp.array_split(c, 2, axis=-1)
       
    def __call__(self, x, c, training: bool, return_aux: bool):
        x = nn.with_logical_constraint(x, ("B", "N", "D"))
        c = nn.with_logical_constraint(c, ("B", "D"))

        aux = {}
        inp_norm = jnp.square(x).mean() ** 0.5
        
        # Prepare conditioning type
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c)
        
        # First conditioning
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(x, training, return_aux)
        if return_aux:
            attn_out, attn_aux = attn_out
            aux.update(
                {f"{self.attn.name}.{k}": v for k, v in attn_aux.items()})
        attn_out_norm = jnp.square(attn_out).mean() ** 0.5
        x = x + gate_msa[:, None, ...] * attn_out
        x = nn.with_logical_constraint(x, ("B", "N", "D"))
        
        # Second conditioning
        x= modulate(self.norm2(x), shift_mlp, scale_mlp)
        if self.blockwise_mlp:  
            mlp_out = blockwise_ffn(
                self.mlp, x, chunk_size=self.blockwise_mlp_chunk_size, 
                training=training)
        else:
            mlp_out = self.mlp(x, training)
        x = x + gate_mlp[:, None, ...] * mlp_out
        out_norm = jnp.square(x).mean() ** 0.5
        x = nn.with_logical_constraint(x, ("B", "N", "D"))
        
        aux.update({
            "dit_block": {
                "shift_msa": shift_msa.mean(),
                "scale_msa": scale_msa.mean(),
                "gate_msa": gate_msa.mean(),
                "shift_mlp": shift_mlp.mean(),
                "scale_mlp": scale_mlp.mean(),
                "gate_mlp": gate_mlp.mean(),
                "out_norm": out_norm,
                "inp_norm": inp_norm,
                "attn_out_norm": attn_out_norm,
        }})
        
        if return_aux:
            return x, aux 
        return x
  
 
class DiTBlockScan(DiTBlock):
    """
    Dirty code, trying to match the signature in this block:
    - https://github.com/google-research/big_vision/blob/
      b8dab6e4de3436849415f37c591399c93b1eaf39/big_vision/models/vit.py#L79
    """
    
    def __call__(self, x_c, training: bool, return_aux: bool):
        x, c = x_c
        out = super().__call__(x, c, training, return_aux)
        x, aux = out if return_aux else (out, {})
        out = (x, c)
        return out, aux 
    
    
class DiTBlockScanRemat(nn.Module):
    """Rollout [n_blocks] many of inner DiTBlocks. """
    n_blocks: int 
    block_args: Optional[dict] = field(default_factory=dict)
    
    def setup(self):
        block_fn = nn.remat(
            DiTBlockScan, 
            prevent_cse=False,
            static_argnums=(2, 3), # 0=self, 2=train, 3=return_aux
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.rollout_blocks = nn.scan(
            block_fn,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            metadata_params={nn.PARTITION_NAME: None},
            in_axes=nn.broadcast,
            length=self.n_blocks
        )(**self.block_args)
        
    def __call__(self, x_c, training: bool, return_aux: bool):
        x_c, aux = self.rollout_blocks(x_c, training, return_aux)
        return x_c, aux 


class FinalLayer(nn.Module):
    """ The final layer of DiT. """
    hidden_size: int
    patch_size: int 
    out_channels: int
    
    def setup(self):
        self.norm_final = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=1e-6)
        self.linear = nn.Dense(
            self.patch_size ** 2 * self.out_channels, use_bias=True,
            # DiT use zero init for output layer
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init()
        )
        self.adaLN_mlp = nn.Dense(
            2 * self.hidden_size, use_bias=True,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init()
        )
        
    def adaLN(self, c): 
        c = nn.activation.silu(c)
        c = self.adaLN_mlp(c)
        # Original pytorch code
        return jnp.array_split(c, 2, axis=1)

    def __call__(self, x, c):
        shift, scale = self.adaLN(c)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    hidden_size: int
    frequency_embedding_size : int = 256
    max_period : int = 10000   
    
    """
    Embeds scalar timesteps into vector representations.
    """
    def setup(self):
        self.dense1 = nn.Dense(
            self.hidden_size, use_bias=True,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        self.dense2 = nn.Dense(
            self.hidden_size, use_bias=True,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )

    def timestep_embedding(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = self.frequency_embedding_size // 2
        freqs = jnp.exp(
            -jnp.log(self.max_period) * 
            # jnp.arange(start=0, stop=half, dtype=jnp.float_) / half
            jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
        )
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([
            jnp.cos(args), jnp.sin(args)], axis=-1)
        if self.frequency_embedding_size % 2:
            embedding = jnp.concatenate([
                embedding, 
                jnp.zeros_like(embedding[:, :1])
            ], axis=-1)
        return embedding

    def __call__(self, t):
        t_freq = self.timestep_embedding(t) 
        t_emb = nn.activation.silu(self.dense1(t_freq))
        t_emb = self.dense2(t_emb)
        return t_emb


class LabelEmbedder(nn.Module):
    num_classes: int
    hidden_size: int
    dropout_prob: float
    rng_collection: str = 'label_emb'
    
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def setup(self):
        use_cfg_embedding = self.dropout_prob > 0
        # Torch interface:  #embeddings, #emb_dims
        # Jax interface:    #embeddings, #features)
        self.embedding_table = nn.Embed(
            self.num_classes + use_cfg_embedding, 
            self.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )

    def token_drop(self, rng, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if rng is None:
            rng = self.make_rng(self.rng_collection)
        if force_drop_ids is None:
            # Torch. rand returns uniform [0, 1] with shape labels.shape[0]
            # drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            # This creates a mask of shape labels.shape[0]
            drop_ids = jax.random.uniform(
                rng, shape=(labels.shape[0],)
            ) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels

    def __call__(self, labels, training: bool, force_drop_ids=None, rng=None):
        use_dropout = self.dropout_prob > 0
        if (training and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(rng, labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings



#################################################################################
# https://github.com/huggingface/pytorch-image-models/blob/
# e748805be31318da1a0e34b61294704666f50397/timm/layers/patch_embed.py#L26C1-L110C1
#################################################################################
class Model(nn.Module):
    """ Diffusion model with a Transformer backbone.  Jax implementation. """
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    depth: int = 28
    num_heads:int =16
    mlp_ratio:float =4.0
    class_dropout_prob:float = 0.1
    num_classes:int =1000
    learn_sigma: bool = False 
    package_name: str = "dit" 

    # Attention parameters
    attn_type: str = "torch"
    ssm_expansion: int = 2
    attn_kwargs: Optional[dict] = field(default_factory=dict)
    grad_checkpoint: bool = False
    dit_block_kwargs: Optional[dict] = field(default_factory=dict)
    scan_remat: bool = False
    scan_remat_block_size: int = 1
    
    # Patching and unpatching
    patch_type: str = "vit" # vae | simple_diffusion
    x_emb_kwargs: Optional[dict] = field(default_factory=dict)
    unpat_kwargs: Optional[dict] = field(default_factory=dict)
    
    # Skip connections
    skip_layers: list = field(default_factory=list)
    
    def setup(self):
        self.out_channels = (
            self.in_channels * 2 if self.learn_sigma else self.in_channels)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.y_embedder = LabelEmbedder(
            self.num_classes, self.hidden_size, self.class_dropout_prob)
        
        match self.patch_type: 
            case "vit":
                self.x_embedder = PatchEmbed(
                    patch_size=self.patch_size, embed_dim=self.hidden_size, 
                    bias=True,
                    **self.x_emb_kwargs)
                self.final_layer = FinalLayer(
                    hidden_size=self.hidden_size, patch_size=self.patch_size, 
                    out_channels=self.out_channels)
                self.unpatchify = UnPatchify(
                    self.out_channels, self.patch_size,
                    **self.unpat_kwargs)
            case "simple_diffusion":
                self.x_embedder = SimpleDiffusionEmbedder(
                    patch_size=self.patch_size, embed_dim=self.hidden_size,
                    **self.x_emb_kwargs)
                self.final_layer = lambda x, c: x
                self.unpatchify = SimpleDiffusionUnpacify(
                    self.out_channels, self.patch_size,
                    **self.unpat_kwargs)
            case _:
                raise NotImplemented
                
        
        # PyTorch to Jax version
        # Fixed, not optimizing
        num_patches_per_dim = self.input_size // self.patch_size
        self.pos_emb = get_2d_sincos_pos_embed(
            self.hidden_size, num_patches_per_dim)[None, ...]

        block_args = {
            "hidden_size": self.hidden_size, 
            "num_heads": self.num_heads, 
            "mlp_ratio": self.mlp_ratio,
            "attn_type": self.attn_type,
            "attn_kwargs": self.attn_kwargs,
            **self.dit_block_kwargs
        }
        if self.grad_checkpoint:
            if self.scan_remat:
                remat_length = self.depth // self.scan_remat_block_size 
                block_fn = nn.remat(
                    DiTBlockScanRemat, 
                    prevent_cse=False,
                    static_argnums=(2, 3), # 0=self, 2=train, 3=return_aux
                    policy=jax.checkpoint_policies.nothing_saveable,
                )
                self.blocks = nn.scan(
                    block_fn,
                    variable_axes={"params": 0},
                    split_rngs={"params": True, "dropout": True},
                    in_axes=nn.broadcast,
                    metadata_params={nn.PARTITION_NAME: None},
                    length=remat_length
                )(
                    n_blocks=self.scan_remat_block_size,
                    block_args=block_args
                )
                
            else:
                block_fn = nn.remat(
                    DiTBlockScan, 
                    prevent_cse=False,
                    static_argnums=(2, 3), # 0=self, 2=train, 3=return_aux
                    policy=jax.checkpoint_policies.nothing_saveable,
                )
                self.blocks = nn.scan(
                    block_fn,
                    variable_axes={"params": 0},
                    split_rngs={"params": True, "dropout": True},
                    in_axes=nn.broadcast,
                    metadata_params={nn.PARTITION_NAME: None},
                    length=self.depth
                )(**block_args)
        else:
            self.blocks = nn.scan(
                DiTBlockScan,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                metadata_params={nn.PARTITION_NAME: None},
                length=self.depth
            )(**block_args)
        

    def __call__(self, x, t, y, training: bool = False, 
                 return_aux: bool = False):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent 
                        representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # Conditioning
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, training)         # (N, D)
        c = t + y                                # (N, D)
        
        # X Embedding
        x = self.x_embedder(x, c)                # (N, T, D)
        xlst = None
        if isinstance(x, tuple):
            x, xlst = x
        # (N, T, D), where T = H * W / patch_size ** 2
        x = x + self.pos_emb                     
        x = nn.with_logical_constraint(x, ("B", "N", "D"))
        c = nn.with_logical_constraint(c, ("B", "D"))
        
        # Diffusions
        aux_lst = []
        moe_counts = []
        moe_route_prob_sums = []
        moe_gate_logits = []
        
        assert len(self.skip_layers) == 0, "Scan doesn't support skip connections"
        (x, _), scan_out = self.blocks((x, c), training, return_aux)
        if self.scan_remat:
            if return_aux:
                for lyr in range(self.depth):
                    idx_i = lyr // self.scan_remat_block_size
                    idx_j = lyr % self.scan_remat_block_size
                    aux_lst.append(
                        jax.tree.map(lambda o, l=lyr: o[idx_i, idx_j], scan_out))
                    aux = aux_lst[-1]
        else:
            if return_aux:
                for lyr in range(self.depth):
                    aux_lst.append(
                        jax.tree.map(lambda o, l=lyr: o[l], scan_out))
                    aux = aux_lst[-1]
    

        x = self.final_layer(x, c)      # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, c, xlst) # (N, out_channels, H, W)
        if return_aux:
            aux_dict = {"dit_blocks": aux_lst}
            moe_info = {
                'counts': jnp.stack(moe_counts) if moe_counts else None,
                'route_prob_sums': jnp.stack(moe_route_prob_sums) if moe_route_prob_sums else None,
                'gate_logits': jnp.stack(moe_gate_logits) if moe_gate_logits else None,
                'experts': self.dit_block_kwargs.get('n_experts', None)
            }
            
            aux_dict['moe'] = moe_info

            return x, aux_dict
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, training: bool = False):
        """
        TODO: guandao - fix this function so that we can do sampling
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = jnp.concatenate([half, half], axis=0)
        model_out = self.__call__(combined, t, y, training=training)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        # TODO: check this line
        # cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        cond_eps, uncond_eps = jnp.split(eps, 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = jnp.concatenate([half_eps, half_eps], axis=0)
        return jnp.concatenate([eps, rest], axis=1)

