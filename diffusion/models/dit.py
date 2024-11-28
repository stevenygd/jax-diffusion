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
    TorchAttention, JaxAttention, MTTTAttention, LinearAttention, 
    ViTTTAttention, ViTTTLinearAttention, DiffuserAttention)
from diffusion.models.layers.s4d_jax import S4DAttention
from diffusion.models.layers.moe_mlp import SwitchFeedForward
from diffusion.models.layers.mttt_lm import TTTLMAttention, TTTLMBiDirAttention, TTTLMBiDirAttentionV2
from diffusion.models.layers.patch_embed import (
    PatchEmbed, UnPatchify, ConvNetEmbed, ConvNetUnPatchify,
    SimpleDiffusionEmbedder, SimpleDiffusionUnpacify
)
from diffusion.models.layers.bpt import blockwise_ffn 
from diffusion.models.utils import precision_str_to_type
from jax.sharding import PartitionSpec as PS
from diffusion.utils.sharding import with_sharding_constraint


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Adapted to Jax from DiT

def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
        pos_embed: [grid_size*grid_size, embed_dim]
                or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # TODO(guandao): do we need float64 here?
    # grid_h = jnp.arange(grid_size, dtype=np.float64)
    # grid_w = jnp.arange(grid_size, dtype=np.float64)
    grid_h = jnp.arange(grid_size, dtype=np.float32)
    grid_w = jnp.arange(grid_size, dtype=np.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = jnp.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = jnp.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    # TOOD: do we need float64 here?
    # omega = jnp.arange(embed_dim // 2, dtype=np.float64)
    omega = jnp.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out) # (M, D/2)
    emb_cos = jnp.cos(out) # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


################################################################################
#                                 Core DiT Model                               #
################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale[:, None, ...]) + shift[:, None, ...]


class DiTBlock(nn.Module):
    """ A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning. """
    hidden_size: int
    num_heads: int
    fst_cond_type: str = "modulate"
    snd_cond_type: str = "modulate"
    mlp_ratio: float = 4.0
    # rnn_dim: int = 128
    # ssm_expansion: int = 2
    attn_kwargs: Optional[dict] = field(default_factory=dict)
    attn_type: str = "torch"
    mlp_type: str = "mlp"
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
            case "torch":
                attn_class = TorchAttention
            case "diffuser":
                attn_class = DiffuserAttention 
            case "mttt":
                attn_class = MTTTAttention
            case "linear":
                attn_class = LinearAttention
            case "vit3_linear":
                attn_class = ViTTTLinearAttention
            case "vit3":
                attn_class = ViTTTAttention
            case "ssm":
                attn_class = S4DAttention
            case "ttt_lm":
                attn_class = TTTLMAttention 
            case "ttt_lm_bd":
                attn_class = TTTLMBiDirAttention
            case "ttt_lm_bd_v2":
                attn_class = TTTLMBiDirAttentionV2
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
        mlp_class = SwitchFeedForward if self.mlp_type == "moe" else Mlp
        if self.grad_checkpoint_mlp:
            mlp_class = nn.remat(
                mlp_class, static_argnums=(2,),
                policy=jax.checkpoint_policies.nothing_saveable
            ) 
       
        mlp_dtype = precision_str_to_type(self.mlp_dtype) 
        mlp_ptype = precision_str_to_type(self.mlp_ptype) 
        if self.mlp_type == "moe":
            # TODO(guandao): implement mixed precision
            self.mlp = mlp_class(
                hidden_size=self.hidden_size,
                intermediate_size=mlp_hidden_dim,
                n_experts=self.n_experts, 
                capacity_factor=1.0,
                drop_tokens=True,
                is_scale_prob=True,
                act_layer="gelu_approx", drop=0,
                ptype=mlp_ptype, dtype=mlp_dtype
            )
        else:
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
            kernel_init=nn.initializers.zeros_init(),
            # kernel_init=nn.with_logical_partitioning(
            #     nn.initializers.zeros_init(),
            #     ("adaln/inp", "adaln/out")
            # ),
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
        elif self.mlp_type == "moe":
            # deciding whether we are going to use FFNs
            mlp_out, counts, route_prob_sum, gate_logits = self.mlp( x, training)
            if return_aux:
                aux.update({
                    "moe_block": {
                        "counts": counts,
                        "route_prob_sum": route_prob_sum,
                        "gate_logits": gate_logits,
                    }
                })
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
    - https://github.com/google-research/big_vision/blob/b8dab6e4de3436849415f37c591399c93b1eaf39/big_vision/models/vit.py#L79
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
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = x.chunk(6, dim=1)
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
    class_dropout_prob:float =0.1
    num_classes:int =1000
    learn_sigma: bool = False 
    package_name: str = "dit" 

    # Conditioning parameters
    fst_cond_type: str = "modulate"  #  concat
    snd_cond_type: str = "modulate"  #  concat
    
    # Attention parameters
    attn_type: str = "torch"
    ssm_expansion: int = 2
    attn_kwargs: Optional[dict] = field(default_factory=dict)
    grad_checkpoint: bool = False
    dit_block_kwargs: Optional[dict] = field(default_factory=dict)
    scan_blocks: bool = False
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
            case "vae":
                self.x_embedder = ConvNetEmbed(
                    patch_size=self.patch_size, embed_dim=self.hidden_size,
                    **self.x_emb_kwargs)
                self.final_layer = lambda x, c: x
                self.unpatchify = ConvNetUnPatchify(
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
            "fst_cond_type": self.fst_cond_type,
            "snd_cond_type": self.snd_cond_type,
            "attn_type": self.attn_type,
            "attn_kwargs": self.attn_kwargs,
            **self.dit_block_kwargs
        }
        if self.scan_blocks:
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
                    
                    # # Doesn't work, can't take static argument
                    # self.blocks = nn.remat_scan(
                    #     DiTBlockScanRemat,
                    #     lengths=(sqrt_depth, int(self.depth // sqrt_depth)),
                    #     policy=jax.checkpoint_policies.nothing_saveable,
                    #     split_rngs={"params": True, "dropout": True},
                    # )(**block_args)
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
        else:
            block_fn = DiTBlock
            if self.grad_checkpoint:
                block_fn = nn.remat(
                    block_fn, 
                    prevent_cse=False,
                    static_argnums=(3, 4), # 0=self, 12=x,c, 3=train, 4=aux
                    policy=jax.checkpoint_policies.nothing_saveable,
                )
            self.blocks = [block_fn(**block_args) for _ in range(self.depth)]
        
        # Initialize for skip connections
        self.skip_blocks = [
            (nn.Dense(self.hidden_size) if i in self.skip_layers 
             else lambda x: x) 
            for i in range(self.depth) 
        ]

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
        if self.scan_blocks:
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
                        if 'moe_block' in aux:
                            moe_counts.append(aux['moe_block']['counts'])
                            moe_route_prob_sums.append(aux['moe_block']['route_prob_sum'])
                            moe_gate_logits.append(aux['moe_block']['gate_logits'])
            else:
                if return_aux:
                    for lyr in range(self.depth):
                        aux_lst.append(
                            jax.tree.map(lambda o, l=lyr: o[l], scan_out))
                        aux = aux_lst[-1]
                        if 'moe_block' in aux:
                            moe_counts.append(aux['moe_block']['counts'])
                            moe_route_prob_sums.append(aux['moe_block']['route_prob_sum'])
                            moe_gate_logits.append(aux['moe_block']['gate_logits'])
        else: 
            skip_lst = {}
            for lid, block in enumerate(self.blocks):
                if lid in self.skip_layers:
                    skip_lst[lid] = x
                elif (len(self.blocks) - lid) in self.skip_layers:
                    x_skip = skip_lst[len(self.blocks) - lid]
                    x_dense = self.skip_blocks[len(self.blocks) - lid]
                    x = x_dense(jnp.concatenate([x, x_skip], axis=-1))
                x = block(x, c, training, return_aux)      # (N, T, D)
                if return_aux:
                    x, aux = x 
                    aux_lst.append(aux)
                    if 'moe_block' in aux:
                        moe_counts.append(aux['moe_block']['counts'])
                        moe_route_prob_sums.append(aux['moe_block']['route_prob_sum'])
                        moe_gate_logits.append(aux['moe_block']['gate_logits'])

        x = self.final_layer(x, c)               # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, c, xlst)          # (N, out_channels, H, W)
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


if __name__ == "__main__":
    import lovely_jax as lj
    lj.monkey_patch()

    input_size = 256
    patch_size = 16
    hidden_size = 512
    num_heads = 8
    depth = 8
    model = DiT(
        input_size=input_size, 
        depth=depth, num_heads=num_heads,
        hidden_size=hidden_size, patch_size=patch_size, 
    )
    B, H, W, C, D = 128, input_size, input_size, 3, 128
    x = jnp.ones((B, C, H, W))
    t = jnp.ones((B,))
    y = jnp.ones((B,), dtype=jnp.int_)
    rng = jax.random.PRNGKey(66)
    params = model.init(rng, x, t, y, training=True)
    print("Done initializing:", model)
    print("Params:", params)
    rng, dropout_key, labelemb_key = jax.random.split(rng, 3)
    out = model.apply(
        params, x, t, y, training=True, 
        rngs={'dropout': dropout_key, "label_emb": labelemb_key})
    print("Train output:", out)
    
    out = model.apply(
        params, x, t, y, training=False,
    )
    print("Test  output:", out)
