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
from jax.sharding import PartitionSpec as PS
from diffusion.models.layers.patch_embed import PatchEmbed, UnPatchify
from diffusion.models.utils import precision_str_to_type
from diffusion.utils.sharding import with_sharding_constraint
from diffusion.models.utils import Identity
from diffusion.models.dit import (
    TimestepEmbedder, LabelEmbedder, get_2d_sincos_pos_embed, FinalLayer,
    DiTBlock, DiTBlockScan, DiTBlockScanRemat
)

################################################################################
#                                 Core DiT Model                               #
################################################################################
class Stage(nn.Module):
    """
        Single Stage of the DiT Blocks. It will have the same context length.
    """
    depth: int = 28
    hidden_size: int = 1152
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    dit_block_kwargs: Optional[dict] = field(default_factory=dict)

    # Conditioning parameters
    fst_cond_type: str = "modulate"  #  concat
    snd_cond_type: str = "modulate"  #  concat
    
    # Attention parameters
    attn_type: str = "torch"
    attn_kwargs: Optional[dict] = field(default_factory=dict)
    
    # Skip layer
    use_skip: bool = True
    
    # gradient checkpointing
    grad_checkpoint: bool = False
    scan_blocks: bool = False
    scan_remat: bool = False
    scan_remat_block_size: int = 1
    
    def setup(self):
        block_fn = DiTBlock
        block_args = {
            "hidden_size": self.hidden_size, 
            "num_heads": self.num_heads, 
            "mlp_ratio": self.mlp_ratio,
            "fst_cond_type": "modulate",
            "snd_cond_type": "modulate",
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
            
        if self.use_skip:
            self.skip_layer = nn.Dense(self.hidden_size)
    
    def __call__(self, x, c, training: bool, return_aux: bool, skip=None):
        if skip is not None:
            x = self.skip_layer(jnp.concatenate([x, skip], axis=-1))
        
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
            for block in self.blocks:
                x = block(x, c, training, return_aux)      # (N, T, D)
                if return_aux:
                    x, aux = x 
                    aux_lst.append(aux)
                    if 'moe_block' in aux:
                        moe_counts.append(aux['moe_block']['counts'])
                        moe_route_prob_sums.append(aux['moe_block']['route_prob_sum'])
                        moe_gate_logits.append(aux['moe_block']['gate_logits'])
                        
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

 
class DownSample(nn.Module):
    out_dim: int
   
    @nn.compact 
    def __call__(self, x):
        B, L, C = x.shape
        H = W = int(np.sqrt(L).astype(int))
        assert H * W == L
        x = x.reshape(B, H, W, C)
        x = nn.avg_pool(x, (2, 2), (2, 2), 'SAME')
        x = nn.Dense(self.out_dim)(x)
        x = x.reshape(B, L // 4, self.out_dim)
        return x
   
    
class UpSample(nn.Module):
    out_dim: int
   
    @nn.compact 
    def __call__(self, x):
        B, L, C = x.shape
        H = W = int(np.sqrt(L).astype(int))
        assert H * W == L
        x = x.reshape(B, H, W, C)
        x = jax.image.resize(x, (B, H * 2, W * 2, C), method='nearest')
        x = nn.Dense(self.out_dim)(x) 
        x = x.reshape(B, L * 4, self.out_dim)
        return x

 
class Model(nn.Module):
    """ Diffusion model with a Transformer backbone.  Jax implementation. 
        UViT with skip connection. It will be formulated in the following way.
       
    patchify 
            \\
             \\
        Stage 1     -   downsample?    --   stage 2     -- ...  --  Stage n
           | skip                              | skip                 / same
        Stage 1 dup -   upsample?      --  stage 2 dup  -- ... -- Stage n 
             //
            //
    unpatchify 
    """
    input_size: int 
    num_classes: int
    in_channels: int
    c_hidden_size: int
    p_hidden_size: int
    package_name: str = "uvit" # Not useful
    class_dropout_prob: float = 0.1
    learn_sigma: bool = False
    
    # Patching and unpatching
    patch_size: int = 1
    patch_type: str = "vit" # vae | simple_diffusion
    x_emb_kwargs: Optional[dict] = field(default_factory=dict)
    unpat_kwargs: Optional[dict] = field(default_factory=dict)
    
    # Stage information
    stage_cfgs: list = field(default_factory=list) # stages that change res
    skip_idxs: list = field(default_factory=list)
    down_idxs: list = field(default_factory=list)
    
    def setup(self):
        # Embedding
        self.out_channels = (
            self.in_channels * 2 if self.learn_sigma else self.in_channels)
        self.t_embedder = TimestepEmbedder(self.c_hidden_size)
        self.y_embedder = LabelEmbedder(
            self.num_classes, self.c_hidden_size, self.class_dropout_prob)
        
        self.x_embedder = PatchEmbed(
            patch_size=self.patch_size, embed_dim=self.p_hidden_size, 
            bias=True, **self.x_emb_kwargs)
        self.final_layer = FinalLayer(
            hidden_size=self.p_hidden_size, patch_size=self.patch_size, 
            out_channels=self.out_channels)
        self.unpatchify = UnPatchify(
            self.out_channels, self.patch_size, **self.unpat_kwargs)
        
        # Fixed, not optimizing
        num_patches_per_dim = self.input_size // self.patch_size
        self.pos_emb = get_2d_sincos_pos_embed(
            self.p_hidden_size, num_patches_per_dim)[None, ...]

        # Initializing stages
        self.stages_frd = [
            Stage(**stage_cfg) for stage_cfg in self.stage_cfgs[:-1]]
        self.down_layers = [
            DownSample(self.stage_cfgs[i+1].hidden_size) 
            if i in self.down_idxs else Identity()
            for i in range(len(self.stages_frd))
        ]
        
        self.center_stage = Stage(**self.stage_cfgs[-1])
        
        self.stages_bck = [
            Stage(**stage_cfg) for stage_cfg in self.stage_cfgs[:-1][::-1]]
        self.up_layers = [
            UpSample(self.stage_cfgs[i].hidden_size) 
            if i in self.down_idxs else Identity()
            for i in range(len(self.stages_frd) - 1, -1, -1)
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
        skip_lst = {}
        for lid, stage in enumerate(self.stages_frd):
            skip_lst[lid] = x if lid in self.skip_idxs else None
            x = stage(x, c, training, return_aux, skip=None)      # (N, T, D)
            if return_aux:
                x, aux = x 
                aux_lst.append(aux)
                if 'moe_block' in aux:
                    moe_counts.append(aux['moe_block']['counts'])
                    moe_route_prob_sums.append(
                        aux['moe_block']['route_prob_sum'])
                    moe_gate_logits.append(aux['moe_block']['gate_logits'])
            x = self.down_layers[lid](x)
       
        # Middle stage 
        x = self.center_stage(x, c, training, return_aux, skip=None) 
        if return_aux:
            x, aux = x 
            aux_lst.append(aux)
            if 'moe_block' in aux:
                moe_counts.append(aux['moe_block']['counts'])
                moe_route_prob_sums.append(aux['moe_block']['route_prob_sum'])
                moe_gate_logits.append(aux['moe_block']['gate_logits'])
                
        for lid, stage in enumerate(self.stages_bck):
            x = self.up_layers[lid](x)
            skip_x = skip_lst[len(self.stages_bck) - 1 - lid]
            x = stage(x, c, training, return_aux, skip=skip_x)      # (N, T, D)
            if return_aux:
                x, aux = x 
                aux_lst.append(aux)
                if 'moe_block' in aux:
                    moe_counts.append(aux['moe_block']['counts'])
                    moe_route_prob_sums.append(
                        aux['moe_block']['route_prob_sum'])
                    moe_gate_logits.append(aux['moe_block']['gate_logits'])

        x = self.final_layer(x, c)      # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, c, xlst) # (N, out_channels, H, W)
        if return_aux:
            aux_dict = {"dit_blocks": aux_lst}
            moe_info = {
                'counts': jnp.stack(moe_counts) if moe_counts else None,
                'route_prob_sums': jnp.stack(moe_route_prob_sums) if moe_route_prob_sums else None,
                'gate_logits': jnp.stack(moe_gate_logits) if moe_gate_logits else None,
                # 'experts': self.dit_block_kwargs.get('n_experts', None)
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
    model = Model(
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
