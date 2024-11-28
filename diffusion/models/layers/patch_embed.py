import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple, List
from dataclasses import field
import numpy as np
from diffusers.models.vae_flax import FlaxDownEncoderBlock2D, FlaxUpDecoderBlock2D

#################################################################################
# https://github.com/huggingface/pytorch-image-models/blob/
# e748805be31318da1a0e34b61294704666f50397/timm/layers/patch_embed.py#L26C1-L110C1
#################################################################################
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    # TODO: order matters
    patch_size: Tuple[int] = (16, 16)
    embed_dim: int = 768
    bias: bool = True

    def setup(self):
        self.proj = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            param_dtype=jnp.float32,
            name="embedding",
            # TODO: check the axis, original torch code with comment
            # # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
            # w = self.x_embedder.proj.weight.data
            # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            kernel_init=nn.initializers.xavier_uniform() ,
            bias_init=nn.initializers.zeros_init(),
        )
        
    def __call__(self, x, *args, **kwargs):
        """ Args:   [x] images with shape (B, C, H, W)
            Rets:   Token list with shape (B, L, C)
        """
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, H, W, C)
        x = x.transpose((0, 2, 3, 1))
        # (B, H, W, C) -> (B, H', W', C)
        x = self.proj(x)
        # (B, H', W', C) -> (B, L, C)
        x = x.reshape(B, -1, self.embed_dim)
        return x
    
    
class FlaxEncoder(nn.Module):
    """ VAE encoder layer, without the middle attention layers. """
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: Tuple[str] = ("DownEncoderBlock2D",)
    block_out_channels: Tuple[int] = (64,)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"
    double_z: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        block_out_channels = self.block_out_channels
        # in
        self.conv_in = nn.Conv(
            block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # downsampling
        down_blocks = []
        output_channel = block_out_channels[0]
        for i, _ in enumerate(self.down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = FlaxDownEncoderBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block,
                resnet_groups=self.norm_num_groups,
                add_downsample=not is_final_block,
                dtype=self.dtype,
            )
            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        # end
        conv_out_channels = 2 * self.out_channels if self.double_z else self.out_channels
        self.conv_norm_out = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-6)
        self.conv_out = nn.Conv(
            conv_out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, sample, deterministic: bool = True):
        # in
        sample = self.conv_in(sample)

        # downsampling
        for block in self.down_blocks:
            sample = block(sample, deterministic=deterministic)

        # end
        sample = self.conv_norm_out(sample)
        sample = nn.swish(sample)
        sample = self.conv_out(sample)

        return sample

    
    
class ConvNetEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    # TODO: order matters
    patch_size: Tuple[int] = (16, 16)
    embed_dim: int = 768
    layers_per_block: int = 1

    def setup(self):
        patch_size = self.patch_size[0] if isinstance(self.patch_size, tuple) else self.patch_size
        num_layers = int(np.round(np.log(self.patch_size)/np.log(2)))
        self.encoder = FlaxEncoder(
            in_channels=3,
            out_channels=self.embed_dim,
            down_block_types=["DownEncoderBlock2D"] * (num_layers + 1),
            block_out_channels=[
                min(512, 32 * 2**i) for i in range(num_layers + 1)],
            layers_per_block=self.layers_per_block,
            norm_num_groups=32,
            act_fn="silu",
            double_z=False
        )
        
    def __call__(self, x, *args, **kwargs):
        """ Args:   [x] images with shape (B, C, H, W)
            Rets:   Token list with shape (B, L, C)
        """
        B, C, H, W = x.shape
        # (B, D, H', W') -> (B, H', W', D)
        x = x.transpose((0, 2, 3, 1))
        # (B, C, H, W, C) -> (B, D, H', W')
        x = self.encoder(x)
        # (B, H', W', D) -> (B, L=H'* W', D)
        x = x.reshape(B, -1, self.embed_dim)
        return x



class UnPatchify(nn.Module):
    out_channels: int
    patch_size: int
    
    @nn.compact    
    def __call__(self, x, *args, **kwargs):
        """
        Args: [x]:      (N, T, patch_size**2 * C)
        Rets: [imgs]:   (N, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = jnp.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], c, h * p, h * p)
        return imgs
    
    
class FlaxDecoder(nn.Module):
    """ VAE Decoder layer, without the middle attention layers. """

    in_channels: int = 3
    out_channels: int = 3
    up_block_types: Tuple[str] = ("UpDecoderBlock2D",)
    block_out_channels: int = (64,)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        block_out_channels = self.block_out_channels

        # z to block_in
        self.conv_in = nn.Conv(
            block_out_channels[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # upsampling
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        up_blocks = []
        for i, _ in enumerate(self.up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = FlaxUpDecoderBlock2D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block + 1,
                resnet_groups=self.norm_num_groups,
                add_upsample=not is_final_block,
                dtype=self.dtype,
            )
            up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.up_blocks = up_blocks

        # end
        self.conv_norm_out = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-6)
        self.conv_out = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, sample, deterministic: bool = True):
        # z to block_in
        sample = self.conv_in(sample)

        # upsampling
        for block in self.up_blocks:
            sample = block(sample, deterministic=deterministic)

        sample = self.conv_norm_out(sample)
        sample = nn.swish(sample)
        sample = self.conv_out(sample)

        return sample

    
   
class ConvNetUnPatchify(nn.Module):
    out_channels: int
    patch_size: int
    layers_per_block: int = 1
    
    def setup(self):
        num_layers = int(np.round(np.log(self.patch_size)/np.log(2)))
        self.decoder = FlaxDecoder(
            out_channels=self.out_channels,
            up_block_types=["UpDecoderBlock2D"] * (num_layers + 1),
            block_out_channels=[
                min(512, 32 * 2**i) for i in range(num_layers + 1)],
            layers_per_block=self.layers_per_block,
            norm_num_groups=32,
            act_fn="silu",
        )
        
    def __call__(self, x, *args, **kwargs):
        """
        Args: [x]:      (N, T, C)
        Rets: [imgs]:   (N, H, W, C)
        """
        b = x.shape[0]
        c = x.shape[-1]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        imgs = x.reshape(b, h, w, c)
        imgs = self.decoder(imgs) 
        imgs = imgs.transpose((0, 3, 1, 2))
        return imgs
    
################################################################################
# Simple diffusion
################################################################################

class SimpleDiffusionResBlock(nn.Module):
    dim: int  # hidden dimension
    
    @nn.compact
    def __call__(self, x, c, skip=None):
        """
        Args:
            [x] (B, H, W, D)
            [c] (B, D')
        Rets:
            (B, H, W, D)
        """
        h = nn.LayerNorm()(x)
        if skip is not None:
            skip = nn.LayerNorm()(skip)
            h = (h + skip) / jnp.sqrt(2.)
        h = nn.swish(h)
        h = nn.Conv(self.dim, (3, 3), (1, 1), 'SAME')(h)
        
        # Conditioninig
        # (B, D) -> (B, 1, 1, 2D) -> (B, 1, 1, D), (B, 1, 1, D)
        emb_out = nn.Dense(
            self.dim * 2, 
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init()
        )(c)[:, None, None, :]
        scale, shift = jnp.split(emb_out, 2, axis=-1)
        h = nn.LayerNorm()(h) * (1 + scale) + shift
        
        h = nn.swish(h)
        h = nn.Conv(
            self.dim, (3, 3), (1, 1), 'SAME',
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init()
        )(h)
        return x + h
   
    
class SimpleDiffusionEmbedder(nn.Module):
    patch_size: int = 1
    embed_dim: int = 768
    
    base_channel: int = 128
    channel_multiplier: List[int] = field(default_factory=lambda: [1, 2, 4])
    num_res_blocks: List[int] = field(default_factory=lambda: [2, 2, 2])
   
    @nn.compact 
    def __call__(self, x, c, *args, **kwargs):
        """
        Args:
            [x] (B, C, H, W)
            [c] (B, D)
        Returns:
            (B, H * W // (2 ** len(num_res_blocks)), emb_dim)
        """
        # EmbededInput
        n_downs = int(np.log(float(self.patch_size)) / np.log(2.))
        
        B, C, H, W = x.shape
        # (B, D, H', W') -> (B, H', W', D)
        x = x.transpose((0, 2, 3, 1))
        
        h = x
        if n_downs > 0:
            h = nn.Conv(
                self.base_channel * self.channel_multiplier[0], 
                (3, 3), (1, 1), 'SAME')(x)
        h_lst = []
        for i in range(n_downs):
            num_blocks = self.num_res_blocks[i]
            channel_mul = self.channel_multiplier[i]
            for _ in range(num_blocks):
                h = SimpleDiffusionResBlock(
                    self.base_channel * channel_mul)(h, c)
            h_lst.append(h)
            h = nn.avg_pool(h, (2, 2), (2, 2), 'SAME')
            if i < n_downs - 1:
                h = nn.Dense(
                    self.base_channel * self.channel_multiplier[i+1])(h)
            
        # (B, H', W', D) -> (B, D, H', W')
        h = PatchEmbed(
            patch_size=1, embed_dim=self.embed_dim)(h.transpose((0, 3, 1, 2)))
        return h, h_lst
    
    
class SimpleDiffusionUnpacify(nn.Module):
    out_channels: int
    patch_size: int
    
    base_channel: int = 128 
    channel_multiplier: List[int] = field(default_factory=lambda: [1, 2, 4])
    num_res_blocks: List[int] = field(default_factory=lambda: [2, 2, 2])
   
    # Whether use conv layer for the final output
    last_conv: bool = False
    
    def upsample(self, x, out_dim: int):
        b, h, w, c = x.shape
        x = jax.image.resize(x, (b, h * 2, w * 2, c), method='nearest')
        x = nn.Dense(out_dim)(x) 
        return x
    
    @nn.compact
    def __call__(self, x, c, xlst: list, *args, **kwargs):
        """
        Args:
            [x] (B, T, emb_dim)
            [c] (B, D)
            [xlst] List of hidden states
        Returns:
            (B, C, H, W)
        """
        
        # (B, H*W, C) -> (B, C, H, W)
        x = UnPatchify(out_channels=x.shape[-1], patch_size=1)(x)
        # (B, C, H, W) -> (B, H, W, C)
        x = x.transpose((0, 2, 3, 1))        
        
        n_ups = int(np.log(float(self.patch_size)) / np.log(2.))
        for i in range(n_ups - 1, -1, -1):
            num_blocks = self.num_res_blocks[i]
            channel_mul = self.channel_multiplier[i]
            x = self.upsample(x, self.base_channel * channel_mul)
            for _ in range(num_blocks):
                x = SimpleDiffusionResBlock(
                    self.base_channel * channel_mul)(x, c, xlst[i])
                
        # Follow the DiT Final layer to have zero init
        if self.last_conv:
            x = nn.Conv(
                self.out_channels,
                (3, 3), (1, 1), 'SAME',
                kernel_init=nn.initializers.zeros_init(),
                bias_init=nn.initializers.zeros_init()
            )(x)
        else:
            x = nn.Dense(
                self.out_channels,
                kernel_init=nn.initializers.zeros_init(),
                bias_init=nn.initializers.zeros_init()
            )(x) 
        
        # (B, H, W, C) -> (B, C, H, W)
        x = x.transpose((0, 3, 1, 2))
        return x