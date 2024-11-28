from diffusion.configs_mlc import CFG, update 
from diffusion.configs_mlc.base import basic_config
from diffusion.configs_mlc import datasets, inference, losses, models, optimizers


def get_config(model_type="ttt_tiny"):
    cfg = basic_config()
    cfg = inference.get_config(cfg, "rectflow_10k")
    cfg = losses.get_config(cfg, "rectflow")
    cfg = optimizers.get_config(cfg, "adamw") 
    cfg = datasets.get_config(cfg, "imagenet256_pixels") 
    cfg.model = models.get_config(model_type)
    cfg.model.patch_size = 4
    if model_type.startswith("ttt"):
        cfg.model.attn_kwargs.mttt_kwargs.max_sequence_length = (256 // 4) ** 2
        cfg.model.attn_kwargs.mttt_kwargs.remat_mini_batch_group_size = 16
    return cfg