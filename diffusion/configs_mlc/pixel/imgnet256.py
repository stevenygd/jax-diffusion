from diffusion.configs_mlc.utils import CFG, update, basic_config
from diffusion.configs_mlc import datasets, inference, losses, models, optimizers


def get_config(model_type="ttt_tiny"):
    cfg = basic_config()
    cfg = inference.get_config(cfg, "rectflow_10k")
    cfg = losses.get_config(cfg, "rectflow")
    cfg = optimizers.get_config(cfg, "adamw") 
    cfg = datasets.get_config(cfg, "imagenet256_pixels") 
    cfg.model = models.get_config(model_type)
    cfg.model.patch_size = 2
    return cfg