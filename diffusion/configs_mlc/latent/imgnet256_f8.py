from diffusion.configs_mlc import CFG, update, basic_config
from diffusion.configs_mlc import datasets, inference, losses, models, optimizers


def get_config(model_type="dit_s"):
    cfg = basic_config()
    cfg = inference.get_config(cfg, "rectflow_10k")
    cfg = losses.get_config(cfg, "rectflow")
    cfg = optimizers.get_config(cfg, "adamw") 
    cfg = datasets.get_config(cfg, "imagenet256_latent_f8") 
    cfg.model = models.get_config(model_type)
    cfg.model.patch_size = 1
    return cfg