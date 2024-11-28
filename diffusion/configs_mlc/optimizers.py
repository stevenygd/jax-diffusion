from diffusion.configs_mlc import CFG 


def get_config(cfg, opt_type):
    match opt_type.lower():
        case "adamw":
            cfg.opt = CFG(
                lr=1e-4,
                wd=0,
                grad_clip=None
            )
        case _:
            raise NotImplemented
    return cfg