from diffusion.configs_mlc import CFG


def get_config(cfg, loss_type):
    match loss_type.lower():
        case "rectflow":
            cfg.loss = CFG(
                diffusion_type="rectflow",
                num_timesteps=1000,
                noise_schedule="lognorm",
            )
        case "rectflow_moe":
            cfg.loss = CFG(
                diffusion_type="rectflow_moe",
                num_timesteps=1000,
                noise_schedule="lognorm",
                moe=CFG(
                    moe_loss_weight=1e-2,
                    z_loss_weight=0.0
                )
            )
        case "rectflow_moe_zloss":
            cfg.loss = CFG(
                diffusion_type="rectflow_moe",
                num_timesteps=1000,
                noise_schedule="lognorm",
                moe=CFG(
                    moe_loss_weight=1e-2,
                    z_loss_weight=1e-3
                )
            )
        case _:
            raise NotImplemented
    return cfg