from diffusion.configs_mlc import CFG


def get_config(cfg, loss_type):
    match loss_type.lower():
        case "rectflow_10k":
            cfg.inference = CFG(
                name="rectflow_10k",
                num_sampling_steps=50,
                num_fid_samples=10_000,
                cfg_scale=1.5,
                sleep_interval=10, 
                ref_batch="./data/imagenet256_reference/VIRTUAL_imagenet256_labeled.npz",
                per_proc_batch_size=32,
                sample_dir="samples",
                remove_existing_sample_dir=False
            )
        case "ddpm_10k":
            cfg.inference = CFG(
                name="ddpm_10k",
                num_sampling_steps=50,
                num_fid_samples=10_000,
                cfg_scale=1.5,
                sleep_interval=10,
                ref_batch="./data/imagenet256_reference/VIRTUAL_imagenet256_labeled.npz",
                per_proc_batch_size=32,
                sample_dir="samples",
                remove_existing_sample_dir=False
            )
        case _:
            raise NotImplemented
    return cfg