from diffusion.configs_mlc import CFG


def basic_config():
    return CFG(
        # Path configurations
        data_dir="./data",
        work_dir="./results",
        
        # Experiment details
        expr_name="test",

        # Train configurations 
        total_iters=400_000, 
        global_batch_size=256,
        grad_acc=1,
        global_seed=0,
        log_every=100,
        ckpt_every=25_000,
        ema_decay=0.9999,
        max_ckpt_keep=16,
        return_aux=True,
        
        dp_dim=-1,
        fsdp_dim=1,
        tp_dim=1,
    )
