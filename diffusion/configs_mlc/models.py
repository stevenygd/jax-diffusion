import numpy as np
from diffusion.configs_mlc import CFG, update


def get_config(model_type):
    model_type = model_type.lower()
    if model_type.startswith("ttt"):
        return get_config_ttt(model_type)
    elif model_type.startswith("dit"):
        return get_config_dit(model_type)
    elif model_type.startswith("ssm"):
        return get_config_ssm(model_type)
    elif model_type.startswith("uvit"):
        return get_config_uvit(model_type)
    else:
        raise NotImplemented
    
    
def get_config_uvit(model_cfg):
    block_args = {
        "depth": 2,
        "hidden_size": -1,
        "num_heads": 6, 
        "attn_type": "ssm",
        "attn_kwargs": CFG(
            d_expansion=1,
            d_state=256,
        ),
        "dit_block_kwargs": CFG(
            grad_checkpoint_attn=False,
            grad_checkpoint_mlp=False,
            grad_checkpoint_adamlp=False,
            blockwise_mlp=False,
            blockwise_mlp_chunk_size=128,
            mlp_dtype="bfloat16",
            adaln_mlp_dtype="bfloat16",
        )
    }
    
    def make_stage(**kwargs):
        return update(CFG(**block_args), **kwargs)
    
    cfg = CFG(
        name="UViT-SSM-S",
        package_name="uvit",
        skip_idxs=[0, 1, 2],
        down_idxs=[0, 1, 2],
        patch_size=2,
        patch_type="vit",
        c_hidden_size=384,
        p_hidden_size=32,
        stage_cfgs=[
            make_stage(depth=2, hidden_size=32, num_heads=1),
            make_stage(depth=2, hidden_size=64, num_heads=2),
            make_stage(depth=2, hidden_size=128, num_heads=4),
            make_stage(depth=12, hidden_size=384, num_heads=6),
        ]
    )
    return cfg


def get_config_dit(model_cfg):
    head_dim = 64
    model_cfg_lst = model_cfg.lower().split("_")
    match "_".join(model_cfg_lst[:2]):
        case "dit_tiny":
            num_heads = 3
            cfg = CFG(
                name="DiT-Ti",
                depth=12,
                hidden_size=192,
                patch_size=1,
                num_heads=num_heads,
                attn_type="jax",
            )
        case "dit_s":
            num_heads = 6
            cfg = CFG(
                name="DiT-S",
                depth=12,
                hidden_size=384,
                patch_size=1,
                num_heads=num_heads,
                attn_type="jax",
            )
        case "dit_b":
            num_heads = 12
            cfg = CFG(
                name="dit-b",
                depth=12,
                hidden_size=768,
                patch_size=1,
                num_heads=num_heads,
                attn_type="jax",
            )
        case "dit_l":
            num_heads = 16
            cfg = CFG(
                name="DiT-L",
                depth=24,
                hidden_size=1024,
                patch_size=1,
                num_heads=num_heads,
                attn_type="jax",
            )
        case "dit_xl":
            num_heads = 16
            cfg = CFG(
                name="DiT-XL",
                depth=28,
                hidden_size=1152,
                patch_size=1,
                num_heads=num_heads,
                attn_type="jax",
            )
        case _:
            raise ValueError(f"Invalid model_cfg: {model_cfg}")
        
    if len(model_cfg_lst) > 2 and model_cfg_lst[2] == "eff":
        cfg = update(
            cfg, 
            attn_type="diffuser",
            attn_kwargs=CFG(use_mem_efficient_attn=True)
        )
    cfg.package_name = "dit"

    return cfg
        
        
def get_config_ssm(model_cfg):
    attn_cfg = CFG(
        d_expansion=1,
        d_state=256
    )
    dit_cfg = CFG(
        grad_checkpoint_attn=False,
        grad_checkpoint_mlp=False,
        grad_checkpoint_adamlp=False,
        blockwise_mlp=False,
        blockwise_mlp_chunk_size=128,
        mlp_dtype="bfloat16",
        adaln_mlp_dtype="bfloat16",
    )
    common_args = {
        "attn_type": "ssm",
        "attn_kwargs": attn_cfg,
        "dit_block_kwargs": dit_cfg,
    }
    model_cfg_lst = model_cfg.lower().split("_")
    match "_".join(model_cfg_lst[:2]):
        case "ssm_tiny":
            cfg = get_config_dit("dit_tiny")
            cfg = update(cfg, name="SSM-Ti", **common_args)
        case "ssm_s":
            cfg = get_config_dit("dit_s")
            cfg = update(cfg, name="SSM-S", **common_args)
        case "ssm_b":
            cfg = get_config_dit("dit_b")
            cfg = update(cfg, name="SSM-B", **common_args)
        case "ssm_l":
            cfg = get_config_dit("dit_l")
            cfg = update(cfg, name="SSM-L", **common_args)
        case "ssm_xl":
            cfg = get_config_dit("dit_xl")
            cfg = update(cfg, name="SSM-XL", **common_args)
        case _:
            raise NotImplemented
    # Pop two elements from the list
    model_cfg_lst.pop(0)
    model_cfg_lst.pop(0)

    if len(model_cfg_lst) > 0 and model_cfg_lst[0] == "simpdiff":
        print("Simple diffusion!")
        cfg.patch_type = "simple_diffusion" 
        model_cfg_lst.pop(0)
        
    if len(model_cfg_lst) > 0 and model_cfg_lst[0] == "uvit":
        print("UVIT!")
        cfg = update(
            cfg, skip_layers=list(
                range(0, int(np.floor(cfg.depth / 2.)) - 1, 1)))
        model_cfg_lst.pop(0)
   
    cfg.package_name = "dit"
    return cfg

        
        
def get_config_ttt(model_cfg):
    attn_cfg = CFG(
        grad_checkpoint_qkv=False,
        grad_checkpoint_out=False,
        qkv_dtype="bfloat16",
        out_dtype="bfloat16",
        separate_qkv=False,
        proj_norm=True,
        mult_norm=False,
        apply_gate=False,
        sigmoid_learnable_token_idx=True,
        mttt_type="mlp_base",
        mttt_kwargs=CFG(
            hidden_size=None,
            num_attention_heads=None,
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/model.py#L76 
            conv_width=4,
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/model.py#L64C9-L64C35 
            initializer_range=0.02,
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/model.py#L72C9-L72C31
            mini_batch_size=16,
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/model.py#L63
            max_sequence_length=16384,
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/model.py#L74C9-L74C31
            rope_theta=10000.0,
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/scripts/ttt_mlp/125m.sh#L24C107-L24C129
            ttt_base_lr=0.1,
            ttt_base_lr_init=0.01,
            ttt_base_lr_warmup=480,
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/model.py#L73C1-L74C1
            remat_mini_batch_group_size=32,
            # https://github.com/test-time-training/ttt-lm-jax/blob/ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/model.py#L213
            remat_conv="",
            output_ttt_stats=True,
        )
    )
    dit_cfg = CFG(
        grad_checkpoint_attn=False,
        grad_checkpoint_mlp=False,
        grad_checkpoint_adamlp=False,
        blockwise_mlp=False,
        blockwise_mlp_chunk_size=128,
        mlp_dtype="bfloat16",
        adaln_mlp_dtype="bfloat16",
    )
    common_args = {
        "attn_type": "ttt_lm_bd",
        "attn_kwargs": attn_cfg,
        "dit_block_kwargs": dit_cfg,
    }
    match model_cfg.lower():
        case "ttt_tiny":
            cfg = get_config_dit("dit_tiny")
            cfg = update(cfg, name="TTT-Ti", **common_args)
        case "ttt_s":
            cfg = get_config_dit("dit_s")
            cfg = update(cfg, name="TTT-S", **common_args)
        case "ttt_b":
            cfg = get_config_dit("dit_b")
            cfg = update(cfg, name="TTT-B", **common_args)
        case "ttt_l":
            cfg = get_config_dit("dit_l")
            cfg = update(cfg, name="TTT-L", **common_args)
        case _:
            raise NotImplemented
        
    cfg.package_name = "dit"
    return cfg
