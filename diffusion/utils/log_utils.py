import jax
import wandb 
import jax.numpy as jnp


def log_dit_stats(aux):
    if "aux" in aux and "dit_blocks" in aux["aux"]:
        # and "dit_block" in aux["aux"]["dit_blocks"]):
        for i, block_aux in enumerate(aux["aux"]["dit_blocks"]):
            if "dit_block" in block_aux:
                stats = block_aux["dit_block"]
                wandb.log({
                    f"dit_stats/{k}_layer{i}": v.mean() 
                    for k, v in stats.items()
                }, commit=False)
            if "attn.norms" in  block_aux:
                norm_stats = block_aux["attn.norms"]
                wandb.log({
                    f"mttt_stats/{k}_layer{i}": v.mean() 
                    for k, v in norm_stats.items()
                }, commit=False)


def log_loss_per_time(aux, num_bins=4, eps=1e-6):
    # Compute per time step loss
    
    @jax.jit
    def _compute_stats_(aux):
        loss_per_time = {
            "loss": {"cnt": 0, "val": 0},
            "mse": {"cnt": 0, "val": 0},
            "vb": {"cnt": 0, "val": 0},
            "moe_loss": {"cnt": 0, "val": 0},
        }
        keys = list(aux["loss_per_t"].keys())
        for key in keys:
            val_new = aux["loss_per_t"][key].reshape(-1)
            cnt_new = aux["t_count"].reshape(-1)
            val_orig = loss_per_time[key]["val"]
            cnt_orig = loss_per_time[key]["cnt"]
            new_ttl = cnt_orig + cnt_new + eps
            loss_per_time[key]["val"] = (
                val_orig * (cnt_orig / new_ttl) + val_new / new_ttl)
            loss_per_time[key]["cnt"] = cnt_orig + cnt_new
        splitted_per_t = {
            key: jnp.array(
                jnp.split(loss_per_time[key]["val"], num_bins)
            ).mean(axis=-1) for key in keys 
        }
        return splitted_per_t
    
    splitted_per_t = _compute_stats_(aux)
    wandb.log(
        # Per time step
        {
            f"loss_per_t/{key}/{i}-{num_bins}": val[i]
            for i in range(num_bins) 
            for key, val in splitted_per_t.items()
        }, commit=False
    ) 
