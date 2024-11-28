import os
import jax
import glob
import time
import wandb
import hydra
import numpy as np
import os.path as osp
from omegaconf import DictConfig, OmegaConf
from utils import dit_flops 
import subprocess
from datetime import datetime
import pytz

from compute_flops_jax import compute_flops_per_iter

def create_sample_dir(args, ckpt_step):
    # Create folder to save samples:
    model_string_name = args.model.name.replace("/", "-")
    folder_name = f"{model_string_name}-"\
                  f"size-{args.image_size}-" \
                  f"cfg-{args.inference.cfg_scale}-" \
                  f"seed-{args.global_seed}-" \
                  f"step-{args.inference.num_sampling_steps}-" \
                  f"nsmp-{args.inference.num_fid_samples}"
    if args.inference.get("mode", "ddpm") == "ddim":
        folder_name = f"{folder_name}-ddim"
    elif args.inference.get("mode", "ddpm") == "rectflow":
        folder_name = f"{folder_name}-rectflow"
                  
    sample_folder_dir = osp.join(
        args.resume, args.inference.sample_dir, str(ckpt_step), folder_name)
    return sample_folder_dir


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args: DictConfig):
    expr_name = args.wandb.expr_name
    group_name = f"group-{expr_name}"

    try:
        # If this failed to resume, start a new ID
        wandb.init(
            entity=args.wandb.entity,       # or args.wandb.entity
            project=args.wandb.project,     # or args.wandb.project
            name=f"{args.inference.mode}sampling-{expr_name}",
            group=group_name,
            config=OmegaConf.to_object(args),
            resume=True
        )
    except:
        id_ = wandb.util.generate_id()
        print("Unable to resume, generate new ID", id_)
        wandb.init(
            entity=args.wandb.entity,       # or args.wandb.entity
            project=args.wandb.project,     # or args.wandb.project
            name=f"{args.inference.mode}sampling-{expr_name}",
            group=group_name,
            config=OmegaConf.to_object(args),
            id=id_
        )

    if args.force_log:
        print("This run immediately starts WandB and finishes logging after a single sweep for only the checkpoints that are ready.")
    else:
        print("This run immediately starts WandB and logs the metrics. If the metric isn't ready yet, it will sleep for 5 minutes and check again.")

    print(f'expr_name: {expr_name}')
    print(f'config: \n{OmegaConf.to_yaml(args, resolve=True)}')

    experiment_dir = args.experiment_dir
    checkpoint_dir = args.checkpoint_dir
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir)
    assert os.path.isdir(checkpoint_dir), f"Checkpoint directory {checkpoint_dir} does not exist."
    
    ckpt_steps = os.listdir(checkpoint_dir)
    processed = []
    todos = [ckpt_step for ckpt_step in ckpt_steps if ckpt_step not in processed]

    while len(todos) > 0:
        script_path = "/mnt/disks/nfs/ujinsong/jax-DiT/scripts/load_data_ujinsong.sh dit-guandao"
        if args.inference.mode == 'rectflow':
            script_path = "/mnt/disks/nfs/ujinsong/jax-DiT/scripts/load_data_ujinsong_rectflow.sh sci-guandao"
        _ = subprocess.run(['bash', script_path], capture_output=True, text=True)
        stage0, stage1, stage2 = [], [], []
        for ckpt_step in ckpt_steps:
            if ckpt_step in processed:
                continue
            sample_folder_dir = create_sample_dir(args, ckpt_step)
            npz_path = f"{sample_folder_dir}.npz"
            metric_path = f"{sample_folder_dir}-metrics.npy"
            if osp.exists(sample_folder_dir):
                if osp.isfile(npz_path):
                    if osp.isfile(metric_path):
                        metrics = np.load(metric_path, allow_pickle=True).item()
                        metrics["ckpt_step"] = int(ckpt_step)
                        wandb.log(metrics)
                        print(f'\n----step: {ckpt_step} metrics logged at wandb ----')
                        print(metrics)
                        todos.remove(ckpt_step)
                        processed.append(ckpt_step)
                    else:
                        assert ckpt_step in todos
                        stage2.append(ckpt_step)
                else:
                    assert ckpt_step in todos
                    stage1.append(ckpt_step)
            else:
                assert ckpt_step in todos
                stage0.append(ckpt_step)

        utc_now = datetime.now(pytz.utc)
        pdt_now = utc_now.astimezone(pytz.timezone('America/Los_Angeles'))
        print("\n(PDT)", pdt_now.strftime('%Y-%m-%d %H:%M:%S %Z%z'))
        print("Todos", todos)
        print(f"\tStage 0: {stage0}\t\t--Sampling hasn't started . (No folder)")
        print(f"\tStage 1: {stage1}\t\t--Sampling is in progress. (No npz file)")
        print(f"\tStage 2: {stage2}\t\t--Metrics are being computed. (No metrics file)")
        print("Processed", processed)

        if args.force_log:
            break
        if len(todos) > 0:
            print("\nSleeping...(%d mins)\n" % (300 / 60.))
            time.sleep(300)
        
        ckpt_steps = os.listdir(checkpoint_dir)
        todos = [ckpt_step for ckpt_step in ckpt_steps if ckpt_step not in processed]
    
    unit, unit_name = 1e12, "TFlops"
    flops_per_iter, block_flops_dict = compute_flops_per_iter(args, backward=True, unit=unit)
    wandb.log({
        "train_flop_per_iter": flops_per_iter.item(), 
        "unit": unit, 
        "unit_name": unit_name,})

    print(f'flops_per_iter: {flops_per_iter.item()} {unit_name}')
    
    flops_breakdown = {k:v.item() for k, v in block_flops_dict.items() if not isinstance(v, dict)}
    wandb.log(flops_breakdown)
    print(f'DiT Block Flops breakdown: {flops_breakdown} ({unit_name})')
    for d in [v for v in block_flops_dict.values() if isinstance(v, dict)]:
        flops_attn_breakdown = {k:v for k, v in d.items()}
        wandb.log(flops_attn_breakdown)
        print(f'DiT Block Attention Flops breakdown: {flops_attn_breakdown} ({unit_name})')

if __name__ == "__main__":
    main()