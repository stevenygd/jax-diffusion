import os
import numpy as np
import json

def get_expr_path(expr_name):
    if os.path.exists(f'/mnt/disks/gs2/outputs/checkpoints/{expr_name}'):
        expr_path = f'/mnt/disks/gs2/outputs/checkpoints/{expr_name}'
        assert os.path.exists(f'{expr_path}/checkpoints'), f"Experiment found, but checkpoints not found in gs2"
    elif os.path.exists(f'/mnt/disks/gs1/outputs/checkpoints/{expr_name}'):
        expr_path = f'/mnt/disks/gs1/outputs/checkpoints/{expr_name}'
        assert os.path.exists(f'{expr_path}/checkpoints'), f"Experiment found, but checkpoints not found in gs1"
    else:
        raise ValueError(f"Experiment {expr_name} not found in gs1 or gs2")
    return expr_path

def get_samples_path(expr_path, checkpoint):
    samples_path = f'{expr_path}/samples/{checkpoint}'
    if not os.path.exists(samples_path):
        print(f"Experiment found, but sample for {checkpoint} not found in gs2")
        return False
    else:
        return samples_path

def get_metric_path(samples_path, suffix):
    metricpath = [p for p in os.listdir(samples_path) if p.endswith(suffix+'-metrics.npy')]
    if len(metricpath) == 0:
        return False
    elif len(metricpath) > 1:
        import pdb; pdb.set_trace()
    else:
        return os.path.join(samples_path, metricpath[0])
    
cfg = 1.0
seed = 0
step = 50
nsmp = 10000
mode = 'rectflow'
folder_suffix = f"cfg-{cfg}-seed-{seed}-step-{step}-nsmp-{nsmp}-{mode if mode!='ddpm' else ''}"

data = {
    "DiT-S-latent": {
        'expr_name': "rectflow-lognorm/latent/ps1-DiT-S-2024-Aug-22-20-28-33",
        'latent': True
    }, 
    "DiT-B-latent": {
        'expr_name': "rectflow-lognorm/latent/ps1-DiT-B-2024-Aug-27-17-50-52",
        'latent': True
    },
    "DiT-L-latent": {
        'expr_name': "rectflow-lognorm/latent/ps1-DiT-L-2024-Sep-09-07-00-43",
        'latent': True
    },
    "DiT-XL-latent": {
        'expr_name': "dit/latent/ps1-DiT-XL-2024-Sep-20-21-30-58",
        'latent': True
    },
    "SSM-Ti-latent":{
        'expr_name':"rectflow-lognorm/latent/ps1-DiT-Ti-SSM-2024-Aug-25-00-03-43",
        'latent': True
    },
    "SSM-S-latent":{
        'expr_name': "rectflow-lognorm/latent/ps1-DiT-S-SSM-2024-Aug-22-20-24-12",
        'latent': True
    },
    "SSM-B-latent":{
        'expr_name': "rectflow-lognorm/latent/ps1-DiT-B-SSM-2024-Aug-30-08-23-39",
        'latent': True
    },
    "SSM-L-latent":{
        'expr_name': "dit-gradacc8/latent/ps1-SSM-L-2024-Sep-09-07-45-58",
        'latent': True
    },
    "SSM-XL-latent":{
        'expr_name': "ssm/latent/ps1-SSM-XL-2024-Sep-20-21-20-25",
        'latent': True
    },
    "TTT-Ti-pixel":{
        'expr_name': "ttt/pixels/ps2-TTT-Ti-lmbd-mlp-2024-Sep-04-20-30-18",
        'latent': False
    },
    "TTT-S-pixel":{
        'expr_name': "ttt/pixels/ps2-TTT-S-lmbd-mlp-2024-Sep-04-21-16-01",
        'latent': False
    },
    "TTT-B-pixel":{
        'expr_name':  "ttt-mlp-bd/pixels/ps2-DiT-B-mt3lm-mlp-bd-2024-Aug-31-06-25-38",
        'latent': False
    },
    "TTT-L-pixel":{
        'expr_name': "ttt/pixels/ps2-TTT-L-lmbd-mlp-2024-Sep-16-09-25-27",
        'latent': False
    },
}

for nickname in data.keys():
    lst = []
    expr_name = data[nickname]['expr_name']
    latent = data[nickname]['latent']
    print(f"Reading {nickname} ({expr_name})...")

    expr_path = get_expr_path(expr_name)
    for checkpoints in os.listdir(f'{expr_path}/samples/'):
        samples_path = get_samples_path(expr_path, checkpoints)
        metric_path = get_metric_path(samples_path, folder_suffix)
        if metric_path:
            metrics = np.load(metric_path, allow_pickle=True).item()
            if f'eval-{nsmp}-{step}/eval_steps' in metrics and f'eval-{nsmp}-{step}/train_flops_TFlops' in metrics and f'eval-{nsmp}-{step}/fid' in metrics:
                minilst = [
                    expr_name,
                    latent,
                    metrics[f'eval-{nsmp}-{step}/eval_steps'],
                    metrics[f'eval-{nsmp}-{step}/train_flops_TFlops'],
                    metrics[f'eval-{nsmp}-{step}/fid']
                ]
                lst.append(minilst)
            else:
                print(f"\tMetric is incomplete for {nickname} {checkpoints}")
        else:
            pass
            # print(f"\tMetric not found for {nickname} {checkpoints}")

    lst = sorted(lst, key=lambda x: x[2])
    data[nickname]=lst # overwrite dictionary with list of lists

print("\n\n")
with open(f"summary.json", 'w') as f:
    json.dump(data, f, indent=4)
print("Done saving summary.json")