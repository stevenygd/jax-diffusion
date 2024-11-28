import numpy as np
import os
EXPR_NAME1 = 'dit-s/1-DiT-S-2024-Jul-23-18-40-48'
EXPR_NAME2 = 'dit-s/2-DiT-S-2024-Jul-23-18-39-14'
EXPR_NAME3 = 'dit-b/2-DiT-B-2024-Jul-24-18-18-03'

expr_name = EXPR_NAME3
expr_mode = expr_name[8:13]
if expr_name == EXPR_NAME1: last_ckpt = 999999
elif expr_name == EXPR_NAME2: last_ckpt = 3999999
elif expr_name == EXPR_NAME3: last_ckpt = 800000
else: raise ValueError(f"expr_name: {expr_name} is not supported")

print(f"expr_name: {expr_name} [last_ckpt: {last_ckpt}]")
print("FID")
file_dir = f"/mnt/disks/gs/outputs/checkpoints/{expr_name}/samples/{last_ckpt}"
seeds = [0, 3877, 2168, 6980]
# seeds = [0, 3877, 2168, 6980]


for step, mode in [(128, 'ddpm'), (128, 'ddim'), (50, 'ddim')]:
    metric = []
    time = 0
    valid_time_cnt = 0
    print(f"====={mode}({step})=====")
    for seed in seeds:
        if mode == 'ddpm': 
            file_name = os.path.join(file_dir, f'{expr_mode}-size-256-cfg-1.5-seed-{seed}-step-{step}-nsmp-10000-metrics.npy')
        else: 
            file_name = os.path.join(file_dir, f'{expr_mode}-size-256-cfg-1.5-seed-{seed}-step-{step}-nsmp-10000-ddim-metrics.npy')
        # assert os.path.exists(file_name), f"file_name: {file_name} does not exist"

        if not os.path.exists(file_name):
            # print(f"\nfile_name: {file_name} does not exist")
            print(f'\n!!  {mode}({step})-seed{seed} does not exist')
        else:
            metrics = np.load(file_name, allow_pickle=True)
            print((metrics.item())[f'eval-10000-{step}/fid'], end=", ")
            metric.append((metrics.item())[f'eval-10000-{step}/fid'])
            if ('sampling_time' in metrics.item()) and (metrics.item())['sampling_time']>0:
                time += (metrics.item())['sampling_time']
                valid_time_cnt += 1

    metric = np.array(metric)
    print(f"\nmean: {np.mean(metric):.4f}, std: {np.std(metric):.4f}")
    if valid_time_cnt>0:
        print(f"(average sampling time: {(time/valid_time_cnt)/60:.2f} min)")
    print()