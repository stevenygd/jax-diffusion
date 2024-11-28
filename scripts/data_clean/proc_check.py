import numpy as np
import os
import argparse
from datetime import datetime

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process some arguments.")
    
    # Add required arguments
    parser.add_argument('expr_name', type=str, help='The experiment name')
    parser.add_argument('checkpoint', type=int, help='The checkpoint number')
    parser.add_argument('--cfg', type=float, default=1.5, help='The config number')
    parser.add_argument('--seed', type=int, default=0, help='The seed number')
    parser.add_argument('--step', type=int, default=50, help='Number of sampling steps')
    parser.add_argument('--nsmp', type=int, default=10000, help='Number of samples')
    parser.add_argument('--mode', type=str, default='rectflow', help='The mode of the experiment')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Access the arguments
    expr_name = args.expr_name
    checkpoint = args.checkpoint
    samples_path = get_samples_path(expr_name, checkpoint)

    mode = args.mode
    folder_suffix = f"cfg-{args.cfg}-seed-{args.seed}-step-{args.step}-nsmp-{args.nsmp}-{args.mode if args.mode!='ddpm' else ''}"
    

    # print(f"fyi, there are these checkpoints {os.listdir(os.path.join(expr_path, 'samples'))}")
    dpath, dd = prcs(samples_path, folder_suffix)
    pngpath, npngs = get_pngs_path(samples_path, folder_suffix)
    print(f"{pngpath}")
    print(f"{len(dd)} iterations recorded, Found {npngs} pngs")
    print(f"\nnpz file found: {get_npz_path(samples_path, folder_suffix)}")
    print(f"\nmetrics file found: {get_metric_path(samples_path, folder_suffix)}\n")

    diffs = []
    for k, l in dd.items():
        start = datetime.strptime(l[1], '%Y-%m-%d %H:%M:%S')
        if len(l)>2:
            end = datetime.strptime(l[2], '%Y-%m-%d %H:%M:%S')
            diff = end - start
            diffs.append(diff.seconds)
            # print(f"{k}: {l[0]} {start.strftime('%H:%M:%S')} ~ {end.strftime('%H:%M:%S')} ({diff})")
        else:
            print(f"{k}: {l[0]} {start.strftime('%H:%M:%S')} ~ ? ")
    if len(diffs):
        print(f"Average time: {np.mean(diffs)/60:.2f} minutes")


def get_checkpoint_path(expr_name):
    if os.path.exists(f'/mnt/disks/gs2/outputs/checkpoints/{expr_name}'):
        expr_path = f'/mnt/disks/gs2/outputs/checkpoints/{expr_name}'
        checkpoint_path = f'{expr_path}/checkpoints'
        assert os.path.exists(checkpoint_path), f"Experiment found, but checkpoints not found in gs2"
    elif os.path.exists(f'/mnt/disks/gs1/outputs/checkpoints/{expr_name}'):
        expr_path = f'/mnt/disks/gs2/outputs/checkpoints/{expr_name}'
        checkpoint_path = f'{expr_path}/checkpoints'
        assert os.path.exists(checkpoint_path), f"Experiment found, but checkpoints not found in gs1"
    else:
        raise ValueError(f"Experiment {expr_name} not found in gs1 or gs2")
    return checkpoint_path

def get_samples_path(expr_name, checkpoint):
    if os.path.exists(f'/mnt/disks/gs2/outputs/checkpoints/{expr_name}'):
        expr_path = f'/mnt/disks/gs2/outputs/checkpoints/{expr_name}'
        samples_path = f'{expr_path}/samples/{checkpoint}'
        if not os.path.exists(samples_path):
            print(f"Experiment found, but sample for {checkpoint} not found in gs2")
            return False
    elif os.path.exists(f'/mnt/disks/gs1/outputs/checkpoints/{expr_name}'):
        expr_path = f'/mnt/disks/gs1/outputs/checkpoints/{expr_name}'
        samples_path = f'{expr_path}/samples/{checkpoint}'
        if not os.path.exists(samples_path):
            print(f"Experiment found, but sample for {checkpoint} not found in gs1")
            return False
    else:
        raise ValueError(f"Experiment {expr_name} not found in gs1 or gs2")
    return samples_path

def read(dpath):
    dd = np.load(dpath, allow_pickle=True).item()
    return dd

def save(dpath, dd):
    np.save(dpath, dd)
    print(f"Saved to {dpath}")

def prcs(samples_path, suffix):
    dpath = [p for p in os.listdir(samples_path) if (p.endswith('process.npy') and suffix in p)]
    if len(dpath) == 0:
        print(f"No process.npy that matches with {suffix} found inside {samples_path}.") 
    elif len(dpath) > 1:
        import pdb; pdb.set_trace()
        dpath = os.path.join(samples_path, [p for p in dpath if suffix in p][0])
    else:
        import pdb; pdb.set_trace()
        dd = read(os.path.join(samples_path,dpath[0]))

    return os.path.join(samples_path,dpath[0]), dd

def get_pngs_path(samples_path, suffix):
    pngpath = [p for p in os.listdir(samples_path) if (os.path.isdir(os.path.join(samples_path, p)) and suffix in p)]
    if len(pngpath) == 0:
        print(f"No dir found inside {samples_path}. Try ls {samples_path}")
    elif len(pngpath) > 1:
        import pdb; pdb.set_trace()
        pngpath = os.path.join(samples_path, [p for p in pngpath if suffix in p][0])
    else:
        pngpath = os.path.join(samples_path, pngpath[0])
        
    return pngpath, len(os.listdir(pngpath))

def get_npz_path(samples_path, suffix):
    npzpath = [p for p in os.listdir(samples_path) if p.endswith(suffix+'.npz')]
    if len(npzpath) == 0:
        return False
    elif len(npzpath) > 1:
        import pdb; pdb.set_trace()
    else:
        return os.path.join(samples_path, npzpath[0])

def get_metric_path(samples_path, suffix):
    metricpath = [p for p in os.listdir(samples_path) if p.endswith(suffix+'-metrics.npy')]
    if len(metricpath) == 0:
        return False
    elif len(metricpath) > 1:
        import pdb; pdb.set_trace()
    else:
        return os.path.join(samples_path, metricpath[0])
    

if __name__ == '__main__':
    main()