import os
import glob
from PIL import Image
from tqdm import tqdm
BASE_DIR = "/mnt/disks/gs/outputs/checkpoints"
def get_sample_folder_dir(expr_name, ckpt):
    sample_folders_dir = os.path.join(BASE_DIR, expr_name, 'samples', ckpt)
    sample_folders = [f for f in os.listdir(sample_folders_dir) if os.path.isdir(os.path.join(sample_folders_dir, f))]
    print(f"{len(sample_folders)} sampling happened here:\n\t", *sample_folders)
    sample_folder_dir = os.path.join(sample_folders_dir, sample_folders[0])
    
    assert os.path.isdir(sample_folder_dir)

    return sample_folder_dir
    

def nsmp_check():
    expr_names=[
        "dit-s/1-DiT-S-2024-Jul-23-18-40-48",
        # "dit-s/2-DiT-S-2024-Jul-23-18-39-14",
        # "dit-s-vae/p4-DiT-S-2024-Aug-04-08-13-11",
        # "fixloader-sa/l1-DiT-S-2024-Aug-05-05-50-41",
        # "fixloader-linattn/l1-DiT-S-linattn-2024-Aug-05-06-21-18",
        # "fixloader-vae/p4-DiT-S-linattn-2024-Aug-05-07-16-40",
        # "fixloader-linattn-vit/p4-DiT-S-linattn-2024-Aug-05-07-03-17",
        # "fixloader-vit/p4-DiT-S-2024-Aug-05-05-37-31",
        # "fixloader-vae/p4-DiT-S-2024-Aug-05-05-19-07",
        # "fixloader-sa/l2-DiT-S-2024-Aug-05-17-52-16",
        # "ssm/l1-DiT-S-SSM-2024-Aug-06-08-16-16",
        # "ssm/l2-DiT-S-SSM-2024-Aug-06-08-13-35",
        # "ssm-vit/p4-DiT-S-SSM-2024-Aug-06-09-16-26",
        # "mt3-mlp-niters4-ilr1.0/l1-DiT-S-mt3mlp-v0-2024-Aug-06-20-34-17",
        # "SA/l1-DiT-S-2024-Aug-08-08-00-21",
        # "SA/l2-DiT-S-2024-Aug-08-06-14-51",
        # "ssm/l1-DiT-S-SSM-2024-Aug-08-08-27-51",
        # "SA/l2-DiT-B-2024-Aug-08-17-00-20",
        # "ssm/p1-DiT-B-SSM-2024-Aug-09-06-43-51",
        # "SA/l1-DiT-B-2024-Aug-08-19-49-24",
    ]

    for expr in expr_names:
        samples_dir = os.path.join(BASE_DIR, expr, 'samples')
        if not(os.path.exists(samples_dir)):
            print(f"Missing samples folder for {expr}")
        else:
            print(f"Expr: {expr}")
            for ckpt in os.listdir(samples_dir):
                # if ckpt!="999999":
                #     continue
                ckpt_dir = os.path.join(samples_dir, ckpt)
                for results in os.listdir(ckpt_dir):
                    sample_dir = os.path.join(ckpt_dir, results)
                    if os.path.isdir(sample_dir):
                        if not("seed-0" in sample_dir):
                            continue
                        nsmp = results[results.index("nsmp"):results.index("nsmp")+10]
                        target_nsmp = int(nsmp.split("-")[1])
                        actual_nsmp = len(os.listdir(sample_dir))
                        ddim = "(ddim)" if "ddim" in results else ""
                        warning = "" if target_nsmp < actual_nsmp else "UNDERSAMPLED"
                        if ddim and target_nsmp == 10000:
                            print(f"\tckpt-{ckpt}, {nsmp}{ddim}, {actual_nsmp} {warning}")


def remove_bad_images():
    expr_name = 'udit/pixels/ps2-DiT-S-2024-Aug-18-09-11-35'

    checkpoints = os.listdir(os.path.join(BASE_DIR, expr_name, 'checkpoints'))
    print (checkpoints)

    for ckpt in checkpoints:
        if ckpt not in ["40000"]:
            continue
        sample_dir = get_sample_folder_dir(expr_name, ckpt)
        print(f"Removing bad samples in: {sample_dir}...", end="\t")

        flst = glob.glob(f"{sample_dir}/*.png")
        print(f"(total {len(flst)} samples found)")
        deleted = 0
        for i in tqdm(range(len(flst))):
            try:
                sample_pil = Image.open(flst[i])
                # print(f"success: {flst[i][-10:]}")
            except Exception as e:
                # print(f"error: {flst[i][-10:]}: {e}")
                # import pdb; pdb.set_trace()
                os.remove(flst[i])
                deleted += 1
            if i%2000==0:
                print(f'{expr_name[-5:]}-{ckpt}, {i+1} (deleted: {deleted} so far)')
        print(f"Deleted {deleted} images in {sample_dir}")

if __name__ == "__main__":
    # nsmp_check()
    remove_bad_images()
    # get_sample_folder_dir("ssm/pixels/ps4_e2-DiT-S-SSM-2024-Aug-16-07-34-38", "600000")