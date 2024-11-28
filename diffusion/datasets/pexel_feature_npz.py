import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import glob
import jax
import jax.numpy as jnp
import numpy as np
from diffusers.models import FlaxAutoencoderKL

TARGET_SIZE = 3840

images_dir = '/mnt/disks/pexel-bucket/dogcat'
# image_list_file = '/mnt/disks/pexel-bucket/image_files.txt'
image_list_file = '/mnt/disks/pexel-bucket/dogcat/image_dirs.txt'

# image_paths = glob.glob(os.path.join(images_dir, '*/*.jpg'))
# with open(image_list_file, 'w') as f:
#     for path in image_paths:
#         f.write(f"{path}\n")

features_dir = '/mnt/disks/pexel-bucket/dogcat/features/cat'
tmp_dir = '/mnt/disks/pexel-bucket/dogcat/tmp'
os.makedirs(tmp_dir, exist_ok=True)

vae_dir = os.path.join('/mnt/disks/gs/sci-guandao', 'vae')
vae_config = np.load(os.path.join(vae_dir, 'config.npy'), allow_pickle=True).item()
vae_params = np.load(os.path.join(vae_dir, 'params.npy'), allow_pickle=True).item()
vae = FlaxAutoencoderKL.from_config(vae_config)

n_devices = jax.local_device_count()
print(f"Number of devices: {n_devices}")

@jax.jit
def encode(rng, x):
    rng = jax.random.fold_in(rng, jax.process_index())
    x = x.transpose((0,3,1,2)) # (B, 256, 256, 3) -> (B, 3, 256, 256)
    return vae.apply(
        {"params": vae_params},
        x, deterministic=False, rngs={"gaussian": rng},
        method=vae.encode
    ).latent_dist.sample(rng).transpose((0, 3, 1, 2)) * 0.18215
encode = jax.pmap(encode, axis_name="devices")

@jax.jit
def decode(rng, z):
    rng = jax.random.fold_in(rng, jax.process_index())
    img = vae.apply(
        {"params": vae_params},
        z / 0.18215,
        method=vae.decode
        ).sample
    img = jnp.clip(127.5 * img + 128.0, min=0, max=255).transpose((0, 2, 3, 1))
    return img
# decode = jax.pmap(decode, axis_name="devices")


def crop_multiple_of(pilimage, f=8):
    H, W = pilimage.size
    h = H // f * f
    w = W // f * f
    
    h_margin = (H - h) // 2
    w_margin = (W - w) // 2

    arr = np.array(pilimage)
    arr = arr[h_margin:h_margin+h, w_margin:w_margin+w]
    assert arr.shape[0] % f == 0 and arr.shape[1] % f == 0, f"Image shape: {arr.shape}"
    return arr

def crop_center(pil_image, target_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * target_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = target_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - target_size) // 2
    crop_x = (arr.shape[1] - target_size) // 2
    return arr[crop_y: crop_y + target_size, crop_x: crop_x + target_size]

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        if Exception=="FileNotFoundError":
            print(f"! File not found {image_path}")
        else:
            print(f"! Error opening {image_path}, removing...")
            os.remove(image_path)
        return None
    # image = crop_multiple_of(image)           # (4000, 4000, 3)
    image = crop_center(image, TARGET_SIZE)            # 3200(o) -> 3600(o) -> 4000(o) -> 4400(o) -> 4800(o) -> 4960(x) -> 5000(x)
    x = (image.astype(np.float32)/255)*2.0 - 1.0
    return x


def get_image_status(image_path):
    image_name = os.path.basename(image_path)
    feature_path = os.path.join(features_dir, image_name.replace('.jpg', '.npz'))
    tmp_path = os.path.join(tmp_dir, image_name.replace('.jpg', '.tmp'))

    if os.path.exists(feature_path):
        return False, f"! Already extracted {feature_path}"

    if os.path.exists(tmp_path):
        return False, f"! Already processing {tmp_path}"
    
    else:
        open(tmp_path, 'a').close()

    return True, "Stacking and Processing..."

def process_image_batch(image_paths, images):
    
    x = np.stack(images, axis=0)                            # (4, 1536, 1536, 3)
    x = jnp.array(x).reshape(n_devices, -1, *x.shape[-3:])  # (4, 1, 1536, 1536, 3)

    rng = jax.random.PRNGKey(42)
    image_ids = [int(image_path.split('/')[-1].split('.')[0]) for image_path in image_paths]
    # rng_ = jax.random.fold_in(rng, image_id)
    rng_ = jnp.array([jax.random.fold_in(rng, image_id) for image_id in image_ids])
    try: 
        features = encode(rng_, x)   # (dev=4, B/dev=1, 4, h/8, w/8) 
        features = features.reshape(-1,1, *features.shape[2:]) # (B, 4, h/8, w/8)
        feature_paths = []
        for i, image_path in enumerate(image_paths):
            image_name = os.path.basename(image_path)
            feature_path = os.path.join(features_dir, image_name.replace('.jpg', '.npz'))
            np.savez_compressed(feature_path, features[i])
            feature_paths.append(os.path.basename(feature_path))
    except Exception as e:
        print(f"\n\n{e}, removing tmp files...")
        remove_tmp_files(image_paths)
        raise e

    return feature_paths

def remove_tmp_files(image_paths):
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        tmp_path = os.path.join(tmp_dir, image_name.replace('.jpg', '.tmp'))
        os.remove(tmp_path)

def decode_check(image_id):
    feature_path = os.path.join(features_dir, f'{image_id}.npz')
    feature = np.load(feature_path)['arr_0']        # (1, 4, 192, 192)
    z = jnp.array(feature)
    print(f"Feature shape: {z.shape}")
    rng = jax.random.PRNGKey(42)
    x_r = decode(rng, z)                            # (1, 1536 ,1536, 3)

    image_r = Image.fromarray(np.array(x_r[0]).astype(np.uint8)) # (1536, 1536, 3)

    recon_image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(feature_path))), 'samples',
                                    os.path.basename(feature_path).replace('.npz', '-recon.jpg'))
    image_r.save(recon_image_path)
    print(f"Decoded image saved at {recon_image_path}")

def main():
    # decode_check('00006260')
    # decode_check('00007517')
    # decode_check('00004602')
    # return
    with open(image_list_file, 'r') as f:
        image_paths = f.read().splitlines()

    pid = os.getpid()
    print("Shuffling images with PID...")
    np.random.seed(pid)
    np.random.shuffle(image_paths)

    pbar = tqdm(image_paths)
    image_paths = []
    images = []
    for image_path in pbar:
        status, message = get_image_status(image_path)
        if status:  
            x = preprocess_image(image_path)
            if x is None:
                continue
            image_paths.append(image_path)   
            images.append(x)
    
            pbar.set_description(f"{[ip.split('/')[-1] for ip in image_paths]} {message}")
            if len(image_paths) == n_devices:
                feature_paths = process_image_batch(image_paths, images)
                remove_tmp_files(image_paths)
                # for i, image_path in enumerate(image_paths):
                #     image_name = os.path.basename(image_path)
                    # Image.fromarray((255*((images[i]+1)/2)).astype(np.uint8)).save(os.path.join(features_dir, image_name.replace('.jpg', '-orig.jpg')))
                    # decode_check(image_name.split('.')[0])
                image_paths = []
                images = []
                pbar.set_description(f"{feature_paths} Success!")  
            

        else:
            pbar.set_description(f"{[ip.split('/')[-1] for ip in image_paths]} ({image_path.split('/')[-1]}: {message})")
        

if __name__ == '__main__':
    main()