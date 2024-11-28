# Loading imagenet to create a balanced reference set
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
from PIL import Image
from torchvision import transforms
from lovely_numpy import lo


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



if __name__ == "__main__":
    # Use argparse to load image net folder and number of reference
    parser = argparse.ArgumentParser(description='Create a reference set for evaluation')
    parser.add_argument(
        '--imagenet_path', type=str, 
        default='/home/guandaoyang/jax-DiT/data/imagenet/train', 
        help='Path to imagenet folder')
    parser.add_argument(
        '--output_prefix', type=str, 
        default='/mnt/gs/sci-guandao/data/imagenet256_trainset_balanced', 
        help='Path to imagenet folder')
    parser.add_argument(
        '--num_ref', type=int, default=10000, 
        help='Number of reference images')
    parser.add_argument(
        '--image_size', type=int, default=256, 
        help='Images resolution')
    args = parser.parse_args()
    
    # Load number of classes 
    classes = os.listdir(args.imagenet_path)
    num_images_per_cls = args.num_ref // len(classes)
    assert args.num_ref % len(classes) == 0, "Number of reference images should be divisible by number of classes"
   
    images_out = [] 
    for cls in tqdm(classes):
        cls_path = osp.join(args.imagenet_path, cls)
        cls_imgs = glob.glob(osp.join(cls_path, '*.JPEG'))
        # Randomly sampled images from [cls_imgs]
        cls_smps = np.random.choice(
            cls_imgs, num_images_per_cls, replace=False)
        images_out += cls_smps.tolist() 
        
    # Create a python dataloader for the reference image
    # Apply central crop and resize to 256x256
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ]) 
    data_out = []
    for img_path in tqdm(images_out):
        img = Image.open(img_path).convert("RGB")
        img = np.array(transform(img)).transpose(1,2,0)
        img = ((img + 1) * 0.5 * 255).astype(np.uint8)
        data_out.append(img)
    data_out = np.stack(data_out, axis=0)
    print(f"Generated samples: {data_out.shape} ")
    
    output_path = f'{args.output_prefix}_{args.num_ref//1000}k.npz' 
    np.savez(output_path, arr_0=data_out)
    print(f"Saved to {output_path}")