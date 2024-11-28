from diffusion.configs_mlc import update


def get_config(base_cfg, dataset_type):
    match dataset_type:
        case "imagenet256_pixels":
            return update(
                base_cfg,
                data_loader="imagenet_pixel_tfds",
                num_workers=4,
                image_size=256,
                feature_path="imagenet256_pixels_tfdata_sharded/",
                num_classes=1000,
                latent_dim=3,
                use_latent=False,
                scalar=1.,
                global_batch_size=256,
            )
        case "imagenet256_latent_f8":
            return update(
                base_cfg,
                data_loader="imagenet_feature_tfds",
                num_workers=4,
                image_size=256,
                feature_path="imagenet256_flax_tfdata_sharded/",
                num_classes=1000,
                latent_dim=4,
                use_latent=True,
                scalar=1.,
                global_batch_size=256,
            ) 
        case _:
            raise NotImplemented