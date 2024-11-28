import hydra
from omegaconf import DictConfig
import os
import os.path as osp
import jax
import jax.numpy as jnp
from jax.lib import xla_bridge
import numpy as np
from diffusion.utils import train_utils
from diffusion.utils import sharding 
from jax.sharding import NamedSharding
import json
import orbax.checkpoint as ocp

from diffusion.configs_mlc import CFG
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("resume_config", None, "Resume config.")
flags.DEFINE_integer("resume_step", None, "Resume step.")
flags.DEFINE_string("output_dir", None, "Output directory.")
flags.DEFINE_string("class_dict", None, "Class dictionary.")
flags.DEFINE_integer("global_seed", 0, "Global seed.")
flags.DEFINE_boolean("multi_process", False, "Multi process.")
flags.DEFINE_integer("image_size", 256, "Image size.")


def json_to_dict(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

def main(_):
    if FLAGS.multi_process:
        jax.distributed.initialize()
    
    assert FLAGS.resume_config is not None, "Please provide resume config."
    with open(FLAGS.resume_config, "r") as f:
        config_dict = json.load(f)

    args = CFG(**config_dict)
    jax_platform = xla_bridge.get_backend().platform.lower()
    assert jax_platform in ["tpu", "gpu"], f"Jax not using GPU:{jax_platform}"

    rng = jax.random.PRNGKey(FLAGS.global_seed)
    rng = jax.random.fold_in(rng, jax.process_index())

    checkpoint_dir = os.path.join(os.path.dirname(FLAGS.resume_config), 'ckpt')
    assert os.path.isdir(checkpoint_dir), f"Checkpoint directory {checkpoint_dir} does not exist."
    assert os.path.isdir(os.path.join(checkpoint_dir, str(FLAGS.resume_step))), f"{FLAGS.resume_step} does not exist in {checkpoint_dir}"
    
    # Set-up data
    use_latent = args.use_latent
    if hasattr(args, "image_size"): 
        if use_latent:
            latent_size = args.image_size // 8 
        else:
            latent_size = args.image_size
    else: 
        _, _, _, latent_size, _ = args.data_shape.x
    if hasattr(args, "latent_dim"): 
        latent_dim = args.latent_dim
    else: 
        _, _, latent_dim, _, _ = args.data_shape.x

    args.dp_dim=-1
    mesh = sharding.get_mesh(args)
    with mesh:
        print("Shading mesh:", mesh)

        data = (
            # x
            jnp.ones((
                4, latent_dim, latent_size, latent_size)),
            # y
            jnp.ones((4,), dtype=jnp.int32)
        )
        # Create train state
        rng, spl = jax.random.split(rng)
        ema_state, ema_state_sharding, model = train_utils.create_train_state(
            args, mesh, spl, data[0], data[1])

    options = ocp.CheckpointManagerOptions()
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir, orbax_checkpointer, options)
    
    ema_state, train_step = train_utils.restore_checkpoint(
        checkpoint_manager, ema_state, 
        logger=None, resume_step=FLAGS.resume_step)
    params = ema_state.ema_params
    # model = model.bind(params)
    model_b = model.bind(params)

    class_dict = json_to_dict(FLAGS.class_dict)
    y = jnp.array([int(key) for key in class_dict["dog_classes"].keys()])
    dog_embedding= jnp.mean(model.y_embedder.apply({'params':params['params']['y_embedder']},y,False), axis=0)
    print(f'Dog embedding shape: {dog_embedding.shape}')

    y = jnp.array([int(key) for key in class_dict["cat_classes"].keys()])
    cat_embedding= jnp.mean(model.y_embedder.apply({'params':params['params']['y_embedder']},y,False), axis=0)
    print(f'Cat embedding shape: {cat_embedding.shape}')

    np.savez(osp.join(FLAGS.output_dir, "ref_embedding"), dog=dog_embedding, cat=cat_embedding)


@hydra.main(version_base=None, config_path=".", config_name="config")
def hydra_main(args: DictConfig):
    """Run sampling. """
    if args.multi_process:
        jax.distributed.initialize()
    print("Devices", jax.devices())
    # setup_tpu_metrics()
    rng = jax.random.PRNGKey(args.global_seed)
    rng = jax.random.fold_in(rng, jax.process_index())

    assert args.resume_step > 0, f"Something wrong in lauch eval script, resume_step = {args.resume_step}"
    assert str(args.image_size) in args.inference.ref_batch, f"Ref batch {args.inference.ref_batch} doesn't match image size {args.image_size}"
    
    # Resume configuration
    assert args.resume is not None
    experiment_dir = args.resume
    # Stores saved model checkpoints
    checkpoint_dir = osp.join(f"{experiment_dir}", "checkpoints") 
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir)
    assert os.path.isdir(checkpoint_dir), f"Checkpoint directory {checkpoint_dir} does not exist."
    assert os.path.isdir(os.path.join(checkpoint_dir, f'{args.resume_step}')), \
        f"Desired step does not exist. Please check if there is a checkpoint file under {os.path.join(checkpoint_dir, f'{args.resume_step}')}"

    use_latent = args.get("use_latent", True)
    if hasattr(args, "image_size"): latent_size = args.image_size // 8 if use_latent else args.image_size
    else: _, _, _, latent_size, _ = args["data_shape"]["x"]

    if hasattr(args, "latent_dim"): latent_dim = args.latent_dim
    else: _, _, latent_dim, _, _ = args["data_shape"]["x"]


    data = (
        jnp.ones((1, latent_dim, latent_size, latent_size)),   # x
        jnp.ones((1,), dtype=jnp.int32)                        # y
    )
    rng, spl = jax.random.split(rng) 
    
    model = train_utils.create_model(args)
    # tx = train_utils.make_opt(args)
    spl1, spl2, spl3, spl4 = jax.random.split(spl, 4)

    x = np.array(data[0]).reshape(-1, *data[0].shape[-3:])[:1]
    y = np.array(data[1])[:1]
    t = jnp.ones(shape=y.shape, dtype=jnp.float32)
    params = model.init(
        {'params': spl1, 'dropout': spl2, "label_emb": spl3, "mt3": spl4},
        x, t, y, training=False)
    model = model.bind(params)

    class_dict = json_to_dict(args.class_dict)
    y = jnp.array([int(key) for key in class_dict["dog_classes"].keys()])
    dog_embedding= jnp.mean(model.y_embedder(y,False), axis=0)
    print(f'Dog embedding shape: {dog_embedding.shape}')

    y = jnp.array([int(key) for key in class_dict["cat_classes"].keys()])
    cat_embedding= jnp.mean(model.y_embedder(y,False), axis=0)
    print(f'Cat embedding shape: {cat_embedding.shape}')

    np.savez(osp.join(args.output_dir, "ref_embedding.npy"), dog=dog_embedding, cat=cat_embedding)
        
if __name__ == "__main__":
    # hydra_main()
    app.run(main)
    '''
    run with:
    python y_embed.py \
    --resume_config=${BUCKET_MNT}/xm/checkpoints/test-DiT-L-2024-Nov-07-22-47-32/config.json \
    --resume_step=399999 \
    --output_dir=${OUTPUT_DIR} \
    --class_dict=${OUTPUT_DIR}/imagenet_labels.json 
    '''