# JAX-Diffusion

This repository provides training for image generation models based on JAX and evaluates the trained checkpoints. Below are brief explanations of each directory.

```python
jax-diffusion
├── diffusion
│   ├── configs    # Contains Hydra configuration files for model definition, loss functions, inference settings, and optimizers.
│   ├── datasets   # dataset class definitions
│   ├── evaluation # code for evaluating checkpoints
│   ├── losses     # diffusion utility functions for managing denoising steps.
│   ├── models     # Defines models and sub-layers that constitute the model.
│   └── utils      # utility functions for FLOP computations and other training utilities.
├── scripts
│   ├── data_prep  # Scripts for downloading dataset and extracting features and saving images in a format readable by our defined dataloader, given downloaded image data.
```

## 0. Environment Setting and Data Preparation

### Environment installation

Create envorionment by running:
```shell
conda create -n jax-diff
conda activate jax-diff
```

The necessary environment settings are stored in `environment-jax.yml`. To install the environment for GPU, run:
```shell
bash scripts/install_env_gpu.sh
```
If you are using TPU, run:
```shell
bash scripts/install_env_tpu.sh
```

### VAE
The models are being trained on latent features of ImageNet, which means that the data has already gone through the 'encoder' part of a variational autoencoder (VAE). We are using the pretrained weights of a well-performing VAE. Therefore, during inference, the model should be able to decode the generated latent code into image space. We provide the configuration and weights for the VAE that fit this model. Download these from [this link](https://drive.google.com/drive/folders/19UKnB0x9-cHoGRlrnIN7-8iakp_9EWwh) to your local directory.

The directory will include the following two files: `config.npy`(model configuration) and `params.npy`(pretrained weights).

### Dataset

You need to download the dataset to train the model. We use ImageNet (image size=256 or 512) and have already extracted the latent features (f=8) and saved them in TFRecord format. To download the TFRecords, run:
```shell
bash scripts/data_prep/download256.sh # image size: 256
bash scripts/data_prep/download512.sh # image size: 512
```
This will save approximately 1K TFRecord files in a newly created `data` directory inside the current directory (or in the directory you define as `save_dir` inside `download256.sh`).


If you want to train on your own dataset, you need to create TFRecords for that dataset using scipts under `scripts/datap_prep/extract_features_jax.py`. Run:
```
python scripts/data_prep/extract_features_jax.py --data-dir=dir/to/image/folder --vae=dir/to/vae
```




## 1. Train the Model

Before running the training script, you need to modify the following contents inside `scripts/run_train_gpu.sh`(or `scripts/run_train_tpu.sh` if using TPU):

**1-1. WandB Configuration** : We use WandB for logging the training loss and evaluation results. You will need the WandB project name and API key. Set these as:
```bash
export WANDB_API_KEY="2weq2934u..."
export WANDB_PROJECT="jax-diffusion"
export WANDB_TEAM=
...
```
**1-2. Directory of the Codebase, output and dataset** : Set the directory of this cloned repository(`CODE_DIR`), directory for output(`OUT_DIR`), such as model checkpoints that are generated as training proceeds, and directory storing the TFRecords(`DATA_DIR`) from previous data preparation Step:
```bash
CODE_DIR=../jax-diffusion
OUT_DIR=../output
DATA_DIR=../data
```
**1-3. Select Model Name and Name for Current Run** : Specify the model name and a name for the current run. We currently provide two types of models in two different sizes, so the model name should be one of the following four.

```bash
MODEL_NAME=ssm_xl  # should be one of the following: dit_l, dit_xl, ssm_l, ssm_xl
EXPR_NAME=my_train
```
*Note: `EXPR_NAME` will be identified by appending the model name and running time(e.g. `my_train-DiT-XL-2024-Dec-11-04-02-15`), so you don't need to worry too much about the naming.*:

**1-4. Change other train configuration** : 
By default, train configuration is called with the default parameters specified inside `diffusion/configs/config.yaml`. If you want to modify specific parameters, feel free to change them to match your usage below line `--config-name`. For example:

```bash
  --config-name config \
    ...
    global_batch_size=16 \
    ckpt_every=50 \
    ...
```


After setting all these configurations, finally run:
```bash
bash scripts/run_train_gpu.sh
```
or 
```bash
bash scripts/run_train_tpu.sh
```

## 2. Evaluate the Model


Before running the testing script, you need to modify the following contents inside `scripts/run_train_gpu.sh`(`scripts/run_train_tpu.sh` if using TPU) as we did for training:

**1-1. WandB Configuration** : We use WandB for logging the training loss and evaluation results. You will need the WandB project name and API key. Set these as:
```bash
export WANDB_API_KEY="2weq2934u..."
export WANDB_PROJECT="jax-diffusion"
export WANDB_TEAM=
...
```
**1-2. Directory of the Codebase, output and dataset** : `CODE_DIR` and `OUT_DIR` should be the same as what you defined in the training script. Set the VAE directory (`VAE_DIR`) to the location where the pretrained weights of the autoencoder are stored.:
```bash
CODE_DIR=../jax-diffusion
OUT_DIR=../output
VAE_DIR=../vae
```
**1-3. Select Model Name and Name for Current Run** : Specify the experiment name, inclusive of the model name and running time that was automatically set during training. If you want to evaluate a specific single checkpoint, define that at `RESUME_CHECKPOINT`. If you want to evaluate all existing checkpoints, set `RESUME_CHECKPOINT` equal to -1.

```bash
EXPR_NAME=my_train-DiT-XL-2024-Dec-11-04-02-15
RESUME_CHECKPOINT=-1
```

**1-4. Change other evaluation configuration** : 
By default, evaluation configuration is specified inside `diffusion/configs/config.yaml` when we first train the model. If you want to modify specific parameters, feel free to change them to match your usage below line `--config-path`. For example:

```bash
  --config-path . \
    ...
    inference.per_proc_batch_size=32 \
    inference.num_fid_samples=50000
    ...
```

After setting all these configurations, finally run:
```bash
bash scripts/run_eval_gpu.sh
```
or
```bash
bash scripts/run_eval_tpu.sh
```