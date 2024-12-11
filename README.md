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
│   ├── create_data  # Scripts for extracting features and saving images in a format readable by our defined dataloader, given downloaded image data.
│   ├── data_download# Script for downloading the ImageNet dataset (image size=256 or 512).
│   ├── run_train.sh # Script for launching training with user-defined parameters using train.py.
│   └── run_eval.sh  # Script for launching evaluation with user-defined parameters using eval.py.
```

## 0. Environment Setting and Data Preparation

### Environment installation

The necessary environment settings are stored in `environment-jax.yml`. To install the environment for GPU, run:
```shell
bash scripts/install_env_gpu.sh
```
If you are using TPU, run:
```shell
bash scripts/install_env_tpu.sh
```

### Dataset

You need to download the dataset to train the model. We use ImageNet (image size=256 or 512) and have already extracted the latent features (f=8) and saved them in TFRecord format. To download the TFRecords, run:
```shell
bash /scripts/data_download/download256.sh # image size: 256
bash /scripts/data_download/download.sh # image size: 512
```
This will save approximately 1K TFRecord files in a newly created `data` directory inside the current directory (or in the directory you define as `save_dir` inside `download256.sh`).


If you want to train on your own dataset, you need to create TFRecords for that dataset using scipts under `scripts/create_data`. Further description of this process will be updated soon.


## 1. Train the Model

Before running the training script, you need to modify the following contents inside `scripts/run_train.sh`:

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
*Note: `EXPR_NAME` will be identified by appending the model name and running time, so you don't need to be very specific about it*:

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
bash scripts/run_train.sh
```


## 2. Evaluate the Model

To be updated