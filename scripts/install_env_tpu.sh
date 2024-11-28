#!/bin/bash

# Install python dependency
pip install \
  --upgrade 'jax[tpu]==0.4.31' \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --upgrade tensorflow
pip install --upgrade tensorflow-datasets
pip install --upgrade hydra-core
pip install --upgrade wandb
pip install --upgrade plotly matplotlib
pip install --upgrade flax
pip install --upgrade optax
pip install orbax==0.1.9
pip install orbax-checkpoint==0.5.20
pip install --upgrade chex
pip install --upgrade diffusers["flax"] transformers
pip install einops
pip install jaxtyping