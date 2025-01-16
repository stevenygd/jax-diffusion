#!/bin/bash

# Install python dependency
pip install -U "jax[cuda12]==4.3.31"
pip install --upgrade tensorflow-cpu
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