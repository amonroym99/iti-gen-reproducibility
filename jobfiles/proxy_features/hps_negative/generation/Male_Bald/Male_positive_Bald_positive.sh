#!/bin/bash

python models/sd/scripts/txt2img.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --prompt="a headshot of a bald man" \
    --negative_prompt="" \
    --outdir="results/proxy_features/hps_negative/Male_Bald/Male_positive_Bald_positive" \
    --skip_grid \
    --n_iter=1 \
    --n_samples=8 \
    --seed=0
