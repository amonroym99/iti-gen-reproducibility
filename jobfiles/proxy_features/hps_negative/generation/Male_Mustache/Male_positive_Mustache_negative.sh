#!/bin/bash

python models/sd/scripts/txt2img.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --prompt="a headshot of a man" \
    --negative_prompt="mustache" \
    --outdir="results/proxy_features/hps_negative/Male_Mustache/Male_positive_Mustache_negative" \
    --skip_grid \
    --n_iter=1 \
    --n_samples=8 \
    --seed=6
