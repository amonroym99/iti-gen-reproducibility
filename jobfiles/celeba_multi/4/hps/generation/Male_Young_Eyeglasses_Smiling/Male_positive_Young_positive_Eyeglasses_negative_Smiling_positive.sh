#!/bin/bash

python models/sd/scripts/txt2img.py \
    --config='models/sd/configs/stable-diffusion/v1-inference.yaml' \
    --ckpt='models/sd/models/ldm/stable-diffusion-v1/model.ckpt' \
    --plms \
    --prompt="a headshot of a smiling old man without eyeglasses" \
    --outdir="results/celeba_multi/4/hps/Male_Young_Eyeglasses_Smiling/Male_positive_Young_positive_Eyeglasses_negative_Smiling_positive" \
    --skip_grid \
    --n_iter=13 \
    --n_samples=8 \
    --seed=1
