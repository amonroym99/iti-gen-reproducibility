#!/bin/bash

python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Male,No_Beard" \
    --outdir="results/proxy_features/iti_gen/Male_No_Beard" \
    --prompt-path="ckpts/a_headshot_of_a_person_Male_No_Beard/original_prompt_embedding/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=1 \
    --n_samples=8 \
    --seed=42
