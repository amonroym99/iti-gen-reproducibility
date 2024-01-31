# CelebA (single attributes)
This experiment compares the diversity and image quality of vanilla Stable Diffusion, ITI-GEN, HPS and HPS with negative prompting.
## Training
### Train ITI-GEN tokens
You need to run `jobfiles/celeba_single/iti_gen/train/${attribute}.sh`, for example `jobfiles/celeba_single/iti_gen/train/Young.sh`. To run all of them, do the following:
```shell
find jobfiles/celeba_single/iti_gen/train/*.sh | xargs -n1 bash
```
## Image generation
### Generate images with vanilla Stable Diffusion
```shell
python models/sd/scripts/txt2img.py \
    --config='models/sd/configs/stable-diffusion/v1-inference.yaml' \
    --ckpt='models/sd/models/ldm/stable-diffusion-v1/model.ckpt' \
    --plms \
    --prompt="a headshot of a person" \
    --negative_prompt="" \
    --outdir='results/celeba_single/vanilla' \
    --skip_grid \
    --n_iter=630 \>
    --n_samples=8 \
    --seed=42
```
### Generate images with ITI-GEN
You need to run `jobfiles/celeba_single/iti_gen/generation/${attribute}.sh`, for example `jobfiles/celeba_single/iti_gen/generation/Young.sh`. To run all of them, do the following
```shell
find jobfiles/celeba_single/iti_gen/generation/*.sh | xargs -n1 bash
```
### Generate images with HPS
You need to run `jobfiles/celeba_single/hps/generation/${attribute}.sh`, for example `jobfiles/celeba_single/hps/generation/Young.sh`. To run all of them, do the following
```shell
find jobfiles/celeba_single/hps/generation/*.sh | xargs -n1 bash
```
### Generate images with HPS with negative prompting
You need to run `jobfiles/celeba_single/hps_negative/generation/${attribute}.sh`, for example `jobfiles/celeba_single/hps_negative/generation/Young.sh`. To run all of them, do the following
```shell
find jobfiles/celeba_single/hps_negative/generation/*.sh | xargs -n1 bash
```
## Evaluation
### Compute FID score
Using the default parameters, 104 images are generated per category, making a total of $104 \times 80 = 8320$ images. In the original paper only 5K images are used, so we decide to take just 63 images per category, which adds up to $63 \times 80 = 5040$ images. 
```shell
python fid.py
```
### Compute KL-divergences
You need to run `jobfiles/celeba_single/${method}/evaluation/${attribute}.sh`, for example `jobfiles/celeba_single/hps_negative/evaluation/Young.sh`. To run all of them, do the following
```shell
find jobfiles/celeba_single -name "*.sh" | grep evaluation | xargs -n1 bash
```

# Multiple domains
This experiment demonstrates how ITI-GEN works on multiple domains.
## Training
### People
```shell
python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="Skin_tone" \
    --epochs=30 \
    --save-ckpt-per-epochs=10
```
### Natural scenes
```shell
python train_iti_gen.py \
    --prompt='a natural scene' \
    --attr-list='Colorful' \
    --epochs=30 \
    --save-ckpt-per-epochs=10
```
## Image generation
### People
```shell
python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Skin_tone" \
    --outdir="./results/multiple_domains/a_headshot_of_a_person_Skin_tone" \
    --prompt-path="ckpts/a_headshot_of_a_person_Skin_tone/original_prompt_embedding/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=3  \
    --n_samples=8 \
    --seed 42
```
### Natural scenes
```shell
python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Colorful" \
    --outdir="./results/multiple_domains/a_natural_scene_Colorful" \
    --prompt-path="ckpts/a_natural_scene_Colorful/prepend_prompt_embedding_a_headshot_of_a_person/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=3 \
    --n_samples=8 \
    --seed 42
```

# Train-once-for-all
This experiment shows the plug-and-play capabilities of ITI-GEN: the text prompt used for training need not be the same as the one used for generation.

## Train
### Young
Note that Young is a binary feature, not to be confused with Age.
```shell
python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="Young" \
    --epochs=30 \
    --save-ckpt-per-epochs=10
```
### Male
```shell
python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="Male" \
    --epochs=30 \
    --save-ckpt-per-epochs=10
```

## Preprend
This step will append the inclusive tokens to a different text prompt.

### Firefighter ++ Young
```shell
python prepend.py \
    --prompt='a headshot of a person' \
    --attr-list='Young' \
    --load-model-epoch=29 \
    --prepended-prompt='a headshot of a firefighter'
```
### Doctor ++ Male
```shell
python prepend.py \
    --prompt="a headshot of a person" \
    --attr-list="Male" \
    --load-model-epoch=29 \
    --prepended-prompt="a headshot of a doctor"
```

## Generate images

### Firefighter ++ Young
```shell
python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Young" \
    --outdir="./results/train_once_for_all/firefighter" \
    --prompt-path="ckpts/a_headshot_of_a_person_Young/prepend_prompt_embedding_a_headshot_of_a_firefighter/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=3 \
    --n_samples=4 \
    --seed 42
```
### Doctor ++ Male
```shell
python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Male" \
    --outdir="./results/train_once_for_all/doctor" \
    --prompt-path="ckpts/a_headshot_of_a_person_Male/prepend_prompt_embedding_a_headshot_of_a_doctor/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=3 \
    --n_samples=4 \
    --seed 42
```

# Proxy features
This experiment shows how ITI-GEN uses gender as a proxy attribute when learning other attributes, such as "Bald", "Mustache" or "Beard". 

## Train ITI-GEN inclusive tokens
You have to run `jobfiles/proxy_features/iti_gen/generation/${attribute}.sh`. To run all of them
```shell
find jobfiles/proxy_features/iti_gen/generation/ -name "*.sh" | xargs -n1 bash
```

## Generation
You have to run `jobfiles/proxy_features/${method}/generation/${attribute}.sh`. To run all of them
```shell
find jobfiles/proxy_features/*/generation -name *.sh | xargs -n1 bash
```
