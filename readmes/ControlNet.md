# ITI-GEN + ControlNet
This experiment proves the plug-and-play capabilities of ITI-GEN when it is used with [ControlNet 1.0](/guides/content/editing-an-existing-page).

## Setup
First, you need to change the directory to `controlnet` and create a new conda environment.

```shell
cd controlnet
conda env create -f environment.yaml
conda activate control
```

Then, you need to download the pretrained models from [the official Hugging Face page for ControlNet](https://huggingface.co/lllyasviel/ControlNet). All the SD models _must be downloaded in the "models/controlnet" folder._

For our experiments, we generated images using Canny Edge detection, human pose, and depth map. Therefore, the only pretrained models needed are _control_sd15_canny.pth_, _control_sd15_openpose.pth_, _control_sd15_depth.pth_.

Some of the input images can be found in "test_imgs" folder.

## Image generation
### ControlNet with Canny Edge
The images are generated using Stable Diffusion 1.5 + ControlNet using a simple Canny edge detection.

Run the following command:
```
python canny2image.py \
    --attr-list='Colorful' \
    --outdir='results/Colorful' \
    --prompt-path='../ckpts/a_natural_scene_Colorful/prepend_prompt_embedding_photograph_of_mount_katahdin/basis_final_embed_29.pt' \
    --input_image='test_imgs/mountain.png' \
    --num_samples=3
```
- `--attr_list`: attributes separated by comma. They should all be aligned with those used in training ITI-GEN.
- `--outdir`: the path to the folder in which the generated images are going to be saved.
- `prompt-path`: path to the trained inclusive prompt.
- `input-image`: the image on which the Canny Edge algorithm is applied.
- `num_samples`: number of samples generated per category. 

### ControlNet with Human Pose
The images are generated using Stable Diffusion 1.5 + ControlNet using human pose.

Run the following command:
```
python pose2image.py \
    --attr-list='Age' \
    --outdir='results/Age' \
    --prompt-path='../ckpts/a_headshot_of_a_person_Age/prepend_prompt_embedding_photo_of_a_famous_woman/basis_final_embed_29.pt' \
    --input_image='test_imgs/pose1.png' \
    --num_samples=3
```
- `--attr_list`: attributes separated by comma. They should all be aligned with those used in training ITI-GEN.
- `--outdir`: the path to the folder in which the generated images are going to be saved.
- `prompt-path`: path to the trained inclusive prompt.
- `input-image`: the generated images are guided by the human pose from this photo.
- `num_samples`: number of samples generated per category.


### ControlNet with Depth
The images are generated using Stable Diffusion 1.5 + ControlNet using depth map.

Run the following command:
```
python depth2image.py \
    --attr-list='Colorful' \
    --outdir='results/Colorful' \
    --prompt-path='../ckpts/a_natural_scene_Colorful/prepend_prompt_embedding_photograph_of_mount_katahdin/basis_final_embed_29.pt' \
    --input_image="test_imgs/mountain.png" \
    --num_samples=3
```
- `--attr_list`: attributes separated by comma. They should all be aligned with those used in training ITI-GEN.
- `--outdir`: the path to the folder in which the generated images are going to be saved.
- `prompt-path`: path to the trained inclusive prompt.
- `input-image`: the image that is used as a depth map.
- `num_samples`: number of samples generated per category.
