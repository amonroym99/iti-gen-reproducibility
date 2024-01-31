from pathlib import Path

base_script = """
#!/bin/bash

python models/sd/scripts/txt2img.py \\
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \\
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \\
    --plms \\
    --prompt="{prompt}" \\
    --negative_prompt="{negative_prompt}" \\
    --outdir="{outdir}" \\
    --skip_grid \\
    --n_iter=13 \\
    --n_samples=8 \\
    --seed={seed}
"""
# Remove leading "\n"
base_script = base_script[1:]

hard_prompts = {
    "Bald_Man": ("a headshot of a bald man", ""),
    "Bald_Woman": ("a headshot of a bald woman", ""),
    "Not_Bald_Man": ("a headshot of a man", "bald"),
    "Not_Bald_Woman": ("a headshot of a woman", "bald"),

    "Mustache_Man": ("a headshot of a man with a mustache", ""),
    "Mustache_Woman": ("a headshot of a woman with a mustache", ""),
    "Not_Mustache_Man": ("a headshot of a man", "mustache"),
    "Not_Mustache_Woman": ("a headshot of a woman", "mustache"),

    "Beard_Man": ("a headshot of a man with a beard", ""),
    "Beard_Woman": ("a headshot of a woman with a beard", ""),
    "Not_Beard_Man": ("a headshot of a man", "beard"),
    "Not_Beard_Woman": ("a headshot of a woman", "beard"),
}

seed = 0
for attribute, (prompt, negative_prompt) in hard_prompts.items():
    outdir = f"results/proxy_features/hps_negative/{attribute}"
    script = base_script.format(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        outdir=outdir,
        seed=seed
    )
    jobfile = Path(f"jobfiles/proxy_features/hps_negative/generation/{attribute}.sh")
    jobfile.parent.mkdir(exist_ok=True, parents=True)
    jobfile.write_text(script)
    seed += 1