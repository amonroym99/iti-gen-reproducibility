from pathlib import Path

base_script = """
#!/bin/bash

python models/sd/scripts/txt2img.py \\
    --config='models/sd/configs/stable-diffusion/v1-inference.yaml' \\
    --ckpt='models/sd/models/ldm/stable-diffusion-v1/model.ckpt' \\
    --plms \\
    --prompt="{prompt}" \\
    --outdir="{outdir}" \\
    --skip_grid \\
    --n_iter=13 \\
    --n_samples=8 \\
    --seed={seed}
"""
# Remove leading '\n'
base_script = base_script[1:]

prompt_template = "a headshot of {smile} {age} {gender}"
ages = ['old', 'young']
genders = ['man', 'woman']
smiling = ["a smiling", "a non-smiling"]

attr_names = [
    "Young",
    "Male",
    "Smiling",
]

idx_to_bool = [
    "positive",
    "negative",
]

seed = 0
for idx_age, age in enumerate(ages):
    for idx_gender, gender in enumerate(genders):
        for idx_smile, smile in enumerate(smiling):
                combination = ""
                for attr, idx in zip(attr_names, [idx_age, idx_gender, idx_smile]):
                    combination += f"{attr}_{idx_to_bool[idx]}_"
                combination = combination[:-1] # Remove the trailing '_'

                outdir = f"results/celeba_multi/2/hps/{combination}"
                prompt = prompt_template.format(smile=smile, age=age, gender=gender)
                
                # script = base_script.format(prompt=prompt, outdir=outdir, seed=seed)
                # jobfile = Path(f"jobfiles/celeba_multi/2/hps/generation/{prompt}.sh")
                # jobfile.parent.mkdir(exist_ok=True, parents=True)
                # jobfile.write_text(script)
                # seed += 1