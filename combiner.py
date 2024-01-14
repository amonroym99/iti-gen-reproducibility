import numpy as np
import torch.backends.mps
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from generation import load_model_from_config
from models.sd.ldm.models.diffusion.plms import PLMSSampler

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 0
else:
    device = 'cpu'


def get_prompt(text_prompt, inclusive_prompt_category_by_attribute):
    import clip

    inclusive_prompt = 0
    for attribute, category in inclusive_prompt_category_by_attribute.items():
        # The stored weights have shape num_categories x 3 x 768
        weights = torch.load(attribute, map_location=device)['para']

        inclusive_prompt += weights[category]

    tokenized_text_prompt = clip.tokenize(text_prompt).to(device)
    # The text will be "<start> ${text_prompt} <end>", and <end> has the highest index
    text_prompt_end_index = tokenized_text_prompt.argmax(dim=-1)
    inclusive_prompt_start_index = text_prompt_end_index + 1

    clip_model, _ = clip.load('ViT-L/14')
    clip_model = clip_model.eval().to(device)
    text_embedding = clip_model.token_embedding(tokenized_text_prompt).type(clip_model.dtype)
    text_embedding += clip_model.positional_embedding.type(clip_model.dtype)

    # prompt has shape bs(1) x max_num_words(77) x embedding_dimension(768)
    prompt = text_embedding
    prompt[:, inclusive_prompt_start_index:inclusive_prompt_start_index + len(inclusive_prompt), :] = inclusive_prompt

    prompt = prompt.permute(1, 0, 2)
    prompt = clip_model.transformer(prompt)
    prompt = prompt.permute(1, 0, 2)
    prompt = clip_model.ln_final(prompt).type(clip_model.dtype)

    return prompt


def get_sample(prompt):
    model = load_model_from_config(
        config=OmegaConf.load('models/sd/configs/stable-diffusion/v1-inference.yaml'),
        ckpt='models/sd/models/ldm/stable-diffusion-v1/model.ckpt'
    ).to(device)
    sampler = PLMSSampler(model)

    samples, _ = sampler.sample(
        S=50,
        conditioning=prompt,
        batch_size=1,
        shape=(4, 64, 64),
        verbose=False,
        # TODO: what are these doing?
        unconditional_guidance_scale=6,
        unconditional_conditioning=model.get_learned_conditioning([""]),
        eta=0.0,
        x_T=None
    )
    samples = model.decode_first_stage(samples)
    samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
    samples = samples.cpu().permute(0, 2, 3, 1).numpy()

    sample = samples[0]
    return sample


if __name__ == '__main__':
    text_prompt = 'A photo of a doctor'
    # Male categories: [0, 1]
    # Age categories: [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # Skin_tone categories: [0, 1, 2, 3, 4, 5]
    inclusive_prompt_category_by_attribute = {
        'ckpts/a_headshot_of_a_person_Male_Skin_tone_Age/basis_perturbation_embed_29_Male.pth': 1,
        'ckpts/a_headshot_of_a_person_Male_Skin_tone_Age/basis_perturbation_embed_29_Age.pth': 1,
        # 'ckpts/a_headshot_of_a_person_Male_Skin_tone_Age/basis_perturbation_embed_29_Skin_tone.pth': 5,
    }

    prompt = get_prompt(text_prompt, inclusive_prompt_category_by_attribute)
    sample = get_sample(prompt)
    plt.imsave('sample.png', sample)
