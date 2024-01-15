import argparse

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from generation import load_model_from_config
from models.sd.ldm.models.diffusion.plms import PLMSSampler

if torch.cuda.is_available():
    device = 'cuda:0'
elif torch.backends.mps.is_available():
    device = 'mps'
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


def get_images(prompt, num_samples):
    model = load_model_from_config(
        config=OmegaConf.load('models/sd/configs/stable-diffusion/v1-inference.yaml'),
        ckpt='models/sd/models/ldm/stable-diffusion-v1/model.ckpt'
    ).to(device)
    sampler = PLMSSampler(model)

    for _ in range(num_samples):
        # TODO: increase batch size!!!!
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

        yield samples[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Prompt combiner',
        description='Combines a text prompt with several inclusive prompts'
    )
    parser.add_argument('--text-prompt', type=str, required=True)
    parser.add_argument(
        '--inclusive-prompt',
        type=str,
        default=[],
        action='append',
        help='You can specify this multiple times. It should be of the form "path-to-basis-perturbation.pth::value", '
             'for example, ckpts/a_headshot_of_a_person_Male_Skin_tone_Age/basis_perturbation_embed_29_Male.pth::0',
    )
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    inclusive_prompt_category_by_attribute = {}
    for inclusive_prompt in args.inclusive_prompt:
        attribute_path, category = inclusive_prompt.split('::')
        inclusive_prompt_category_by_attribute[attribute_path] = int(category)

    prompt = get_prompt(args.text_prompt, inclusive_prompt_category_by_attribute)
    for i, image in enumerate(get_images(prompt, args.num_samples)):
        image_path = f'{args.output}-{i}.png'
        plt.imsave(image_path, image)
