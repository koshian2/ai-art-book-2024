from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
import torch
import os
from openai import OpenAI
import requests

prompts = {
    "anime" : [
        "1girl, japanese school anime, girl, solo, best quality",
        "1girl, magical girl, anime, solo, best quality"
    ],
    "real" : [
        "A black colored banana.",
        "One cat and one dog sitting on the grass."
    ]
}

def stable_diffusion_main():
    models = {
        "sd15": "runwayml/stable-diffusion-v1-5",
        "sd21": "stabilityai/stable-diffusion-2-1",
        "acertain": "JosephusCheung/ACertainThing",
        "anime_pastel_dream": "sinkinai/anime-pastel-dream-soft-baked-vae",
        "anythingv5": "stablediffusionapi/anything-v5",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "animagine_xl_31" : "cagliostrolab/animagine-xl-3.1"
    }

    for model_name, model_id in models.items():
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None)

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True
        )
        pipe = pipe.to("cuda")

        for category, sub_prompts in prompts.items():
            for pid, prompt in enumerate(sub_prompts):
                print(f"Generating {model_name}_{category}_{pid}")
                image = pipe(prompt).images[0]
                output_path = f"aesthetics_toy_images/{category}/{model_name}_{pid}.png"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path)


def dalle_main():
    models = {
        "dall_e_3": "dall-e-3",
        "dall_e_2": "dall-e-2",
    }

    for model_name, model_id in models.items():
        for category, sub_prompts in prompts.items():
            for pid, prompt in enumerate(sub_prompts):
                print(f"Generating {model_name}_{category}_{pid}")
                output_path = f"aesthetics_toy_images/{category}/{model_name}_{pid}.png"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                client = OpenAI()

                response = client.images.generate(
                    model=model_id,
                    prompt=prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1
                )

                image_url = response.data[0].url
                output_path = f"aesthetics_toy_images/{category}/{model_name}_{pid}.png"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                r = requests.get(image_url, stream=True)
                with open(output_path, "wb") as f:
                    f.write(r.content)

if __name__ == "__main__":
    stable_diffusion_main()
    dalle_main()