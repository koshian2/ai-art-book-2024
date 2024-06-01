# AnimateDiff-Lightning Demo based on https://huggingface.co/spaces/ByteDance/AnimateDiff-Lightning

import gradio as gr
import torch
import uuid
import os

from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from diffusers.utils.export_utils import export_to_video, export_to_gif
from safetensors.torch import load_file

# Constants
pipe = None
step_loaded = None
base_loaded = None
motion_loaded = None

# Function 
def generate_image(prompt, base, motion, step, progress=gr.Progress()):
    global step_loaded
    global base_loaded
    global motion_loaded
    global pipe
    print(prompt, base, step)

    # Base model and Lightning Adapter
    if base_loaded != base or step_loaded != step:
        # Ensure model and scheduler are initialized in GPU-enabled function
        if not torch.cuda.is_available():
            raise NotImplementedError("No GPU detected!")

        device = "cuda"
        dtype = torch.float16

        repo = "ByteDance/AnimateDiff-Lightning"
        ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"

        adapter = MotionAdapter().to(device, dtype)
        adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))
        pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
        pipe.enable_vae_slicing()
        step_loaded = step
        base_loaded = base

    # Motion LoRA
    if motion_loaded != motion:
        pipe.unload_lora_weights()
        if motion != "":
            pipe.load_lora_weights(motion, adapter_name="motion")
            pipe.set_adapters(["motion"], [0.7])
        motion_loaded = motion

    progress((0, step))
    def progress_callback(i, t, z):
        progress((i+1, step))

    output = pipe(prompt=prompt, guidance_scale=1.0, num_inference_steps=step, callback=progress_callback, callback_steps=1, width=512, height=512)

    name = str(uuid.uuid4()).replace("-", "")
    path_mp4 = f"/tmp/{name}.mp4"
    path_gif = f"/tmp/{name}.gif"
    os.makedirs("/tmp", exist_ok=True)
    export_to_video(output.frames[0], path_mp4, fps=10)
    export_to_gif(output.frames[0], path_gif)
    return [path_gif, path_mp4]


# Gradio Interface
with gr.Blocks(css="style.css") as demo:
    gr.HTML(
        "<h1><center>AnimateDiff-Lightning ⚡</center></h1>" +
        "<p><center>Lightning-fast text-to-video generation</center></p>" +
        "<p><center><a href='https://huggingface.co/ByteDance/AnimateDiff-Lightning'>https://huggingface.co/ByteDance/AnimateDiff-Lightning</a></center></p>"
    )
    with gr.Group():
        with gr.Row():
            prompt = gr.Textbox(
                label='Prompt (English)',
                interactive=True
            )
        with gr.Row():
            select_base = gr.Textbox(
                label='Base model',
                value="sinkinai/anime-pastel-dream-soft-baked-vae",
                interactive=True
            )
            select_motion = gr.Dropdown(
                label='Motion',
                choices=[
                    ("Default", ""),
                    ("Zoom in", "guoyww/animatediff-motion-lora-zoom-in"),
                    ("Zoom out", "guoyww/animatediff-motion-lora-zoom-out"),
                    ("Tilt up", "guoyww/animatediff-motion-lora-tilt-up"),
                    ("Tilt down", "guoyww/animatediff-motion-lora-tilt-down"),
                    ("Pan left", "guoyww/animatediff-motion-lora-pan-left"),
                    ("Pan right", "guoyww/animatediff-motion-lora-pan-right"),
                    ("Roll left", "guoyww/animatediff-motion-lora-rolling-anticlockwise"),
                    ("Roll right", "guoyww/animatediff-motion-lora-rolling-clockwise"),
                ],
                value="",
                interactive=True
            )
            select_step = gr.Dropdown(
                label='Inference steps',
                choices=[
                    ('1-Step', 1), 
                    ('2-Step', 2),
                    ('4-Step', 4),
                    ('8-Step', 8)],
                value=4,
                interactive=True
            )
            submit = gr.Button(
                scale=1,
                variant='primary'
            )
        with gr.Row():
            video = gr.Image(
                label='AnimateDiff-Lightning',
                height=512,
                width=512,
                elem_id="video_output"
            )
            download_file = gr.File(label="Download mp4")

    examples = gr.Examples(
        examples=[
            ["a girl is looking at flowers on a hill, yellow flower field, 1girl, solo, upper body, blue sky, cliff above the sea, best quality", 
            "sinkinai/anime-pastel-dream-soft-baked-vae", "guoyww/animatediff-motion-lora-tilt-up", 8],
            ["A girl smiling", "emilianJR/epiCRealism", "", 4],
            ["A beautiful girl in a pink dress", "frankjoshua/toonyou_beta6", "guoyww/animatediff-motion-lora-zoom-in", 4],
        ],
        inputs=[prompt, select_base, select_motion, select_step],
    )

    prompt.submit(
        fn=generate_image,
        inputs=[prompt, select_base, select_motion, select_step],
        outputs=[video, download_file],
    )
    submit.click(
        fn=generate_image,
        inputs=[prompt, select_base, select_motion, select_step],
        outputs=[video, download_file],
    )

demo.launch(server_name="0.0.0.0")