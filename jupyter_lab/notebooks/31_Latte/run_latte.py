import os
import torch
from diffusers.schedulers import PNDMScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from diffusers.utils import export_to_gif, export_to_video
from transformers import T5EncoderModel, T5Tokenizer
from latte_t2v import LatteT2V


# import sys
# sys.path.append(os.path.split(sys.path[0])[0])
from download import find_model
from pipeline_videogen import VideoGenPipeline
from PIL import Image

## == Params ==
video_length = 16
pretrained_model_path = "./share_ckpts/t2v_required_models"
t2v_checkpoint_path = "./share_ckpts/t2v_v20240523.pt"
enable_vae_temporal_decoder = True
## =======

# torch.manual_seed(args.seed)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

# LatteT2V
transformer_model = LatteT2V.from_pretrained_2d(
    pretrained_model_path, 
    subfolder="transformer", 
    video_length=video_length)
state_dict = find_model(t2v_checkpoint_path)
transformer_model.load_state_dict(state_dict)
transformer_model.half()

# VAE
if enable_vae_temporal_decoder:
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pretrained_model_path, 
        subfolder="vae_temporal_decoder", 
        torch_dtype=torch.float16)
    # tilingもslicingも不可能
else:
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_path, 
        subfolder="vae", 
        torch_dtype=torch.float16)
    vae.enable_tiling()
    vae.enable_slicing()

# Encoder
tokenizer = T5Tokenizer.from_pretrained(
    pretrained_model_path, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(
    pretrained_model_path, 
    subfolder="text_encoder", 
    torch_dtype=torch.float16)

# set eval mode
transformer_model.eval()
vae.eval()
text_encoder.eval()

# Scheduler selection
scheduler = PNDMScheduler.from_pretrained(
    pretrained_model_path,
    subfolder="scheduler",
    beta_start=0.0001, 
    beta_end=0.02, 
    beta_schedule="linear",
    variance_type="learned_range"
)

videogen_pipeline = VideoGenPipeline(
    vae=vae, 
    text_encoder=text_encoder, 
    tokenizer=tokenizer, 
    scheduler=scheduler, 
    transformer=transformer_model
)
# videogen_pipeline.to(device)
videogen_pipeline.enable_model_cpu_offload()


def run_t2v(text_prompt,
            save_base_path,
            width=512, height=512,
            num_inference_steps=50,
            guidance_scale=7.5,
            enable_temporal_attentions=True):
    
    os.makedirs(os.path.dirname(save_base_path), exist_ok=True)

    videos = videogen_pipeline(
        text_prompt, 
        video_length=video_length, 
        height=width, 
        width=height, 
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        enable_temporal_attentions=enable_temporal_attentions,
        num_images_per_prompt=1,
        mask_feature=True,
        enable_vae_temporal_decoder=enable_vae_temporal_decoder
    ).video[0] # uint8 tensor video = [B, T, H, W, C]
    frames = [Image.fromarray(videos[i].cpu().numpy()) for i in range(videos.shape[0])]

    export_to_gif(frames, save_base_path+".gif", fps=8)
    export_to_video(frames, save_base_path+".mp4", fps=8)

text_prompt = [
    'Yellow and black tropical fish dart through the sea.',
    'An epic tornado attacking above aglowing city at night.',
    'Slow pan upward of blazing oak fire in an indoor fireplace.',
    'a cat wearing sunglasses and working as a lifeguard at pool.',
    'Sunset over the sea.',
    'A dog in astronaut suit and sunglasses floating in space.'
]

if __name__ == "__main__":
    for i, prompt in enumerate(text_prompt):
        run_t2v(prompt, f"generated/video_{i:02}")