import sys
gmflow_dir = "/home/34_Fresco/gmflow"
sys.path.insert(0, gmflow_dir)

import argparse
from PIL import Image
import cv2
import torch
import os
from diffusers import ControlNetModel,DDIMScheduler, DiffusionPipeline
from diffusers.utils import make_image_grid
from diffusers.utils.export_utils import export_to_video
from controlnet_aux import CannyDetector, MidasDetector, PidiNetDetector, OpenposeDetector
from random import seed

controlnet_mapper = {
    "canny": [
        "lllyasviel/control_v11p_sd15_canny", 
        CannyDetector()
    ],
    "depth": [
        "lllyasviel/control_v11f1p_sd15_depth", 
        MidasDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
    ],
    "soft_edge": [
        "lllyasviel/control_v11p_sd15_softedge", 
        PidiNetDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
    ],
    "pose": [
        "lllyasviel/control_v11p_sd15_openpose",
        OpenposeDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
    ]
}


def video_to_frame(video_path: str, interval: int, 
                   max_frames: int=None):
    vidcap = cv2.VideoCapture(video_path)
    success = True

    count = 0
    res = []
    while success:
        count += 1
        success, image = vidcap.read()
        if count % interval != 1:
            continue
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res.append(image)

    vidcap.release()
    if max_frames is not None:
        res = res[:max_frames]
    return res


def run_fresco_video(input_video_path : str, output_video_base_path : str,
                     base_sd15_model_name : str, prompt : str, negative_prompt : str,
                     controlnet_name : str="canny",
                     input_interval: int=5, width: int=512, height: int=512, max_frames: int=None,
                     controlnet_weight: float=0.7, seed : int=42, strength: float=0.75, save_frames: bool=False):
    # Extract frames
    frames = video_to_frame(
        input_video_path, input_interval, max_frames=max_frames)
    frames = [Image.fromarray(frame) for frame in frames]
    print("Frames len : ", len(frames))

    # Append controlnet
    controlnet = ControlNetModel.from_pretrained(
        controlnet_mapper[controlnet_name][0], torch_dtype=torch.float16)

    # get condition images
    control_frames = []
    for frame in frames:
        image = controlnet_mapper[controlnet_name][1](frame)
        image = image.resize((width, height)).convert("RGB")
        control_frames.append(image)
 
    # resize frames
    frames = [frame.resize((width, height)).convert("RGB") for frame in frames]
    print(len(frames), frames[0].size, len(control_frames), control_frames[0].size)

    # Fresco pipeline
    pipe = DiffusionPipeline.from_pretrained(
        base_sd15_model_name, controlnet=controlnet, custom_pipeline='fresco_v2v', torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_sequential_cpu_offload()

    output_frames = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        frames=frames,
        control_frames=control_frames,
        num_inference_steps=20,
        strength=strength,
        controlnet_conditioning_scale=controlnet_weight,
        generator=torch.manual_seed(seed),
    ).images

    os.makedirs(os.path.dirname(output_video_base_path), exist_ok=True)
    output_frames[0].save(output_video_base_path+".gif", save_all=True,
                    append_images=output_frames[1:], duration=int(1000/30*input_interval), loop=0)
    export_to_video(output_frames, output_video_base_path+".mp4", fps=int(30/input_interval))
    if save_frames:
        skip_frames = len(output_frames) // 4
        make_image_grid(frames[:skip_frames*4:skip_frames], rows=1, cols=4).save(output_video_base_path+"_input_frames.png")
        make_image_grid(output_frames[:skip_frames*4:skip_frames], rows=1, cols=4).save(output_video_base_path+"_output_frames.png")


def main():
    parser = argparse.ArgumentParser(description='Run Fresco Video Generation')
    parser.add_argument('--input_video_path', type=str, help='Path to the input video')
    parser.add_argument('--output_video_base_path', type=str, help='Base path for the output video')
    parser.add_argument('--base_sd15_model_name', type=str, help='Name of the base SD15 model')
    parser.add_argument('--prompt', type=str, help='Text prompt for the video')
    parser.add_argument('--negative_prompt', type=str, help='Negative prompt for the video')
    parser.add_argument('--controlnet_name', type=str, default='canny', help='Name of the controlnet')
    parser.add_argument('--input_interval', type=int, default=5, help='Input interval')
    parser.add_argument('--width', type=int, default=512, help='Width of the video')
    parser.add_argument('--height', type=int, default=512, help='Height of the video')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames')
    parser.add_argument('--controlnet_weight', type=float, default=0.7, help='Controlnet weight')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--strength', type=float, default=0.75, help='Strength of the effect')
    parser.add_argument('--save_frames', action='store_true', default=False, help='Save frames as png')

    args = parser.parse_args()

    run_fresco_video(args.input_video_path, args.output_video_base_path, args.base_sd15_model_name,
                     args.prompt, args.negative_prompt, args.controlnet_name, args.input_interval,
                     args.width, args.height, args.max_frames, args.controlnet_weight, args.seed, args.strength, args.save_frames)

if __name__ == '__main__':
    main()





