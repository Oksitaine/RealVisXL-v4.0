# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from diffusers import EulerAncestralDiscreteScheduler
import torch
from PIL import Image
from typing import List

# Constants
MODEL_PIPELINE_CACHE = "diffusers-cache"
MODEL_noVAE = "SG161222/RealVisXL_V4.0"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        from diffusers import ( DiffusionPipeline )
        self.rv_VAE = DiffusionPipeline.from_pretrained(
            MODEL_noVAE,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors = True
        ).to("cuda")


    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Enter a prompt", default="RAW photo, a portrait photo of a latina woman in casual clothes, natural skin, 8k uhd, high quality, film grain, Fujifilm XT3"),
        negative_prompt: str = Input(description="Enter a negative prompt", default="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"),
        scheduler: str = Input(description="Scheduler to use", default="DDIM", choices=[
            "DDIM",
            "DPMSolverMultistep",
            "HeunDiscrete",
            "KarrasDPM",
            "K_EULER_ANCESTRAL",
            "K_EULER",
            "PNDM"
        ]),
        width: int = Input(description="Enter a width", default=512),
        height: int = Input(description="Enter a height", default=768),
        guidance_scale: int = Input(description="Enter a guidance scale", default=7),
        num_inference_steps: int = Input(description="Enter a number of inference steps", default=20),
        seed: int = Input(description="Enter a seed", default=42),
        number_picture: int = Input(description="Enter a number of picture", default=1, le=4, ge=1),
    ) -> List[Path]:

        generator = torch.Generator("cuda").manual_seed(seed)
        Parameters = {
            "prompt": [prompt] * number_picture,
            "negative_prompt": [negative_prompt] * number_picture,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator
        }

        print(Parameters)

        self.rv_VAE.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.rv_VAE.scheduler.config
        )
        image = self.rv_VAE(
            **Parameters
        )

        print(image)

        output = []
        for i, sample in enumerate(image.images):
            output_path = f"/tmp/picture_{i}.png"
            sample.save(output_path)
            output.append(Path(output_path))

        return output