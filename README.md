# What do and how work this model

## What do this model

This model name **realvisxl4** use RealVisXL 4.0 ( Pretrained model using Stable Diffusion XL ) for generate picture.
You can make prediction with this model in this Replicate model :

**REPLICATE :** https://replicate.com/zelenioncode/realvisxl4

## How this model work

Before start, we need to have **Cog** and **Docker**. For learn Cog, [click her for Github Doc](https://github.com/replicate/cog/tree/main).
But for start, use brew for install Cog :

```bash
brew install cog
```

After for this model, i use only 2 files :

- [cog.yaml](https://github.com/WGlint/RealVisXL-v4.0/blob/main/cog.yaml)

- [predict.py](https://github.com/WGlint/RealVisXL-v4.0/blob/main/predict.py)

All the code is in this [**repo Github**](https://github.com/WGlint/RealVisXL-v4.0).

Or, let check all code her :

[cog.yaml](https://github.com/WGlint/RealVisXL-v4.0/blob/main/cog.yaml)
```yaml
# Configuration for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/yaml.md

build:
  # set to true if your model requires a GPU
  gpu: true
  cuda: "11.8"
  python_version: "3.9"
  python_packages:
    - "torch==2.0.1"
    - "torchvision==0.15.2"
    - "transformers==4.31.0"
    - "safetensors==0.3.1"
    - "diffusers==0.19.0"
    - "accelerate==0.21.0"
    - "numpy==1.25.1"
    - "omegaconf==2.3.0"
    - "xformers"
    - "invisible-watermark==0.2.0"
    - "fire==0.5.0"
    - "opencv-python>=4.1.0.25"
    - "mediapipe==0.10.2"

  run : 
    - "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y"

predict: "predict.py:Predictor"
image: "r8.im/xx/xx" # <-- ENTER YOURS IMAGE NAME MODEL FROM REPLICATE
```
[predict.py](https://github.com/WGlint/RealVisXL-v4.0/blob/main/predict.py)
```python
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
```

Let's check my other model !
