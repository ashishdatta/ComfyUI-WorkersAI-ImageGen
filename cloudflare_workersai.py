import base64
import json
import time
from io import BytesIO

import numpy as np
import torch
from cloudflare import Cloudflare
from PIL import Image

MODELS = [
    "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    "@cf/bytedance/stable-diffusion-xl-lightning",
    "@cf/black-forest-labs/flux-1-schnell",
]


class CloudflareWorkersAI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "CLOUDFLARE_API_TOKEN": ("STRING",),
                "CLOUDFLARE_ACCOUNT_ID": ("STRING",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                    },
                ),
                "num_steps": (
                    "INT",
                    {
                        "default": 20,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 123,
                    },
                ),
                "model": (MODELS,),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    OUTPUT_NODE = True
    CATEGORY = "api/image"

    def generate_image(
        self,
        CLOUDFLARE_API_TOKEN,
        CLOUDFLARE_ACCOUNT_ID,
        prompt,
        width,
        height,
        num_steps,
        seed,
        model,
    ):

        client = Cloudflare(api_token=CLOUDFLARE_API_TOKEN)

        response = client.workers.ai.with_raw_response.run(
            model,
            account_id=CLOUDFLARE_ACCOUNT_ID,
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            seed=seed,
        )
        if model == "@cf/black-forest-labs/flux-1-schnell":
            img_data = base64.b64decode(response.json()["result"]["image"])
        else:
            img_data = response.read()

        return (self.process_image(img_data),)

    def process_image(self, img_data):
        img = Image.open(BytesIO(img_data)).convert("RGBA")
        return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]

    @classmethod
    def IS_CHANGED():
        return time.time()


NODE_CLASS_MAPPINGS = {
    "CloudflareWorkersAI": CloudflareWorkersAI,
}
