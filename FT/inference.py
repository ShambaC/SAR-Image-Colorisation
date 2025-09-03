import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

model_id = "ShambaC/SAR-Intruct-Pix2Pix"
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_auth_token=True
).to("cuda")

image_path = "https://chibi.winkfor.me/GYuLP7lBQhwi.png"
image = load_image(image_path)

image = pipeline("Colorize the image, Region: tropical, Season: fall", image=image).images[0]
image.save("image.png")