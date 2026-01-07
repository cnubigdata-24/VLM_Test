# pip install -q transformers pillow torch

from transformers import pipeline
from PIL import Image

# 이미지 사진
img = Image.open("demo.jpg").convert("RGB")
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

print(captioner(img, max_new_tokens=30))
