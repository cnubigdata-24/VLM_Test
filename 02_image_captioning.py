# pip install -q transformers pillow torch

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 1. Load the image
img = Image.open("demo.jpg").convert("RGB")

# 2. Load the specific BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 3. Preprocess the image (passing only image data)
inputs = processor(images=img, return_tensors="pt")

# 4. Generate caption and decode the output tokens
outputs = model.generate(**inputs, max_new_tokens=30)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print(caption)
