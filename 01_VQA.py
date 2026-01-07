# pip install -q transformers accelerate pillow torch 

from transformers import pipeline 
from PIL import Image 

# 이미지 경로
img = Image.open("demo.jpg").convert("RGB") 
vqa = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")

question = "What is the person holding?" 

print(vqa(image=img, question=question))
