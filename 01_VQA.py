# pip install -q transformers accelerate pillow torch 

from transformers import pipeline 
from PIL import Image 

img = Image.open("demo.jpg").convert("RGB") 
vqa = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")

question = "What is the person holding?"
print(vqa(image=img, question=question))


# The image processor of type `ViltImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 
# [{'score': 0.9782189726829529, 'answer': 'cat'}, {'score': 0.01485508494079113, 'answer': 'nothing'}, {'score': 0.0146604860201478, 'answer': 'remote'}, {'score': 0.01343337818980217, 'answer': 'remote control'}, {'score': 0.008481381461024284, 'answer': 'kitten'}]
