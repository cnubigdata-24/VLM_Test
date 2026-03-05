!pip install -q transformers accelerate pillow torch matplotlib

import requests
from PIL import Image

# 샘플 이미지 다운로드 (고양이 사진)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image.save("demo.jpg")
print("테스트용 demo.jpg 저장 완료!")


import torch
from transformers import pipeline, CLIPProcessor, CLIPModel
from PIL import Image

# --- Example 1) VQA (ViLT), pipeline 사용  ---
print("\n[Task 1] VQA 실행 중...")
vqa = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
# 'What is the color of the cat?' 질문에 대한 답변 추출
result_vqa = vqa(image=image, question="What color is the cat?")
print(f"답변: {result_vqa}")

# --- Example 2) Image Captioning (BLIP), 클래스 직접 활용 ---
print("\n[Task 2] 이미지 캡셔닝 실행 중...")
from transformers import BlipProcessor, BlipForConditionalGeneration

# 1. 모델과 전처리기 로드
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 2. 이미지 전처리
inputs = processor(image, return_tensors="pt")

# 3. 캡션 생성
out = model.generate(**inputs, max_new_tokens=30)
caption = processor.decode(out[0], skip_special_tokens=True)

print(f"캡션(Stable Version): {caption}")

# --- Example 3) Image Retrieval (CLIP) ---
print("\n[Task 3] CLIP 텍스트-이미지 매칭 실행 중...")
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor_clip(
    text=["a cat", "a dog", "a car"], 
    images=image, 
    return_tensors="pt", 
    padding=True
)

outputs = model_clip(**inputs)
logits_per_image = outputs.logits_per_image  # 이미지와 텍스트 간 유사도 점수
probs = logits_per_image.softmax(dim=1)      # 확률값으로 변환

print(f"텍스트 후보별 확률: a cat({probs[0][0]:.2f}), a dog({probs[0][1]:.2f}), a car({probs[0][2]:.2f})")
