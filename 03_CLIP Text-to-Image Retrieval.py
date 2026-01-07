# !pip install transformers torch pillow matplotlib

import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Tuple

# 디렉토리에서 이미지 파일들을 로드
def load_images_from_dir(
    img_dir: str, 
    exts: tuple = (".jpg", ".jpeg", ".png", ".webp")
) -> Tuple[List[Image.Image], List[str]]:
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {img_dir}")
    
    paths = [
        os.path.join(img_dir, f)
        for f in sorted(os.listdir(img_dir))
        if f.lower().endswith(exts)
    ]
    
    if not paths:
        raise ValueError(f"이미지 파일이 없습니다. (지원 확장자: {exts})\n경로: {img_dir}")
    
    images = []
    valid_paths = []
    
    for p in paths:
        try:
            images.append(Image.open(p).convert("RGB"))
            valid_paths.append(p)
        except Exception as e:
            print(f"[WARN] 이미지 로드 실패: {p} - {type(e).__name__}: {e}")
    
    if not valid_paths:
        raise ValueError("모든 이미지 로드에 실패했습니다. 파일 손상/권한/형식 등을 확인하세요.")
    
    return images, valid_paths

# CLIP 모델을 사용한 텍스트-이미지 검색
def clip_text_to_image_retrieval(
    image_dir: str, 
    query: str, 
    top_k: int = 3, 
    model_name: str = "openai/clip-vit-base-patch32"
) -> List[Dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")
    
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    
    images, paths = load_images_from_dir(image_dir)
    print(f"로드된 이미지 수: {len(images)}")
    
    img_inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        img_emb = model.get_image_features(**img_inputs)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    
    txt_inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        txt_emb = model.get_text_features(**txt_inputs)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    
    scores = (img_emb @ txt_emb.T).squeeze(1)
    
    k = min(top_k, len(paths))
    top_scores, top_idx = torch.topk(scores, k=k)
    
    results = []
    for s, i in zip(top_scores.tolist(), top_idx.tolist()):
        results.append({
            "path": paths[i], 
            "score": float(s),
            "filename": os.path.basename(paths[i])
        })
    
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return results

# Top-K 이미지를 시각화하는 함수
def show_topk_images(results: List[Dict], cols: int = 3, figsize: tuple = (12, 4)):
    k = len(results)
    cols = min(cols, k)
    rows = (k + cols - 1) // cols
    
    plt.figure(figsize=figsize)
    
    for j, r in enumerate(results, start=1):
        img = Image.open(r["path"]).convert("RGB")
        ax = plt.subplot(rows, cols, j)
        ax.imshow(img)
        ax.set_title(f"Top{j} | {r['score']:.4f}\n{r['filename']}", fontsize=9)
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_dir = "./images"
    query = "a person riding a bicycle"
    top_k = 3
    
    # 한글 쿼리 시 다국어 모델 사용 
    # model_name = "Bingsu/clip-vit-large-patch14-ko"
    
    results = clip_text_to_image_retrieval(
        image_dir, 
        query, 
        top_k=top_k
    )
    
    print(f"\nQuery: {query}")
    print("="*60)
    for idx, r in enumerate(results, start=1):
        print(f"Top{idx}: {r['filename']:30s} score={r['score']:.4f}")
    
    show_topk_images(results, cols=3, figsize=(12, 4))
