# !pip install transformers torch pillow matplotlib

import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Tuple

# 이미지 로드
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
        raise ValueError(f"이미지 없음. (확장자: {exts})\n경로: {img_dir}")

    images, valid_paths = [], []
    for p in paths:
        try:
            images.append(Image.open(p).convert("RGB"))
            valid_paths.append(p)
        except Exception as e:
            print(f"[WARN] 로드 실패: {p} - {e}")

    if not valid_paths:
        raise ValueError("모든 이미지 로드 실패.")

    return images, valid_paths


# 이미지 임베딩 추출
def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: List[Image.Image],
    device: str,
    batch_size: int = 16
) -> torch.Tensor:
    all_embeds = []

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        pixel_values = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)

        with torch.no_grad():
            # pooler_output: [CLS] 토큰의 projected 표현 (512차원)
            vision_out = model.vision_model(pixel_values=pixel_values)
            pooled = vision_out.pooler_output  # (B, hidden)
            embeds = model.visual_projection(pooled)  # (B, 512)

        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_embeds.append(embeds)

    return torch.cat(all_embeds, dim=0)


# 텍스트 임베딩 추출
def encode_text(
    model: CLIPModel,
    processor: CLIPProcessor,
    query: str,
    device: str
) -> torch.Tensor:
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_out = model.text_model(**inputs)

        pooled = text_out.last_hidden_state[
            torch.arange(text_out.last_hidden_state.shape[0]),
            inputs["input_ids"].argmax(dim=-1)
        ]
        embeds = model.text_projection(pooled)  # (1, 512)

    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds


# CLIP으로 텍스트 쿼리와 유사한 이미지 상위 K개 반환
def clip_retrieval(
    image_dir: str,
    query: str,
    top_k: int = 3,
    batch_size: int = 16,
    model_name: str = "openai/clip-vit-base-patch32"
) -> List[Dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device}")

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    images, paths = load_images_from_dir(image_dir)
    print(f"로드된 이미지 수: {len(images)}")

    img_emb = encode_images(model, processor, images, device, batch_size)
    txt_emb = encode_text(model, processor, query, device)

    # CLIP 논문과 동일한 scaled cosine similarity
    logit_scale = model.logit_scale.exp().clamp(max=100)
    scores = (logit_scale * img_emb @ txt_emb.T).squeeze(1)

    k = min(top_k, len(paths))
    top_scores, top_idx = torch.topk(scores, k=k)

    if device == "cuda":
        torch.cuda.empty_cache()

    return [
        {"path": paths[i], "score": float(s), "filename": os.path.basename(paths[i])}
        for s, i in zip(top_scores.tolist(), top_idx.tolist())
    ]


# 이미지 시각화
def show_results(results: List[Dict], query: str, cols: int = 3, figsize: tuple = (12, 4)):
    k = len(results)
    cols = min(cols, k)
    rows = (k + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    fig.suptitle(f'Query: "{query}"', fontsize=12, fontweight="bold")

    for j, r in enumerate(results):
        ax = axes[j // cols][j % cols]
        ax.imshow(Image.open(r["path"]).convert("RGB"))
        ax.set_title(f"Top{j+1} | {r['score']:.2f}\n{r['filename']}", fontsize=9)
        ax.axis("off")

    for j in range(k, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_dir = "./images"
    query = "a person riding a bicycle"
    top_k = 3

    # 한글 쿼리 사용 모델 교체
    # model_name = "Bingsu/clip-vit-large-patch14-ko"

    results = clip_retrieval(image_dir, query, top_k=top_k)

    print(f"\nQuery: {query}")
    print("=" * 60)
    for idx, r in enumerate(results, start=1):
        print(f"Top{idx}: {r['filename']:30s} score={r['score']:.4f}")

    show_results(results, query, cols=3, figsize=(12, 4))
