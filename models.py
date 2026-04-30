import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import streamlit as st
from transformers import CLIPProcessor, CLIPModel

# ── Emotion definitions ────────────────────────────────────────────────────────

EMOTIONS = [
    "Awe", "Amusement", "Sadness", "Fear",
    "Anger", "Disgust", "Contentment", "Excitement", "Something Else"
]

EMOTION_PROMPTS = {
    "Awe":          "This artwork evokes a profound sense of awe, wonder, and reverence.",
    "Amusement":    "This artwork feels joyful, playful, and amusing with lively energy.",
    "Sadness":      "This artwork conveys deep sadness, melancholy, and sorrow.",
    "Fear":         "This artwork instils dread, fear, and a sense of dark unease.",
    "Anger":        "This artwork expresses intense anger, aggression, and raw tension.",
    "Disgust":      "This artwork provokes feelings of disgust, revulsion, and discomfort.",
    "Contentment":  "This artwork radiates peaceful contentment, calm, and quiet harmony.",
    "Excitement":   "This artwork pulses with excitement, dynamic energy, and anticipation.",
    "Something Else": "This artwork evokes a complex, ambiguous, and hard-to-name emotion.",
}

CAPTIONS = {
    "Awe":          "The expansive composition and dramatic lighting evoke a profound sense of awe and wonder — as if the viewer is confronted with something vast and transcendent.",
    "Amusement":    "The bright, contrasting colours and dynamic elements create a joyful, amusing atmosphere that invites lighthearted engagement.",
    "Sadness":      "Muted tones and solitary figures cast a melancholic mood — the piece invites quiet reflection on loss and longing.",
    "Fear":         "Dark, unsettling imagery and sharp tonal contrasts instil a creeping sense of apprehension and dread.",
    "Anger":        "Aggressive brushwork and intense reds visually communicate deep-seated anger and confrontational tension.",
    "Disgust":      "Unconventional forms and jarring textures unsettle the viewer, provoking feelings of discomfort and unease.",
    "Contentment":  "Soft pastels and harmonious compositional balance induce a feeling of peaceful contentment and inner calm.",
    "Excitement":   "Vivid contrasts and energetic line work generate a palpable sense of excitement and anticipation.",
    "Something Else": "The artwork resists easy categorisation, evoking a layered and ambiguous emotional response.",
}

EMOTION_COLORS = {
    "Awe":          "#818cf8",
    "Amusement":    "#34d399",
    "Contentment":  "#2dd4bf",
    "Excitement":   "#fbbf24",
    "Sadness":      "#60a5fa",
    "Disgust":      "#a3e635",
    "Fear":         "#c084fc",
    "Anger":        "#f87171",
    "Something Else": "#9ca3af",
}


# ── Safe CLIP embedding helpers ────────────────────────────────────────────────
# These bypass get_image_features() / get_text_features() entirely and
# call the sub-models directly, which works across all transformers versions.

def _encode_image(model, processor, image: Image.Image, device: str):
    """
    Encodes a PIL image into a normalised CLIP embedding.
    Uses model.vision_model + model.visual_projection directly to avoid
    the BaseModelOutputWithPooling.norm attribute error in newer transformers.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
        # pooler_output is the [CLS] token embedding — shape (1, hidden_size)
        pooled = vision_outputs.pooler_output
        # Project into shared CLIP embedding space
        projected = model.visual_projection(pooled)
    return F.normalize(projected, dim=-1)


def _encode_texts(model, processor, texts: list, device: str):
    """
    Encodes a list of strings into normalised CLIP text embeddings.
    Uses model.text_model + model.text_projection directly.
    """
    inputs = processor(
        text=texts, return_tensors="pt",
        padding=True, truncation=True, max_length=77
    ).to(device)
    with torch.no_grad():
        text_outputs = model.text_model(**inputs)
        # pooler_output is the [EOS] token embedding — shape (N, hidden_size)
        pooled = text_outputs.pooler_output
        projected = model.text_projection(pooled)
    return F.normalize(projected, dim=-1)


# ── Model loading ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading CLIP model…")
def load_clip_model():
    """
    Loads CLIP ViT-B/32 from Hugging Face.
    Cached so it only downloads and loads once per session.
    """
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor, device


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_image_features(image: Image.Image, model, processor, device):
    """
    Extracts and L2-normalises CLIP image embeddings.
    Kept for backward compatibility with existing callers in app.py.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    return _encode_image(model, processor, image, device)


def predict_emotions(image: Image.Image, model, processor, device):
    """
    Zero-shot emotion classification using CLIP cosine similarity.

    Scores each of the 9 emotion prompts against the image embedding
    and returns a sorted list of dicts:
        [{"emotion": str, "probability": float, "similarity": float}, ...]
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Image embedding — (1, D)
    img_feat = _encode_image(model, processor, image, device)

    # Text embeddings — (9, D)
    prompts   = list(EMOTION_PROMPTS.values())
    txt_feats = _encode_texts(model, processor, prompts, device)

    # Cosine similarities — (9,)
    sims = (img_feat @ txt_feats.T).squeeze(0)

    # Temperature-scaled softmax → probabilities
    # Dividing by 10 softens the distribution so minority emotions
    # don't collapse to near-zero probability
    logit_scale = model.logit_scale.exp().item()
    probs = F.softmax(sims * logit_scale / 10, dim=0)

    sims_np  = sims.cpu().float().numpy()
    probs_np = probs.cpu().float().numpy()

    results = [
        {
            "emotion":     emotion,
            "probability": float(probs_np[i]),
            "similarity":  float(sims_np[i]),
        }
        for i, emotion in enumerate(EMOTIONS)
    ]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def predict_emotions_mock(image_features):
    """Kept for backward compatibility — returns random probabilities."""
    import random
    raw    = [random.random() for _ in EMOTIONS]
    total  = sum(raw)
    normed = [p / total for p in raw]
    results = [{"emotion": e, "probability": p, "similarity": 0.0}
               for e, p in zip(EMOTIONS, normed)]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def generate_caption(top_emotion: str, similarity: float):
    """
    Returns (caption_str, qualifier_str) for the predicted emotion.
    Qualifier reflects how strongly CLIP matched the emotion prompt.
    """
    base = CAPTIONS.get(top_emotion, CAPTIONS["Something Else"])
    if similarity >= 0.28:
        qualifier = "strongly"
    elif similarity >= 0.22:
        qualifier = "clearly"
    elif similarity >= 0.18:
        qualifier = "subtly"
    else:
        qualifier = "faintly"
    return base, qualifier


def generate_caption_mock(image_features, top_emotion: str):
    """Kept for backward compatibility."""
    return CAPTIONS.get(top_emotion, CAPTIONS["Something Else"])