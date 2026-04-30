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

# One descriptive prompt per emotion — more specific prompts give better
# CLIP similarity scores than bare emotion labels alone
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

# Affective captions — shown alongside the predicted emotion
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


# ── Model loading ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading CLIP model…")
def load_clip_model():
    """
    Loads CLIP ViT-B/32 from Hugging Face.
    Cached so it only loads once per Streamlit session.
    """
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor, device


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_image_features(image: Image.Image, model, processor, device):
    """
    Extracts and L2-normalises CLIP image embeddings.
    Identical signature to original — existing callers unchanged.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    return F.normalize(feats, dim=-1)


# ── Zero-shot emotion classification (replaces mock) ──────────────────────────

def predict_emotions(image: Image.Image, model, processor, device):
    """
    Real zero-shot emotion classification using CLIP similarity.

    Encodes the image and all emotion text prompts into the shared
    CLIP embedding space, then ranks emotions by cosine similarity.

    Returns a sorted list of dicts:
        [{"emotion": str, "probability": float, "similarity": float}, ...]

    No trained model required.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # ── Image embedding ────────────────────────────────────────────────────
    img_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_feat = F.normalize(model.get_image_features(**img_inputs), dim=-1)

    # ── Text embeddings for all emotion prompts ────────────────────────────
    prompts    = list(EMOTION_PROMPTS.values())
    txt_inputs = processor(
        text=prompts, return_tensors="pt",
        padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        txt_feats = F.normalize(model.get_text_features(**txt_inputs), dim=-1)

    # ── Cosine similarities → softmax probabilities ────────────────────────
    # Temperature scaling (logit_scale) sharpens the distribution
    logit_scale = model.logit_scale.exp().item()
    sims        = (img_feat @ txt_feats.T).squeeze(0)          # (9,)
    probs       = F.softmax(sims * logit_scale / 10, dim=0)    # (9,)

    sims_np  = sims.cpu().numpy()
    probs_np = probs.cpu().numpy()

    results = [
        {
            "emotion":    emotion,
            "probability": float(probs_np[i]),
            "similarity":  float(sims_np[i]),
        }
        for i, emotion in enumerate(EMOTIONS)
    ]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def predict_emotions_mock(image_features):
    """
    Kept for backward compatibility.
    Now returns a notice — use predict_emotions() for real results.
    """
    import random
    raw   = [random.random() for _ in EMOTIONS]
    total = sum(raw)
    normed = [p / total for p in raw]
    results = [{"emotion": e, "probability": p}
               for e, p in zip(EMOTIONS, normed)]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


# ── Caption generation ─────────────────────────────────────────────────────────

def generate_caption(top_emotion: str, similarity: float):
    """
    Returns an affective caption for the predicted emotion.
    Includes a confidence qualifier based on the CLIP similarity score.
    """
    base = CAPTIONS.get(top_emotion, CAPTIONS["Something Else"])

    # Add a confidence qualifier so the user knows how certain CLIP is
    if similarity >= 0.28:
        qualifier = "strongly"
    elif similarity >= 0.22:
        qualifier = "clearly"
    elif similarity >= 0.18:
        qualifier = "subtly"
    else:
        qualifier = "faintly"

    return base, qualifier


def generate_caption_mock(image_features, top_emotion):
    """Kept for backward compatibility."""
    return CAPTIONS.get(top_emotion, CAPTIONS["Something Else"])
