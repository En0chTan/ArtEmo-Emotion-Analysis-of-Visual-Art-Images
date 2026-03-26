import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import random

# Cache the model to ensure it doesn't reload on every interaction
@st.cache_resource
def load_clip_model():
    # Load pretrained CLIP model and processor from Hugging Face
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

def extract_image_features(image: Image.Image, model, processor, device):
    """
    Extracts high-dimensional visual features from the image using the CLIP model.
    """
    # Ensure image is in RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features

def predict_emotions_mock(image_features):
    """
    Mock function to simulate emotion classification based on extracted features.
    In a real system, this would feed `image_features` into a trained classifier (e.g., an MLP).
    Returns a sorted list of dictionaries with emotion labels and probabilities.
    """
    emotions = ["Awe", "Amusement", "Sadness", "Fear", "Anger", "Disgust", "Contentment"]
    
    # Generate random probabilities and normalize them to sum to 1.0 (Mocking softmax)
    raw_probs = [random.random() for _ in emotions]
    total = sum(raw_probs)
    normalized_probs = [p / total for p in raw_probs]
    
    # Create the result dictionary
    results = [{"emotion": em, "probability": prob} for em, prob in zip(emotions, normalized_probs)]
    
    # Sort by probability descending
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results

def generate_caption_mock(image_features, top_emotion):
    """
    Mock function to simulate affective caption generation based on features and predicted emotion.
    In a real system, this would use a generative language model (e.g., an LLM or encoder-decoder) 
    conditioned on the image features and the sentiment/emotion label.
    """
    mock_captions = {
        "Awe": "The expansive composition and dramatic lighting evoke a profound sense of awe and wonder.",
        "Amusement": "The bright, contrasting colors and dynamic elements create a joyful and amusing atmosphere.",
        "Sadness": "Muted tones and solitary figures cast a melancholic and sorrowful mood over the piece.",
        "Fear": "The dark, unsettling imagery and sharp contrasts instill a sense of apprehension and fear.",
        "Anger": "Aggressive brushstrokes and intense reds visually communicate deep-seated anger and tension.",
        "Disgust": "Unconventional forms and jarring textures provoke feelings of discomfort and disgust.",
        "Contentment": "Soft pastels and harmonious balance induce a feeling of peaceful contentment."
    }
    
    # Return the mapped caption, or a default fallback
    return mock_captions.get(top_emotion, "The artwork evokes a complex blend of emotional responses.")
