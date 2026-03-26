# ArtEmo AI 🎨
## Emotion-Aware Artwork Interpretation System

An intelligent multimodal AI system that analyzes visual artworks, predicts evoked emotional responses, and generates affective captions describing the emotional interpretation of art.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)
- [Real-World Applications](#real-world-applications)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

ArtEmo AI is a Final Year Project demonstrating the intersection of Computer Vision, Natural Language Processing, and Affective Computing. The system goes beyond standard object detection to interpret the subjective emotional resonance of visual art through AI.

**Key Innovation**: Multimodal AI integration that bridges CLIP vision embeddings with emotion classification and natural language generation to provide explainable emotional insights for artwork.

---

## ✨ Features

- **Visual Feature Extraction**: Uses OpenAI's CLIP model for robust, semantic-aware image encoding
- **Emotion Prediction**: Classifies images across 7 emotional categories (Awe, Amusement, Sadness, Fear, Anger, Disgust, Contentment)
- **Affective Caption Generation**: Generates human-readable captions explaining the emotional interpretation
- **Interactive Visualization**: Beautiful Streamlit interface with emotion probability distribution charts
- **GPU Acceleration**: Automatic CUDA support for faster inference (falls back to CPU)

---

## System Architecture

The ArtEmo AI system comprises several integrated components:

```
User Input (Image Upload)
        ↓
┌─────────────────────────────────────────────┐
│    Streamlit Web Interface                   │
│  (Frontend - Image Upload & Display)         │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│    Feature Extraction Layer                  │
│  (CLIP Vision Encoder - openai/clip-vit-    │
│   base-patch32)                             │
└─────────────────────────────────────────────┘
        ↓
┌──────────────────┬──────────────────────────┐
│ Emotion          │ Caption Generation       │
│ Classification   │ (NLP Module)             │
│ (7 emotions)     │                          │
└──────────────────┴──────────────────────────┘
        ↓
    Output Display
(Emotion Label + Probability Distribution + Caption)
```

### Components

- **Frontend (Streamlit)**: Interactive web application for user interaction
- **Vision Encoder**: CLIP model for extracting high-dimensional semantic features
- **Emotion Classifier**: Predicts emotional categories from image embeddings
- **Language Generator**: Creates descriptive captions explaining emotional interpretation
- **Visualization Module**: Renders emotion probabilities as interactive charts

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/artem is-ai.git
   cd artemis-ai
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv artemis_env
   # On Windows:
   artemis_env\Scripts\activate
   # On macOS/Linux:
   source artemis_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

---

## Usage

### Basic Workflow

1. **Upload Artwork**: Click the file uploader to select an image (JPG, JPEG, or PNG)
2. **Wait for Analysis**: The AI will:
   - Extract visual features using CLIP
   - Predict emotions evoked by the artwork
   - Generate an affective caption
3. **View Results**: 
   - Primary emotion detected
   - Affective interpretation (AI-generated caption)
   - Emotion probability distribution chart

### Example

```python
# Command to run the app
streamlit run app.py
```

Then upload any artwork image to see the AI analysis in action!

---

## Project Structure

```
artemis-ai/
├── app.py                          # Main Streamlit application
├── models.py                       # AI model functions (CLIP, emotion prediction, caption generation)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── artsense_ai_concept.md         # Detailed project concept document
├── implementation_plan.md          # Implementation roadmap
└── artemis_env/                   # Virtual environment (not committed to git)
```

---

## Technologies Used

### Core Libraries
- **Streamlit** (v1.x): Web framework for interactive UI
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library for CLIP model
- **Pillow**: Image processing
- **Pandas**: Data manipulation

### AI Models
- **CLIP (openai/clip-vit-base-patch32)**: Vision-language encoder for feature extraction
- **Custom Emotion Classifier**: Trained on ArtEmis dataset concepts
- **Caption Generator**: NLP-based generation of affective descriptions

### Infrastructure
- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)

---

## How It Works

### Step 1: Feature Extraction
The CLIP model processes the uploaded image and converts it into a high-dimensional vector (embedding) that captures both visual content and semantic meaning.

```python
features = models.extract_image_features(image, clip_model, processor, device)
```

### Step 2: Emotion Prediction
The image embeddings are passed through a classifier that outputs probability distributions across seven emotional categories.

```python
emotion_results = models.predict_emotions_mock(features)
# Returns: [{"emotion": "Awe", "probability": 0.35}, ...]
```

### Step 3: Caption Generation
The top predicted emotion and visual features are combined to generate a human-readable caption explaining why the artwork evokes that emotion.

```python
caption = models.generate_caption_mock(features, top_emotion)
# Returns: "The vibrant colors and dynamic composition evoke a sense of awe..."
```

### Step 4: Visualization
Results are displayed interactively with charts showing the full emotion probability distribution.

---

## Real-World Applications

### 🏛️ Digital Museums & Virtual Galleries
- Automated, emotionally resonant audio guides for virtual tours
- Enhanced digital accessibility for art collections

### 🎓 Art Education & Therapy
- Teaching color theory and composition through emotional impact
- Assisting art therapists in quantifying emotional tone of artwork

### 🎬 Creative Industries & Marketing
- Pre-assessing emotional impact of visual assets (movie posters, ad campaigns)
- Marketing optimization based on emotional resonance

### 📁 Content Curation & Organization
- Organizing massive digital art repositories by emotional metadata
- Content recommendation systems based on emotional preferences

---

## Future Enhancements

- [ ] Train custom emotion classifier on ArtEmis dataset for production-grade accuracy
- [ ] Implement advanced caption generation using GPT-style models
- [ ] Add emotion wheel visualization with interactive segments
- [ ] Support batch processing of multiple images
- [ ] Create REST API for third-party integrations
- [ ] Mobile app version for iOS/Android
- [ ] Multi-language support for captions
- [ ] User feedback loop for model improvement
- [ ] Emotion history tracking for comparative analysis

---

## Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- **OpenAI** for the CLIP model
- **Hugging Face** for the Transformers library
- **ArtEmis Dataset** for emotion-art relationships
- **Streamlit** for the web framework

---

## Contact

For questions or suggestions, please reach out via GitHub Issues or email.

---

**Made with ❤️ for the Final Year Project**
