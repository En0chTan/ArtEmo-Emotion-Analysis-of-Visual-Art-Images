# ArtSense AI Implementation Plan

This document outlines the technical approach to building the MVP (Minimum Viable Product) for the ArtSense AI Streamlit application.

## Proposed Changes

We will create a new project in `C:\Users\enoch\.gemini\antigravity\scratch\artsense-ai`.

### Core Application Files

#### [NEW] `requirements.txt`
Dependencies required for the project:
`streamlit`, `torch`, `transformers`, `Pillow`, `pandas`, `altair`

#### [NEW] `app.py`
The main Streamlit application file holding the UI structure and user interaction flow.
- Configures the page title and layout.
- Provides a file uploader for images (`png`, `jpg`, `jpeg`).
- Displays the uploaded image and a loading state while processing.
- Shows the resulting emotion breakdown in a chart.
- Displays the generated affective caption.

#### [NEW] `models.py`
The module responsible for initializing AI models and processing the image.
- **Model Loader**: Uses Streamlit's `@st.cache_resource` to load the Hugging Face CLIP model (`openai/clip-vit-base-patch32`) once and prevent reloading on every interaction.
- **Feature Extraction**: Function to run the image through the CLIP vision model.
- **Mock Emotion Classifier**: A placeholder function that simulates emotion prediction by returning fixed or randomized probability distributions across categories (Awe, Amusement, Sadness, Fear, Anger, Disgust) for the MVP.
- **Mock Caption Generator**: A placeholder function returning a sample affective caption based on the top predicted emotion.

## Verification Plan

### Manual Verification
1.  Run the application using `streamlit run app.py` from the project directory.
2.  Open the local web URL.
3.  Upload various test images.
4.  Verify that the image is displayed correctly.
5.  Verify that the models run (or mock models return data) and the progress/spinner works.
6.  Verify that the visualization chart correctly renders the emotion probabilities.
7.  Check that the UI looks clean, modern, and fits the concept.
