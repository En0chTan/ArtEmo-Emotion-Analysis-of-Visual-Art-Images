import streamlit as st
import pandas as pd
from PIL import Image
import models

# Configure the Streamlit page layout and title
st.set_page_config(
    page_title="ArtEmo AI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .caption-box {
        background-color: #f8f9fa;
        border-left: 5px solid #3498db;
        padding: 1.5rem;
        border-radius: 5px;
        margin-top: 1rem;
        font-style: italic;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3212/3212608.png", width=100)
        st.title("ArtEmo AI")
        st.write("Emotion-Aware Artwork Interpretation System.")
        st.markdown("---")
        st.write("**How it works:**")
        st.write("1. Upload an artwork image.")
        st.write("2. AI extracts visual features using a CLIP model.")
        st.write("3. Our classification model predicts the primary evoked emotion.")
        st.write("4. An affective caption is generated explaining the emotional interpretation.")
        st.markdown("---")
        st.info("Note: This is a prototype demonstrating multimodal AI integrating vision and language.")

    # Main Content Area
    st.markdown('<p class="main-header">🎨 ArtEmo AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover the emotional resonance of visual art through AI.</p>', unsafe_allow_html=True)

    # File Uploader
    uploaded_file = st.file_uploader("Upload an artwork...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Load the image
            image = Image.open(uploaded_file)
            
            # Create two columns for layout: left for image, right for analysis
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Uploaded Artwork")
                # Using use_container_width instead of use_column_width for modern Streamlit versions
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
            with col2:
                st.subheader("🤖 AI Emotional Analysis")
                
                with st.spinner('Loading AI models... (This may take a moment on first run)'):
                    clip_model, processor, device = models.load_clip_model()
                    
                with st.spinner('Extracting visual features and predicting emotions...'):
                    # 1. Feature Extraction
                    features = models.extract_image_features(image, clip_model, processor, device)
                    
                    # 2. Emotion Prediction (Mock)
                    emotion_results = models.predict_emotions_mock(features)
                    top_emotion = emotion_results[0]["emotion"]
                    top_prob = emotion_results[0]["probability"]
                    
                    # 3. Affective Caption Generation (Mock)
                    caption = models.generate_caption_mock(features, top_emotion)
                    
                # Display Results
                st.success(f"**Primary Emotion Detected:** {top_emotion} ({top_prob:.1%})")
                
                # Display Caption
                st.markdown("### Affective Interpretation")
                st.markdown(f'<div class="caption-box">"{caption}"</div>', unsafe_allow_html=True)
                
                # Display Visualization Chart
                st.markdown("### Emotion Probability Distribution")
                
                # Convert results to DataFrame for charting
                df = pd.DataFrame(emotion_results)
                
                # Create horizontal bar chart using Streamlit native charts
                st.bar_chart(
                    data=df.set_index("emotion"),
                    horizontal=True,
                    height=300
                )
                
        except Exception as e:
            st.error(f"An error occurred processing the image: {e}")

if __name__ == "__main__":
    main()
