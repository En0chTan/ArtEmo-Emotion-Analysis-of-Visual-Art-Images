import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import models

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ArtEmo",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300;1,400&family=Outfit:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    font-weight: 300;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 2rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0a0a0f !important; border-right: 1px solid #1a1a2e; }
[data-testid="stSidebar"] * { color: #c8c8d4 !important; }
[data-testid="stSidebar"] hr { border-color: #1e1e30 !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1.2rem; }

.step-pill {
    display: flex; align-items: flex-start; gap: 10px;
    margin: 0.45rem 0; padding: 0.5rem 0.7rem;
    background: #0f0f1a; border-radius: 8px;
    border: 1px solid #1e1e30;
    font-size: 0.79rem; line-height: 1.5; color: #9090a8 !important;
}
.step-num {
    min-width: 20px; height: 20px; background: #6366f1;
    border-radius: 4px; display: flex; align-items: center;
    justify-content: center; font-size: 0.7rem; font-weight: 500;
    color: #fff !important; flex-shrink: 0; margin-top: 1px;
}

/* ── Dark museum header ── */
.museum-header {
    background: #0f0f14; border-radius: 14px;
    padding: 26px 30px 22px; margin-bottom: 1.4rem;
    border: 1px solid #1e1e2e;
}
.museum-eyebrow {
    font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase;
    color: #6366f1; font-weight: 500; margin: 0 0 10px;
    font-family: 'Outfit', sans-serif;
}
.museum-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.2rem; font-weight: 300;
    color: #f0f0f8; margin: 0 0 2px; line-height: 1.0;
    letter-spacing: -0.02em;
}
.museum-title em { color: #818cf8; font-style: italic; }
.museum-rule { width: 40px; height: 1px; background: #6366f1;
               margin: 10px 0; border: none; }
.museum-sub {
    font-size: 0.9rem; color: #6b7280; margin: 0 0 16px;
    font-weight: 300; font-family: 'Outfit', sans-serif;
}
.museum-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.museum-chip {
    font-size: 11px; padding: 3px 10px; border-radius: 4px;
    border: 1px solid #2a2a3e; color: #9090a8;
    font-family: 'Outfit', sans-serif; background: #16161f;
}

/* ── Section heads ── */
.section-head {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem; font-weight: 400; color: #e0e0f0;
    border-bottom: 1px solid #1e1e30;
    padding-bottom: 0.35rem; margin: 1.2rem 0 0.8rem;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #2a2a3e !important;
    border-radius: 12px !important; background: #0f0f14 !important;
}
[data-testid="stFileUploader"]:hover { border-color: #6366f1 !important; }
[data-testid="stFileUploader"] * { color: #9090a8 !important; }

/* ── Emotion hero card ── */
.emotion-hero {
    border-radius: 14px; padding: 1.6rem 1.4rem;
    border: 1px solid #1e1e30; background: #0f0f14;
    margin-bottom: 1rem; text-align: center;
}
.emotion-icon { font-size: 2rem; margin-bottom: 0.3rem; }
.emotion-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem; font-weight: 300;
    letter-spacing: 0.02em; margin: 0; line-height: 1;
}
.confidence-row {
    display: flex; align-items: center; justify-content: center;
    gap: 8px; margin-top: 0.5rem;
}
.conf-bar-track {
    width: 120px; height: 3px;
    background: #1e1e30; border-radius: 99px; overflow: hidden;
}
.conf-bar-fill { height: 100%; border-radius: 99px; }
.conf-label { font-size: 0.78rem; color: #6b7280; }
.qualifier-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.72rem; font-weight: 500; margin-top: 6px;
    background: #16161f; border: 1px solid #2a2a3e; color: #9090a8;
}

/* ── Caption box ── */
.caption-box {
    background: #0f0f14; border-left: 3px solid #6366f1;
    padding: 1.1rem 1.3rem; border-radius: 0 10px 10px 0;
    margin: 0.8rem 0; font-family: 'Cormorant Garamond', serif;
    font-style: italic; font-size: 1.12rem;
    color: #c8c8e0; line-height: 1.7;
    border-top: 1px solid #1e1e30; border-right: 1px solid #1e1e30;
    border-bottom: 1px solid #1e1e30;
}
.caption-meta {
    font-size: 0.72rem; color: #4b4b6a;
    font-family: 'Outfit', sans-serif;
    font-style: normal; margin-top: 0.5rem;
}

/* ── Zero-shot info banner ── */
.zeroshot-banner {
    display: flex; align-items: flex-start; gap: 10px;
    background: #0f0f1f; border: 1px solid #2a2a3e;
    border-radius: 10px; padding: 10px 14px;
    font-size: 0.78rem; color: #6b7280; margin-bottom: 1rem;
    line-height: 1.6;
}
.zs-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 3.5rem 1rem;
    border: 1.5px dashed #1e1e30; border-radius: 14px;
    background: #0a0a0f; margin-top: 0.5rem;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid #1e1e30; gap: 0; }
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 400 !important;
    color: #4b4b6a !important; padding: 0.5rem 1rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #818cf8 !important; border-bottom-color: #818cf8 !important;
}
[data-testid="stSpinner"] p { font-size: 0.82rem; color: #4b4b6a; }
</style>
""", unsafe_allow_html=True)


# ── Emotion icon map ───────────────────────────────────────────────────────────
EMOTION_ICONS = {
    "Awe": "✦", "Amusement": "◉", "Contentment": "◈",
    "Excitement": "◆", "Sadness": "◇", "Disgust": "◎",
    "Fear": "▲", "Anger": "◼", "Something Else": "○",
}


# ── Plotly chart ───────────────────────────────────────────────────────────────
def make_emotion_chart(results: list) -> go.Figure:
    df     = pd.DataFrame(results).sort_values("probability")
    colors = [models.EMOTION_COLORS.get(e, "#9ca3af") for e in df["emotion"]]

    fig = go.Figure(go.Bar(
        x=df["probability"], y=df["emotion"], orientation="h",
        marker=dict(color=colors, opacity=0.9),
        text=[f"{p:.1%}" for p in df["probability"]],
        textposition="outside",
        textfont=dict(size=11, family="Outfit", color="#9090a8"),
        cliponaxis=False,
    ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=65, t=6, b=6),
        xaxis=dict(
            range=[0, min(df["probability"].max() * 1.3, 1)],
            showticklabels=False, showgrid=False, zeroline=False
        ),
        yaxis=dict(
            tickfont=dict(size=11, family="Outfit", color="#9090a8"),
            showgrid=False
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def make_similarity_radar(results: list) -> go.Figure:
    """Radar chart showing raw CLIP similarity scores per emotion."""
    ems  = [r["emotion"] for r in results]
    sims = [r.get("similarity", 0) for r in results]

    fig = go.Figure(go.Scatterpolar(
        r=sims + [sims[0]],
        theta=ems + [ems[0]],
        fill='toself',
        fillcolor='rgba(99,102,241,0.15)',
        line=dict(color='#818cf8', width=2),
        marker=dict(color='#818cf8', size=5),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, showticklabels=False,
                gridcolor='#1e1e30', linecolor='#1e1e30'
            ),
            angularaxis=dict(
                tickfont=dict(size=9, color='#9090a8'),
                gridcolor='#1e1e30', linecolor='#1e1e30'
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=300, margin=dict(l=30, r=30, t=20, b=20),
        font=dict(family='Outfit'),
        showlegend=False,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<p style="font-family:\'Cormorant Garamond\',serif;font-size:1.7rem;'
            'font-weight:300;color:#e8e8f0;margin:0 0 0.1rem;">ArtEmo AI</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="font-size:0.78rem;color:#4b4b6a;margin:0 0 1rem;">'
            'Zero-shot emotion interpretation via CLIP</p>',
            unsafe_allow_html=True
        )
        st.markdown("---")

        for num, text in [
            ("1", "Upload any WikiArt or artwork image"),
            ("2", "CLIP encodes the image into a shared embedding space"),
            ("3", "9 emotion prompts are compared via cosine similarity"),
            ("4", "The best-matching emotion + caption is displayed"),
        ]:
            st.markdown(
                f'<div class="step-pill">'
                f'<span class="step-num">{num}</span>{text}</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Prompt style selector
        st.markdown(
            '<p style="font-size:0.78rem;color:#6b7280;margin:0 0 0.4rem;">'
            'Prompt style</p>', unsafe_allow_html=True
        )
        prompt_style = st.radio(
            "", ["Descriptive (default)", "Short label only"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.72rem;color:#2a2a3e;line-height:1.7;">'
            'CLIP ViT-B/32 · openai/clip-vit-base-patch32<br>'
            'ArtEmis v2.0 · 9 emotion classes<br>'
            'FYP02-DS · MMU · 2026</p>',
            unsafe_allow_html=True
        )

    # ── Dark museum header ────────────────────────────────────────────────
    st.markdown("""
    <div class="museum-header">
        <p class="museum-eyebrow">
            Emotion Analysis &nbsp;·&nbsp; Visual Art &nbsp;
        </p>
        <h1 class="museum-title">Art<em>Emo</em></h1>
        <hr class="museum-rule">
        <p class="museum-sub">
            Discover the emotional resonance of visual art through AI
        </p>
        <div class="museum-chips">
            <span class="museum-chip">CLIP ViT-B/32</span>
            <span class="museum-chip">9 Emotion Classes</span>
            <span class="museum-chip">ArtEmis v2.0</span>
            <span class="museum-chip">WikiArt</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── File uploader ─────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload an artwork — JPG, JPEG or PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    # ── Analysis ──────────────────────────────────────────────────────────
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            col1, col2 = st.columns([1, 1], gap="large")

            # ── Left: artwork ──────────────────────────────────────────
            with col1:
                st.markdown('<div class="section-head">Uploaded artwork</div>',
                            unsafe_allow_html=True)
                st.image(image, caption=uploaded_file.name,
                         use_container_width=True)

            # ── Right: analysis ────────────────────────────────────────
            with col2:
                st.markdown('<div class="section-head">AI emotional analysis</div>',
                            unsafe_allow_html=True)

                # Zero-shot explanation banner
                st.markdown("""
                <div class="zeroshot-banner">
                    <span class="zs-icon">🔍</span>
                    <span>
                        <strong style="color:#818cf8;">Zero-shot mode</strong> —
                        CLIP compares your image against 9 emotion text prompts
                        directly in the shared embedding space. No fine-tuned
                        classifier is used. Results reflect CLIP's visual-language
                        understanding of the artwork.
                    </span>
                </div>
                """, unsafe_allow_html=True)

                with st.spinner("Loading CLIP… (first run only)"):
                    clip_model, processor, device = models.load_clip_model()

                with st.spinner("Analysing emotional content…"):
                    # Extract image features (kept for API compatibility)
                    features = models.extract_image_features(
                        image, clip_model, processor, device
                    )

                    # Real zero-shot classification
                    results  = models.predict_emotions(
                        image, clip_model, processor, device
                    )

                top_emotion  = results[0]["emotion"]
                top_prob     = results[0]["probability"]
                top_sim      = results[0].get("similarity", 0.0)
                caption, qualifier = models.generate_caption(top_emotion, top_sim)

                # Emotion hero card
                color = models.EMOTION_COLORS.get(top_emotion, "#9ca3af")
                icon  = EMOTION_ICONS.get(top_emotion, "○")

                st.markdown(f"""
                <div class="emotion-hero"
                     style="border-color:{color}30;background:{color}08;">
                    <div class="emotion-icon" style="color:{color};">{icon}</div>
                    <p class="emotion-name" style="color:{color};">{top_emotion}</p>
                    <div class="confidence-row">
                        <div class="conf-bar-track">
                            <div class="conf-bar-fill"
                                 style="width:{top_prob*100:.0f}%;
                                        background:{color};"></div>
                        </div>
                        <span class="conf-label">{top_prob:.1%} confidence</span>
                    </div>
                    <div>
                        <span class="qualifier-badge">
                            CLIP similarity: {top_sim:.3f} · {qualifier} evoked
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Affective caption
                st.markdown('<div class="section-head">Affective interpretation</div>',
                            unsafe_allow_html=True)
                st.markdown(f"""
                <div class="caption-box">
                    "{caption}"
                    <div class="caption-meta">
                        CLIP zero-shot · ViT-B/32 · similarity {top_sim:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Charts in tabs
                st.markdown('<div class="section-head">Emotion analysis</div>',
                            unsafe_allow_html=True)

                tab_bar, tab_radar, tab_table = st.tabs([
                    "Distribution", "Radar", "All scores"
                ])

                with tab_bar:
                    st.plotly_chart(
                        make_emotion_chart(results),
                        use_container_width=True,
                        config={"displayModeBar": False}
                    )

                with tab_radar:
                    st.plotly_chart(
                        make_similarity_radar(results),
                        use_container_width=True,
                        config={"displayModeBar": False}
                    )

                with tab_table:
                    df = pd.DataFrame([{
                        "Emotion":    r["emotion"],
                        "Probability": f"{r['probability']:.1%}",
                        "CLIP similarity": f"{r.get('similarity', 0):.4f}",
                    } for r in results])
                    st.dataframe(
                        df, use_container_width=True, hide_index=True
                    )

        except Exception as e:
            st.error(f"An error occurred: {e}")

    else:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:2rem;margin-bottom:0.6rem;color:#2a2a3e;">◎</div>
            <div style="font-family:'Cormorant Garamond',serif;font-size:1.2rem;
                        font-style:italic;color:#4b4b6a;">
                Upload an artwork above to begin analysis
            </div>
            <div style="font-size:0.8rem;margin-top:0.5rem;color:#2a2a3e;">
                JPG · JPEG · PNG
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
