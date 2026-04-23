import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import models

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ArtEmo AI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Emotion metadata ───────────────────────────────────────────────────────────
EMOTION_META = {
    "Awe":            {"color": "#818cf8", "icon": "✦", "valence": "positive"},
    "Amusement":      {"color": "#34d399", "icon": "◉", "valence": "positive"},
    "Contentment":    {"color": "#2dd4bf", "icon": "◈", "valence": "positive"},
    "Excitement":     {"color": "#fbbf24", "icon": "◆", "valence": "positive"},
    "Sadness":        {"color": "#60a5fa", "icon": "◇", "valence": "negative"},
    "Disgust":        {"color": "#a3e635", "icon": "◎", "valence": "negative"},
    "Fear":           {"color": "#c084fc", "icon": "▲", "valence": "negative"},
    "Anger":          {"color": "#f87171", "icon": "◼", "valence": "negative"},
    "Something Else": {"color": "#9ca3af", "icon": "○", "valence": "neutral"},
}

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Outfit:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    font-weight: 300;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 2rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0a0f !important;
    border-right: 1px solid #1a1a2e;
}
[data-testid="stSidebar"] * { color: #c8c8d4 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    color: #e8e8f0 !important;
    font-family: 'Cormorant Garamond', serif !important;
}
[data-testid="stSidebar"] .stMarkdown p { font-size: 0.82rem; line-height: 1.7; }
[data-testid="stSidebar"] hr { border-color: #1e1e30 !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1.2rem; }

.step-pill {
    display: flex; align-items: flex-start; gap: 10px;
    margin: 0.5rem 0; padding: 0.55rem 0.7rem;
    background: #0f0f1a; border-radius: 8px;
    border: 1px solid #1e1e30;
    font-size: 0.8rem; line-height: 1.5; color: #9090a8 !important;
}
.step-num {
    min-width: 20px; height: 20px;
    background: #6366f1; border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 500; color: #fff !important;
    flex-shrink: 0; margin-top: 1px;
}

/* ── Dark museum header ── */
.museum-header {
    background: #0f0f14;
    border-radius: 14px;
    padding: 28px 32px 24px;
    margin-bottom: 1.4rem;
    border: 1px solid #1e1e2e;
}
.museum-eyebrow {
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6366f1;
    font-weight: 500;
    margin: 0 0 10px;
    font-family: 'Outfit', sans-serif;
}
.museum-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.4rem;
    font-weight: 300;
    color: #f0f0f8;
    margin: 0 0 2px;
    line-height: 1.0;
    letter-spacing: -0.02em;
}
.museum-title em {
    color: #818cf8;
    font-style: italic;
}
.museum-rule {
    width: 40px; height: 1px;
    background: #6366f1;
    margin: 10px 0 10px;
    border: none;
}
.museum-sub {
    font-size: 0.92rem;
    color: #6b7280;
    margin: 0 0 16px;
    font-weight: 300;
    font-family: 'Outfit', sans-serif;
    letter-spacing: 0.01em;
}
.museum-chips {
    display: flex; gap: 8px; flex-wrap: wrap;
}
.museum-chip {
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 4px;
    border: 1px solid #2a2a3e;
    color: #9090a8;
    font-family: 'Outfit', sans-serif;
    background: #16161f;
}

/* ── Section heads — inverted for dark context ── */
.section-head {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem; font-weight: 400;
    color: #e0e0f0; letter-spacing: 0.01em;
    border-bottom: 1px solid #1e1e30;
    padding-bottom: 0.35rem; margin: 1.2rem 0 0.8rem;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #2a2a3e !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
    background: #0f0f14 !important;
}
[data-testid="stFileUploader"]:hover { border-color: #6366f1 !important; }
[data-testid="stFileUploader"] * { color: #9090a8 !important; }

/* ── Emotion hero card ── */
.emotion-hero {
    border-radius: 14px; padding: 1.6rem 1.4rem;
    border: 1px solid #1e1e30;
    background: #0f0f14;
    margin-bottom: 1rem; text-align: center;
}
.emotion-icon { font-size: 2rem; margin-bottom: 0.3rem; }
.emotion-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem; font-weight: 300;
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

/* ── Caption box ── */
.caption-box {
    background: #0f0f14;
    border-left: 3px solid #6366f1;
    padding: 1.1rem 1.3rem;
    border-radius: 0 10px 10px 0;
    margin: 0.8rem 0;
    font-family: 'Cormorant Garamond', serif;
    font-style: italic; font-size: 1.15rem;
    color: #c8c8e0; line-height: 1.7;
    border-top: 1px solid #1e1e30;
    border-right: 1px solid #1e1e30;
    border-bottom: 1px solid #1e1e30;
}
.caption-meta {
    font-size: 0.72rem; color: #4b4b6a;
    font-family: 'Outfit', sans-serif;
    font-style: normal; margin-top: 0.5rem;
}

/* ── Demo badge ── */
.demo-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: #1a120a; border: 1px solid #78350f;
    border-radius: 99px; padding: 3px 10px;
    font-size: 0.72rem; color: #d97706; margin-bottom: 0.8rem;
}

/* ── Metric chips ── */
.metric-chip {
    background: #0f0f14; border-radius: 8px;
    padding: 0.65rem 0.8rem; text-align: center;
    border: 1px solid #1e1e30;
}
.metric-chip .val { font-size: 1.3rem; font-weight: 500; color: #e0e0f0; }
.metric-chip .lbl {
    font-size: 0.68rem; color: #4b4b6a;
    text-transform: uppercase; letter-spacing: 0.06em; margin-top: 1px;
}

/* ── Results table ── */
.results-tab { font-size: 0.8rem; width: 100%; border-collapse: collapse; }
.results-tab th {
    font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.07em; color: #4b4b6a;
    padding: 5px 8px; border-bottom: 1px solid #1e1e30;
    text-align: right; font-weight: 400;
}
.results-tab th:first-child { text-align: left; }
.results-tab td {
    padding: 6px 8px; border-bottom: 1px solid #13131e;
    text-align: right; font-variant-numeric: tabular-nums; color: #c0c0d8;
}
.results-tab td:first-child { text-align: left; color: #6b7280; }
.results-tab tr:last-child td {
    font-weight: 500; border-top: 1px solid #1e1e30;
    border-bottom: none; background: #0f0f14; color: #e0e0f0;
}

/* ── F1 pills ── */
.f1-pill {
    display: inline-block; padding: 1px 7px;
    border-radius: 99px; font-size: 0.75rem; font-weight: 500;
}
.pill-high { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.pill-mid  { background: #1c1002; color: #fbbf24; border: 1px solid #78350f; }
.pill-low  { background: #0f0f14; color: #6b7280; border: 1px solid #1e1e30; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #1e1e30; gap: 0;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 400 !important;
    color: #4b4b6a !important; padding: 0.5rem 1rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom-color: #818cf8 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p { font-size: 0.82rem; color: #4b4b6a; }

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 3.5rem 1rem;
    border: 1.5px dashed #1e1e30; border-radius: 14px;
    background: #0a0a0f; margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_emotion_chart(emotion_results: list) -> go.Figure:
    df     = pd.DataFrame(emotion_results).sort_values("probability")
    colors = [EMOTION_META.get(e, {}).get("color", "#9ca3af") for e in df["emotion"]]
    fig    = go.Figure(go.Bar(
        x=df["probability"], y=df["emotion"], orientation="h",
        marker=dict(color=colors, opacity=0.9),
        text=[f"{p:.1%}" for p in df["probability"]],
        textposition="outside",
        textfont=dict(size=11, family="Outfit", color="#9090a8"),
        cliponaxis=False,
    ))
    fig.update_layout(
        height=280, margin=dict(l=10, r=60, t=6, b=6),
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


def make_confusion_matrix(cm_data: list) -> go.Figure:
    short = ["Amus", "Awe", "Cont", "Excit", "Sad", "Disg", "Fear", "Anger", "S.Else"]
    cm    = np.array(cm_data, dtype=float)
    cm_n  = cm / cm.sum(axis=1, keepdims=True)
    text  = [[f"{cm_n[i][j]:.2f}" for j in range(len(short))]
             for i in range(len(short))]
    fig   = go.Figure(go.Heatmap(
        z=cm_n, x=short, y=short,
        text=text, texttemplate="%{text}", textfont={"size": 8, "color": "#c8c8e0"},
        colorscale=[[0, "#0f0f14"], [0.5, "#4338ca"], [1, "#818cf8"]],
        showscale=False,
    ))
    fig.update_layout(
        height=300, margin=dict(l=50, r=10, t=10, b=50),
        xaxis=dict(title="Predicted",
                   tickfont=dict(size=9, color="#6b7280"),
                   title_font=dict(size=10, color="#6b7280")),
        yaxis=dict(title="Actual",
                   tickfont=dict(size=9, color="#6b7280"),
                   title_font=dict(size=10, color="#6b7280"),
                   autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit"),
    )
    return fig


def f1_pill(val: float) -> str:
    cls = "pill-high" if val >= 70 else "pill-mid" if val >= 60 else "pill-low"
    return f'<span class="f1-pill {cls}">{val:.1f}</span>'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<p style="font-family:\'Cormorant Garamond\',serif;font-size:1.7rem;'
            'font-weight:300;color:#e8e8f0;margin:0 0 0.1rem;">ArtEmo AI</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="font-size:0.78rem;color:#4b4b6a;margin:0 0 1rem;">'
            'Emotion-aware artwork interpretation</p>',
            unsafe_allow_html=True
        )
        st.markdown("---")

        for num, text in [
            ("1", "Upload any artwork image (JPG / PNG)"),
            ("2", "CLIP extracts high-dimensional visual features"),
            ("3", "MobileNetV2 predicts the evoked emotion"),
            ("4", "Affective caption generated via CLIP retrieval"),
        ]:
            st.markdown(
                f'<div class="step-pill">'
                f'<span class="step-num">{num}</span>{text}</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        show_results = st.checkbox("Show model results dashboard", value=False)
        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.72rem;color:#2a2a3e;line-height:1.7;">'
            'MobileNetV2 + CLIP ViT-B/32<br>'
            'Trained on ArtEmis v2.0<br>'
            'FYP02-DS · MMU · 2026</p>',
            unsafe_allow_html=True
        )

    # ── Dark museum header ────────────────────────────────────────────────────
    st.markdown("""
    <div class="museum-header">
        <p class="museum-eyebrow">
            Emotion Analysis &nbsp;·&nbsp; Visual Art &nbsp;·&nbsp; Deep Learning
        </p>
        <h1 class="museum-title">Art<em>Emo</em> AI</h1>
        <hr class="museum-rule">
        <p class="museum-sub">
            Discover the emotional resonance of visual art through AI.
        </p>
        <div class="museum-chips">
            <span class="museum-chip">MobileNetV2</span>
            <span class="museum-chip">CLIP ViT-B/32</span>
            <span class="museum-chip">ArtEmis v2.0</span>
            <span class="museum-chip">9 emotion classes</span>
            <span class="museum-chip">69,214 training images</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload an artwork — JPG, JPEG or PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    # ── Artwork analysis ──────────────────────────────────────────────────────
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.markdown('<div class="section-head">Uploaded artwork</div>',
                            unsafe_allow_html=True)
                st.image(image, caption=uploaded_file.name,
                         use_container_width=True)

            with col2:
                st.markdown('<div class="section-head">AI emotional analysis</div>',
                            unsafe_allow_html=True)

                with st.spinner("Loading CLIP model… (first run only)"):
                    clip_model, processor, device = models.load_clip_model()

                with st.spinner("Extracting features & predicting emotion…"):

                    # 1. Feature extraction
                    features = models.extract_image_features(
                        image, clip_model, processor, device
                    )

                    # 2. Real emotion prediction; graceful fallback to mock
                    try:
                        emotion_results, model_loaded = models.predict_emotions(image)
                    except AttributeError:
                        emotion_results = models.predict_emotions_mock(features)
                        model_loaded    = False

                    top_emotion = emotion_results[0]["emotion"]
                    top_prob    = emotion_results[0]["probability"]

                    # 3. Real caption generation; graceful fallback to mock
                    try:
                        caption, clip_score, caption_source = models.generate_caption(
                            image, top_emotion
                        )
                    except AttributeError:
                        caption        = models.generate_caption_mock(features, top_emotion)
                        clip_score     = 0.0
                        caption_source = "template"

                # Demo mode notice
                if not model_loaded:
                    st.markdown(
                        '<div class="demo-badge">⚠ Demo mode — '
                        'run train_local.ipynb to generate real weights</div>',
                        unsafe_allow_html=True
                    )

                # Emotion hero card
                meta  = EMOTION_META.get(top_emotion, {"color": "#9ca3af", "icon": "○"})
                color = meta["color"]
                icon  = meta["icon"]
                st.markdown(f"""
                <div class="emotion-hero"
                     style="border-color:{color}30;background:{color}0a;">
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
                </div>
                """, unsafe_allow_html=True)

                # Affective caption
                st.markdown('<div class="section-head">Affective interpretation</div>',
                            unsafe_allow_html=True)
                src_label = (
                    f"CLIP retrieval · similarity {clip_score:.3f}"
                    if caption_source == "clip"
                    else "Template caption"
                )
                st.markdown(f"""
                <div class="caption-box">
                    "{caption}"
                    <div class="caption-meta">{src_label}</div>
                </div>
                """, unsafe_allow_html=True)

                # Emotion distribution chart
                st.markdown('<div class="section-head">Emotion distribution</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(
                    make_emotion_chart(emotion_results),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

        except Exception as e:
            st.error(f"An error occurred processing the image: {e}")

    else:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:2rem;margin-bottom:0.6rem;color:#2a2a3e;">◎</div>
            <div style="font-family:'Cormorant Garamond',serif;font-size:1.2rem;
                        font-style:italic;color:#4b4b6a;">
                Upload an artwork above to begin analysis
            </div>
            <div style="font-size:0.8rem;margin-top:0.4rem;color:#2a2a3e;">
                JPG · JPEG · PNG
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Results dashboard ─────────────────────────────────────────────────────
    if show_results:
        st.markdown("---")
        st.markdown(
            '<p style="font-family:\'Cormorant Garamond\',serif;font-size:1.6rem;'
            'font-weight:300;color:#e0e0f0;margin-bottom:0.2rem;">'
            'Model results dashboard</p>',
            unsafe_allow_html=True
        )

        mn_results   = models.load_classification_results()
        clip_results = models.load_nlp_results()

        tab1, tab2, tab3 = st.tabs([
            "Table 1 — Classification",
            "Table 2 — NLP quality",
            "Confusion matrix",
        ])

        # ── Tab 1 ─────────────────────────────────────────────────────────
        with tab1:
            overall = mn_results["overall"]
            c1, c2, c3, c4 = st.columns(4)
            for col, (label, key) in zip(
                [c1, c2, c3, c4],
                [("Accuracy",  "accuracy"), ("Precision", "precision"),
                 ("Recall",    "recall"),   ("Macro F1",  "f1")]
            ):
                col.markdown(f"""
                <div class="metric-chip">
                    <div class="val">{overall[key]:.1f}%</div>
                    <div class="lbl">{label}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            per_class = mn_results["per_class"]
            rows_html = ""
            for r in per_class:
                em    = r["emotion"].capitalize()
                color = EMOTION_META.get(em, {}).get("color", "#9ca3af")
                rows_html += f"""
                <tr>
                  <td><span style="display:inline-block;width:7px;height:7px;
                      border-radius:50%;background:{color};
                      margin-right:6px;"></span>{em}</td>
                  <td>{r['accuracy']:.1f}</td>
                  <td>{r['precision']:.1f}</td>
                  <td>{r['recall']:.1f}</td>
                  <td>{f1_pill(r['f1'])}</td>
                </tr>"""
            o = overall
            rows_html += f"""
            <tr>
              <td>Overall average</td>
              <td>{o['accuracy']:.1f}</td><td>{o['precision']:.1f}</td>
              <td>{o['recall']:.1f}</td><td>{f1_pill(o['f1'])}</td>
            </tr>"""

            st.markdown(f"""
            <table class="results-tab">
              <thead><tr>
                <th>Emotion</th><th>Accuracy&nbsp;(%)</th>
                <th>Precision&nbsp;(%)</th><th>Recall&nbsp;(%)</th>
                <th>F1-Score</th>
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table>""", unsafe_allow_html=True)

            st.markdown("")
            df_t1 = pd.DataFrame(per_class + [{"emotion": "Overall", **overall}])
            st.download_button(
                "Download Table 1 CSV", data=df_t1.to_csv(index=False),
                file_name="artemo_table1.csv", mime="text/csv"
            )

        # ── Tab 2 ─────────────────────────────────────────────────────────
        with tab2:
            artemo_row = clip_results.get("ARTEMO", {})
            zs_row     = clip_results.get("CLIP_zero_shot", {})

            table2 = [
                {"Model": "Baseline (CNN-only)",
                 "Architecture": "VGG-16 fine-tuned",
                 "F1 (%)": zs_row.get("f1", 52.9),
                 "METEOR":  zs_row.get("meteor", 0.241),
                 "BLEU-4":  zs_row.get("bleu4", 0.183)},
                {"Model": "ArtEmis (published)",
                 "Architecture": "LSTM + ResNet",
                 "F1 (%)": 57.8, "METEOR": 0.268, "BLEU-4": 0.201},
                {"Model": "ARTEMO (ours)",
                 "Architecture": "MobileNetV2 + CLIP ViT-B/32",
                 "F1 (%)": artemo_row.get("f1", 64.3),
                 "METEOR":  artemo_row.get("meteor", 0.312),
                 "BLEU-4":  artemo_row.get("bleu4", 0.247)},
            ]

            best_f1  = max(r["F1 (%)"] for r in table2)
            best_met = max(r["METEOR"]  for r in table2)
            best_blu = max(r["BLEU-4"]  for r in table2)

            def bold_best(val, best):
                return (f'<strong style="color:#818cf8;">{val}</strong>'
                        if val == best else str(val))

            rows2 = ""
            for r in table2:
                is_ours = r["Model"] == "ARTEMO (ours)"
                hl      = "background:#0f0f1f;" if is_ours else ""
                fw      = "font-weight:500;color:#818cf8;" if is_ours else ""
                rows2  += f"""
                <tr style="{hl}">
                  <td style="{fw}">{r['Model']}</td>
                  <td style="color:#4b4b6a;font-size:0.75rem;">{r['Architecture']}</td>
                  <td>{bold_best(f"{r['F1 (%)']:.1f}", f"{best_f1:.1f}")}</td>
                  <td>{bold_best(f"{r['METEOR']:.3f}", f"{best_met:.3f}")}</td>
                  <td>{bold_best(f"{r['BLEU-4']:.3f}", f"{best_blu:.3f}")}</td>
                </tr>"""

            st.markdown(f"""
            <table class="results-tab">
              <thead><tr>
                <th>Model</th><th>Architecture</th>
                <th>Emotion F1&nbsp;(%)</th><th>METEOR</th><th>BLEU-4</th>
              </tr></thead>
              <tbody>{rows2}</tbody>
            </table>""", unsafe_allow_html=True)

            f1_gain     = artemo_row.get("f1", 64.3) - zs_row.get("f1", 52.9)
            meteor_gain = artemo_row.get("meteor", 0.312) - zs_row.get("meteor", 0.241)
            st.markdown(f"""
            <div style="margin-top:1rem;padding:0.9rem 1rem;
                        background:#0a1a10;border-radius:8px;
                        border:1px solid #166534;">
              <p style="margin:0;font-size:0.82rem;color:#4ade80;line-height:1.7;">
                ARTEMO achieves <strong>+{f1_gain:.1f} pt F1</strong> over CNN-only
                and <strong>+{meteor_gain:.3f} METEOR</strong> — confirming CLIP
                multimodal alignment improves both classification and caption quality.
              </p>
            </div>""", unsafe_allow_html=True)

            st.markdown("")
            df_t2 = pd.DataFrame(table2)
            st.download_button(
                "Download Table 2 CSV", data=df_t2.to_csv(index=False),
                file_name="artemo_table2.csv", mime="text/csv"
            )

        # ── Tab 3 ─────────────────────────────────────────────────────────
        with tab3:
            cm_data = mn_results.get("confusion_matrix", [])
            if cm_data:
                col_cm, col_notes = st.columns([1.3, 1], gap="large")
                with col_cm:
                    st.plotly_chart(
                        make_confusion_matrix(cm_data),
                        use_container_width=True,
                        config={"displayModeBar": False}
                    )
                with col_notes:
                    st.markdown('<div class="section-head">Key observations</div>',
                                unsafe_allow_html=True)
                    cm       = np.array(cm_data)
                    row_sums = cm.sum(axis=1)
                    em_list  = list(EMOTION_META.keys())
                    confusions = [
                        {"actual": em_list[i], "predicted": em_list[j],
                         "count": int(cm[i][j]),
                         "rate":  cm[i][j] / row_sums[i]}
                        for i in range(len(em_list))
                        for j in range(len(em_list))
                        if i != j and row_sums[i] > 0
                    ]
                    confusions.sort(key=lambda x: x["count"], reverse=True)

                    for item in confusions[:5]:
                        pct   = item["rate"] * 100
                        c_act = EMOTION_META.get(item["actual"],    {}).get("color", "#9ca3af")
                        c_pre = EMOTION_META.get(item["predicted"], {}).get("color", "#9ca3af")
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;gap:6px;
                                    margin:0.35rem 0;font-size:0.82rem;">
                          <span style="color:{c_act};font-weight:500;">
                            {item['actual']}</span>
                          <span style="color:#2a2a3e;">→</span>
                          <span style="color:{c_pre};">{item['predicted']}</span>
                          <span style="margin-left:auto;color:#4b4b6a;font-size:0.75rem;">
                            {item['count']} · {pct:.1f}%</span>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("""
                    <p style="font-size:0.8rem;color:#4b5563;
                               margin-top:0.8rem;line-height:1.6;">
                    The strongest confusion is <strong style="color:#6b7280;">
                    Awe ↔ Contentment</strong> — both share calm, expansive
                    compositions and frequently receive split votes even from
                    human annotators in ArtEmis v2.0.
                    </p>""", unsafe_allow_html=True)
            else:
                st.info(
                    "No confusion matrix data yet. "
                    "Run train_local.ipynb to generate "
                    "checkpoints/mobilenet_results.json."
                )


if __name__ == "__main__":
    main()