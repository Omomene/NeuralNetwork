import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.express as px
from ultralytics import YOLO

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="IKEA Furniture AI",
    layout="wide"
)

# ==================================================
# DATA
# ==================================================

DATA_PATH = "dataset/IKEA_SA_Furniture_Web_Scrapings_sss.csv"

CLASS_NAMES = [
    "Beds",
    "Cabinets & cupboards",
    "Sofas & armchairs",
    "Tables & desks"
]

try:
    df = pd.read_csv(DATA_PATH)

    df = df[
        df["category"].isin(CLASS_NAMES)
    ].copy()

except:
    df = pd.DataFrame()

# ==================================================
# MODELS
# ==================================================

@st.cache_resource
def load_models():

    cnn = tf.keras.models.load_model(
        "models/furniture_cnn.keras"
    )

    mobilenet = tf.keras.models.load_model(
        "models/mobilenet_model.keras"
    )

    yolo = YOLO(
        "runs/detect/runs/furniture_detector-2/weights/best.pt"
    )

    return cnn, mobilenet, yolo


cnn_model, mobilenet_model, yolo_model = load_models()

def style_ikea_chart(fig):

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family="Arial",
            size=15,
            color=IKEA_DARK_BLUE
        ),
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20
        ),
        xaxis=dict(
            showgrid=False,
            title=None
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#EAEAEA",
            zeroline=False
        )
    )

    return fig
# ==================================================
# STYLE (IKEA INSPIRED)
# ==================================================

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

/* HEADER */
.header{
    background: linear-gradient(90deg, #1F3A5F, #2E5EAA);
    color:white;
    padding:16px;
    border-radius:12px;
    text-align:center;
    margin-bottom:10px;
    position: relative;
}

/* IKEA BADGE */
.ikea-logo{
    position:absolute;
    left:20px;
    top:36px;
    background:#FFCC00;
    color:#003399;
    font-weight:bold;
    padding:6px 12px;
    border-radius:6px;
    font-size:26px;
}

/* IKEA ACCENTS */
h1, h2, h3 {
    color:#003399;
}

a {
    color:#FFCC00; !important;
    text-decoration: underline;
}

a:hover {
    color:#FFCC00 !important;
}

/* CARDS */
.chart-card{
    background:white;
    border-radius:10px;
    padding:12px;
}

/* MORE SPACE BETWEEN COLUMNS */
div[data-testid="column"] {
    padding: 14px;
}

/* SMALLER PLOTLY GRAPHS */
.js-plotly-plot .plotly {
    height: 280px !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 22px !important;
    font-weight: bold !important;
}

/* Upload Furniture Image */
[data-testid="stFileUploader"] label {
    font-size: 22px !important;
    font-weight: bold !important;
}

/* Custom CNN / MobileNetV2 / YOLO titles */
h3 {
    font-size: 24px !important;
}

/* Prediction and confidence text */
p {
    font-size: 18px !important;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="header">
    <div class="ikea-logo">IKEA</div>
    <h1>Furniture Classification and Detection</h1>
</div>
""", unsafe_allow_html=True)

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3 = st.tabs([
    "Prediction",
    "Dataset",
    "Models"
])

# ==================================================
# TAB 1 : PREDICTION
# ==================================================

with tab1:

    st.file_uploader(
        "Upload Furniture Image",
        type=["jpg", "jpeg", "png"],
        key="upload"
    )

    uploaded_file = st.session_state.get("upload")

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        img = image.resize((224, 224))

        img_array = np.array(img)

        if len(img_array.shape) == 2:
            img_array = np.stack(
                [img_array] * 3,
                axis=-1
            )

        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        img_array = img_array.astype("float32") / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        cnn_pred = cnn_model.predict(
            img_array,
            verbose=0
        )

        mobile_pred = mobilenet_model.predict(
            img_array,
            verbose=0
        )

        yolo_results = yolo_model.predict(
            image,
            verbose=False
        )

        annotated_image = yolo_results[0].plot()

        detected_objects = []

        for box in yolo_results[0].boxes:

            cls = int(box.cls)

            detected_objects.append(
                yolo_results[0].names[cls]
            )

        pred_cnn = CLASS_NAMES[np.argmax(cnn_pred)]
        pred_mobile = CLASS_NAMES[np.argmax(mobile_pred)]

        conf_cnn = np.max(cnn_pred)
        conf_mobile = np.max(mobile_pred)

        col1, col2, col3, col4 = st.columns(
            [1.8, 1.4, 1.4, 1.4]
        )

        # ==========================
        # IMAGE
        # ==========================

        with col1:

            st.markdown("### Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

        # ==========================
        # CUSTOM CNN
        # ==========================

        with col2:

            st.markdown("### Custom CNN")

            st.markdown(
                f"""
                <div style="
                    background:#F8F9FB;
                    padding:10px;
                    border-left:5px solid #FFCC00;
                    border-radius:8px;
                    margin-bottom:10px;">
                    <b>Prediction:</b> {pred_cnn}<br>
                    <b>Confidence:</b>
                    <span style="color:#0058A3;">
                    {conf_cnn:.2%}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            prob_df = pd.DataFrame({
                "Category": CLASS_NAMES,
                "Probability": cnn_pred[0]
            })

            fig = px.bar(
                prob_df,
                x="Category",
                y="Probability",
                color="Probability",
                color_continuous_scale=[
                    "#FFCC00",
                    "#0058A3"
                ]
            )

            fig.update_coloraxes(
                showscale=False
            )

            fig.update_layout(
                height=260,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(
                    size=14,
                    color="#003399"
                ),
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                ),
                xaxis_title="",
                yaxis_title=""
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ==========================
        # MOBILENET
        # ==========================

        with col3:

            st.markdown("### MobileNetV2")

            st.markdown(
                f"""
                <div style="
                    background:#F8F9FB;
                    padding:10px;
                    border-left:5px solid #FFCC00;
                    border-radius:8px;
                    margin-bottom:10px;">
                    <b>Prediction:</b> {pred_mobile}<br>
                    <b>Confidence:</b>
                    <span style="color:#0058A3;">
                    {conf_mobile:.2%}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            prob_df = pd.DataFrame({
                "Category": CLASS_NAMES,
                "Probability": mobile_pred[0]
            })

            fig = px.bar(
                prob_df,
                x="Category",
                y="Probability",
                color="Probability",
                color_continuous_scale=[
                    "#FFCC00",
                    "#0058A3"
                ]
            )

            fig.update_coloraxes(
                showscale=False
            )

            fig.update_layout(
                height=260,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(
                    size=14,
                    color="#003399"
                ),
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                ),
                xaxis_title="",
                yaxis_title=""
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ==========================
        # YOLO
        # ==========================

        with col4:

            st.markdown("### YOLOv8 Detection")

            if detected_objects:

                st.markdown(
                    f"""
                    <div style="
                        background:#F8F9FB;
                        padding:10px;
                        border-left:5px solid #0058A3;
                        border-radius:8px;
                        margin-bottom:10px;">
                        <b>Detected:</b>
                        {', '.join(sorted(set(detected_objects)))}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div style="
                        background:#F8F9FB;
                        padding:10px;
                        border-left:5px solid #0058A3;
                        border-radius:8px;
                        margin-bottom:10px;">
                        No objects detected
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.image(
                annotated_image,
                use_container_width=True
            )


# ==================================================
# TAB 2 : DATASET
# ==================================================

with tab2:

    st.subheader("Dataset Overview")

    IKEA_BLUE = "#0058A3"
    IKEA_DARK_BLUE = "#003399"
    IKEA_YELLOW = "#FFCC00"

    # ==================================================
    # 1. SUMMARY TABLE
    # ==================================================

    summary_df = pd.DataFrame({
        "Dataset": [
            "IKEA Classification Dataset",
            "YOLOv8 Detection Dataset"
        ],
        "Task": [
            "Image Classification (CNN + MobileNet)",
            "Object Detection (YOLOv8)"
        ],
        "Images": [
            len(df) if not df.empty else "Unknown",
            8055
        ],
        "Classes": [
            len(CLASS_NAMES),
            25
        ]
    })

    st.dataframe(summary_df, use_container_width=True)

    # ==================================================
    # 2. CHART LAYOUT
    # ==================================================

    col1, col2 = st.columns(2)

    # ==================================================
    # IKEA CLASS DISTRIBUTION
    # ==================================================

    with col1:

        if not df.empty and "category" in df.columns:

            counts = (
                df["category"]
                .value_counts()
                .reindex(CLASS_NAMES)
                .fillna(0)
                .reset_index()
            )

            counts.columns = ["Category", "Count"]

            fig = px.bar(
                counts,
                x="Category",
                y="Count",
                title="IKEA Dataset Class Distribution",
                color="Count",
                color_continuous_scale=[
                    IKEA_YELLOW,
                    IKEA_BLUE
                ]
            )

            fig.update_layout(
                height=320,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(color=IKEA_DARK_BLUE),
                title_font=dict(color=IKEA_DARK_BLUE),
                margin=dict(l=10, r=10, t=40, b=10)
            )

            fig.update_coloraxes(showscale=False)

            st.plotly_chart(fig, use_container_width=True)

    # ==================================================
    # YOLO DATASET SPLIT
    # ==================================================

    with col2:

        split_df = pd.DataFrame({
            "Split": ["Train", "Validation", "Test"],
            "Images": [6424, 891, 739]
        })

        fig = px.pie(
            split_df,
            names="Split",
            values="Images",
            title="YOLOv8 Dataset Split",
            color_discrete_sequence=[
                IKEA_BLUE,
                IKEA_YELLOW,
                IKEA_DARK_BLUE
            ]
        )

        fig.update_layout(
            height=280,
            paper_bgcolor="white",
            font=dict(color=IKEA_DARK_BLUE),
            title_font=dict(color=IKEA_DARK_BLUE),
            margin=dict(l=10, r=10, t=40, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)


# ==================================================
# TAB 3 : MODELS
# ==================================================

with tab3:

    st.subheader("Model Performance Overview")

    IKEA_BLUE = "#0058A3"
    IKEA_DARK_BLUE = "#003399"
    IKEA_YELLOW = "#FFCC00"

    # ==================================================
    # 1. MODEL COMPARISON TABLE (UPGRADED)
    # ==================================================

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Custom CNN",
            "MobileNetV2",
            "YOLOv8"
        ],
        "Category": [
            "Machine Learning",
            "Deep Learning",
            "Transfer Learning",
            "Object Detection"
        ],
        "Evaluation Metric": [
            "Accuracy",
            "Accuracy",
            "Accuracy",
            "mAP50-95"
        ],
        "Score": [
            0.59,
            0.42,
            0.62,
            0.24   
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    # ==================================================
    # 2. SCORE COMPARISON CHART (IKEA STYLE)
    # ==================================================

    fig = px.bar(
        comparison,
        x="Model",
        y="Score",
        text="Score",
        color="Model",
        color_discrete_sequence=[
            IKEA_BLUE,
            IKEA_YELLOW,
            IKEA_DARK_BLUE,
            "#4C9AFF"
        ],
        title="Model Performance Comparison"
    )

    fig.update_layout(
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=IKEA_DARK_BLUE),
        title_font=dict(color=IKEA_DARK_BLUE),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="",
        yaxis_title="Score"
    )

    st.plotly_chart(fig, use_container_width=True)
