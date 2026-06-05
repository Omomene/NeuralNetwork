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
    padding:18px;
    border-radius:12px;
    text-align:center;
    margin-bottom:20px;
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
    padding: 12px;
}

/* SMALLER PLOTLY GRAPHS */
.js-plotly-plot .plotly {
    height: 280px !important;
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

    uploaded_file = st.file_uploader(
        "Upload Furniture Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        img = image.resize((224, 224))

        img_array = np.array(img)

        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        img_array = img_array.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        cnn_pred = cnn_model.predict(img_array, verbose=0)
        mobile_pred = mobilenet_model.predict(img_array, verbose=0)

        yolo_results = yolo_model.predict(image, verbose=False)
        annotated_image = yolo_results[0].plot()

        detected_objects = []

        for box in yolo_results[0].boxes:
            cls = int(box.cls)
            detected_objects.append(yolo_results[0].names[cls])

        pred_cnn = CLASS_NAMES[np.argmax(cnn_pred)]
        pred_mobile = CLASS_NAMES[np.argmax(mobile_pred)]

        conf_cnn = np.max(cnn_pred)
        conf_mobile = np.max(mobile_pred)

        # BETTER SPACING (4 columns)
        col1, col2, col3, col4 = st.columns([1.7, 1.5, 1.5, 1.4])

        with col1:
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("### Custom CNN")

            st.write(f"Prediction: **{pred_cnn}**")
            st.write(f"Confidence: **{conf_cnn:.2%}**")

            prob_df = pd.DataFrame({
                "Category": CLASS_NAMES,
                "Probability": cnn_pred[0]
            })

            fig = px.bar(
                prob_df,
                x="Category",
                y="Probability",
            )

            fig.update_traces(marker_line_width=0.5)

            fig.update_layout(height=280)

            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.markdown("### MobileNetV2")

            st.write(f"Prediction: **{pred_mobile}**")
            st.write(f"Confidence: **{conf_mobile:.2%}**")

            prob_df = pd.DataFrame({
                "Category": CLASS_NAMES,
                "Probability": mobile_pred[0]
            })

            fig = px.bar(
                prob_df,
                x="Category",
                y="Probability",
            )

            fig.update_traces(marker_line_width=0.5)
            fig.update_layout(height=280)

            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown("### YOLOv8 Detection")

            if detected_objects:
                st.write(f"Detected: {', '.join(sorted(set(detected_objects)))}")
            else:
                st.write("No objects detected")
            
            st.image(annotated_image, use_container_width=True)

# ==================================================
# TAB 2 : DATASET
# ==================================================

with tab2:

    st.subheader("Dataset Exploration")

    col1, col2 = st.columns(2)

    with col1:

        counts = df["category"].value_counts().reset_index()
        counts.columns = ["Category", "Count"]

        fig = px.bar(
            counts,
            x="Category",
            y="Count",
            title="Category Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.histogram(
            df,
            x="price",
            nbins=30,
            title="Price Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        avg_price = df.groupby("category")["price"].mean().reset_index()

        fig = px.bar(
            avg_price,
            x="category",
            y="price",
            title="Average Price by Category"
        )

        fig.update_layout(height=260)
        st.plotly_chart(fig, use_container_width=True)

    with col2:

        availability = df["sellable_online"].value_counts().reset_index()
        availability.columns = ["Online", "Count"]

        fig = px.pie(
            availability,
            names="Online",
            values="Count",
            title="Online Availability"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df.head(20), use_container_width=True)

# ==================================================
# TAB 3 : MODELS
# ==================================================

with tab3:

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "Custom CNN", "MobileNetV2"],
        "Accuracy": [0.59, 0.42, 0.62]
    })

    fig = px.bar(
        comparison,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        title="Model Accuracy Comparison"
    )

    fig.update_traces(marker_line_width=0.5)
    fig.update_layout(height=280)

    st.plotly_chart(fig, use_container_width=True)

    comparison_table = pd.DataFrame({
        "Metric": ["Accuracy", "Validation Accuracy", "Model Type"],
        "Custom CNN": ["0.42", "0.42", "CNN From Scratch"],
        "MobileNetV2": ["0.62", "0.62", "Transfer Learning"]
    })

    st.dataframe(comparison_table, use_container_width=True)