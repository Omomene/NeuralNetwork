import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.express as px

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

    return cnn, mobilenet


cnn_model, mobilenet_model = load_models()

# ==================================================
# STYLE
# ==================================================

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

.header{
    background: linear-gradient(
        90deg,
        #1F3A5F,
        #2E5EAA
    );

    color:white;
    padding:20px;
    border-radius:12px;
    text-align:center;
    margin-bottom:20px;
}

.chart-card{
    background:white;
    border-radius:10px;
    padding:10px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="header">
<h1>IKEA Furniture Classification</h1>
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

    st.subheader("Furniture Classification")

    uploaded_file = st.file_uploader(
        "Upload Furniture Image",
        type=["jpg", "jpeg", "png"]
    )

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

        img_array = img_array.astype(
            "float32"
        ) / 255.0

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

        pred_cnn = CLASS_NAMES[
            np.argmax(cnn_pred)
        ]

        pred_mobile = CLASS_NAMES[
            np.argmax(mobile_pred)
        ]

        conf_cnn = np.max(cnn_pred)

        conf_mobile = np.max(mobile_pred)

        col1, col2, col3 = st.columns(3)

        with col1:

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.markdown("### Custom CNN")

            st.write(
                f"Prediction: **{pred_cnn}**"
            )

            st.write(
                f"Confidence: **{conf_cnn:.2%}**"
            )

        with col3:

            st.markdown("### MobileNetV2")

            st.write(
                f"Prediction: **{pred_mobile}**"
            )

            st.write(
                f"Confidence: **{conf_mobile:.2%}**"
            )
            
            
        col1, col2 = st.columns(2)
        with col1:
            prob_df = pd.DataFrame({
                "Category": CLASS_NAMES,
                "Probability": cnn_pred[0]
            })

            fig = px.bar(
                prob_df,
                x="Category",
                y="Probability",
                title="Custom CNN Probabilities"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:
            prob_df = pd.DataFrame({
                "Category": CLASS_NAMES,
                "Probability": mobile_pred[0]
            })

            fig = px.bar(
                prob_df,
                x="Category",
                y="Probability",
                title="MobileNetV2 Probabilities"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
        



# ==================================================
# TAB 2 : DATASET
# ==================================================

with tab2:

    st.subheader("Dataset Exploration")

    col1, col2 = st.columns(2)

    with col1:

        counts = (
            df["category"]
            .value_counts()
            .reset_index()
        )

        counts.columns = [
            "Category",
            "Count"
        ]

        fig = px.bar(
            counts,
            x="Category",
            y="Count",
            title="Category Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            df,
            x="price",
            nbins=30,
            title="Price Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    with col1:

        avg_price = (
            df.groupby("category")["price"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            avg_price,
            x="category",
            y="price",
            title="Average Price by Category"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        availability = (
            df["sellable_online"]
            .value_counts()
            .reset_index()
        )

        availability.columns = [
            "Online",
            "Count"
        ]

        fig = px.pie(
            availability,
            names="Online",
            values="Count",
            title="Online Availability"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

# ==================================================
# TAB 3 : MODELS
# ==================================================

with tab3:

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Custom CNN",
            "MobileNetV2"
        ],

        "Accuracy": [
            0.59,
            0.42,
            0.62
        ]
    })

    fig = px.bar(
        comparison,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        title="Model Accuracy Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    comparison_table = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Validation Accuracy",
            "Model Type"
        ],

        "Custom CNN": [
            "0.42",
            "0.42",
            "CNN From Scratch"
        ],

        "MobileNetV2": [
            "0.62",
            "0.62",
            "Transfer Learning"
        ]
    })

    st.dataframe(
        comparison_table,
        use_container_width=True
    )

