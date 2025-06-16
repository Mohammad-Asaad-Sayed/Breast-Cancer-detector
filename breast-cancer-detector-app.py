import streamlit as st
from PIL import Image
from classifier import ImageClassifier
import pandas as pd


# Initialize classifier (cached)
classifier= ImageClassifier()


# App UI
st.set_page_config(page_title="Medical Image Classifier", layout="wide")
st.title("Medical Image Classifier")

# File upload
uploaded_file = st.file_uploader(
    "Upload medical image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to classify as Normal or Malignant"
)

# Results section
if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        # Classify image
    
        class_name, confidence, all_probs = classifier.predict(image)

        # Show results
        st.subheader("Classification Results")
        st.metric("Prediction", f"{class_name} ({confidence * 100:.1f}%)")

        # Probabilities chart
        prob_data = pd.DataFrame({
            "Class": [name.strip() for name in classifier.class_names],
            "Probability": all_probs
        }).sort_values("Probability", ascending=False)

        st.bar_chart(prob_data.set_index("Class"))
