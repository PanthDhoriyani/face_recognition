import streamlit as st
from PIL import Image
import pickle
import torch
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import os

### Title
st.title("Celebrity Face Recognition")
st.write("Upload an image or select demo images to identify the celebrity.")

### Demo images setup
DEMO_FOLDER = "demo_images"  # Folder containing demo images
demo_files = os.listdir(DEMO_FOLDER) if os.path.exists(DEMO_FOLDER) else []

### Multi-option: Upload or demo
option = st.radio("Select Input Type:", ["Upload Image", "Demo Images"])

uploaded_file = None
selected_demo_files = []

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
elif option == "Demo Images":
    if demo_files:
        selected_demo_files = st.multiselect(
            "Select demo images:", demo_files
        )
        if selected_demo_files:
            uploaded_file = selected_demo_files
    else:
        st.warning("No demo images found!")

### Model loader
@st.cache_resource
def load_model():
    names = pickle.load(open("names.pkl", "rb"))
    features_list = pickle.load(open("embedding.pkl", "rb"))
    image_paths = pickle.load(open("images.pkl", "rb"))

    model = InceptionResnetV1(pretrained="vggface2", classify=False)
    model.eval()

    mtcnn = MTCNN(image_size=160, keep_all=False)

    return names, features_list, image_paths, model, mtcnn

names, features_list, image_paths, model, mtcnn = load_model()

### Function to process and predict a single image
def process_image(image):
    face = mtcnn(image)

    if face is None:
        st.warning("Face not detected! Using full image.")
        img = np.array(image)
        img_resized = cv2.resize(img, (160, 160))
        face = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0

    emb = model(face.unsqueeze(0)).detach().numpy()

    results = [
        cosine_similarity(emb.reshape(1, -1), f.reshape(1, -1))[0][0]
        for f in features_list
    ]
    index = int(np.argmax(results))
    confidence = round(results[index] * 100, 2)

    matched_data = image_paths[index]
    matched_img = None

    if isinstance(matched_data, str):
        img = cv2.imread(matched_data)
        if img is not None:
            matched_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(matched_data, np.ndarray):
        matched_img = matched_data
    elif isinstance(matched_data, bytes):
        file_bytes = np.asarray(bytearray(matched_data), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            matched_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(matched_data, Image.Image):
        matched_img = matched_data

    return index, confidence, matched_img

### Submit button
if st.button("Submit"):
    if uploaded_file is None:
        st.warning("Please upload an image or select demo images!")
    else:
        # Show a loading spinner while processing
        with st.spinner("Processing... Please wait!"):
            # Multiple demo images
            if isinstance(uploaded_file, list):
                for demo_img_name in uploaded_file:
                    image_path = os.path.join(DEMO_FOLDER, demo_img_name)
                    image = Image.open(image_path).convert("RGB")

                    index, confidence, matched_img = process_image(image)

                    st.markdown(f"## 🎯 {names[index]} - {demo_img_name}")
                    st.markdown(f"### Confidence: {confidence}%")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Input Image", use_container_width=True)
                    with col2:
                        if matched_img is not None:
                            st.image(matched_img, caption="Matched Image", use_container_width=True)
                        else:
                            st.error("Could not display matched image.")
            else:  # Single uploaded image
                image = Image.open(uploaded_file).convert("RGB")
                index, confidence, matched_img = process_image(image)

                st.markdown(f"# 🎯 {names[index]}")
                st.markdown(f"### Confidence: {confidence}%")

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Input Image", use_container_width=True)
                with col2:
                    if matched_img is not None:
                        st.image(matched_img, caption="Matched Image", use_container_width=True)
                    else:
                        st.error("Could not display matched image.")