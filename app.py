import streamlit as st
from PIL import Image
import pickle
import torch
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity

### Title
st.title("Celebrity Face Recognition")
st.write("Upload an image to identify the celebrity.")


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

### upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")

## prediction model
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

## image same
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

# result
    st.markdown(f"# 🎯 {names[index]}")
    st.markdown(f"### Confidence: {confidence}%")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if matched_img is not None:
            st.image(matched_img, caption="Matched Image", use_container_width=True)
        else:
            st.error("Could not display matched image.")