import streamlit as st
import pickle
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Celebrity Face Recognition",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 Celebrity Face Recognition")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_resource
def load_data():
    names = pickle.load(open('names.pkl', 'rb'))
    embeddings = pickle.load(open('embedding.pkl', 'rb'))
    images = pickle.load(open('images.pkl', 'rb'))
    return names, embeddings, images

names, features_list, image_paths = load_data()

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    mtcnn = MTCNN(image_size=160)
    model = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, model

mtcnn, model = load_model()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Settings")

threshold = st.sidebar.slider(
    "Recognition Threshold",
    0.3, 1.0, 0.6, 0.05
)

mode = st.sidebar.radio(
    "Choose Mode",
    ["Upload Image", "Demo Images"]
)

# -----------------------------
# Prediction Function
# -----------------------------
def predict(img_np):
    face = mtcnn(img_np)

    if face is None:
        img_resized = cv2.resize(img_np, (160, 160))
        face = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0

    emb = model(face.unsqueeze(0))
    emb_np = emb.detach().numpy()

    scores = []

    for f in features_list:
        similarity = cosine_similarity(
            emb_np.reshape(1, -1),
            f.reshape(1, -1)
        )[0][0]

        scores.append(similarity)

    best_idx = np.argmax(scores)
    best_score = scores[best_idx]

    return best_idx, best_score

# -----------------------------
# Upload Mode
# -----------------------------
if mode == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)

        if st.button("🚀 Predict"):
            with st.spinner("Processing... Please wait ⏳"):
                idx, score = predict(img_np)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📤 Input Image")
                    st.image(image, use_container_width=True)

                with col2:
                    st.subheader("🎯 Prediction")

                    matched_name = names[idx]
                    matched_img = image_paths[idx][:, :, ::-1]

                    if score < threshold:
                        st.warning(f"⚠️ Low Confidence Prediction: {matched_name}")
                        st.write(f"Confidence: {score:.2f} (Below Threshold)")
                    else:
                        st.success(f"✅ {matched_name}")
                        st.write(f"Confidence: {score:.2f}")

                    st.image(matched_img, use_container_width=True)
                    st.progress(float(score))

# -----------------------------
# Demo Mode
# -----------------------------
elif mode == "Demo Images":
    st.subheader("🎯 Select Demo Images")

    demo_folder = "demo_images"

    if not os.path.exists(demo_folder):
        st.error("❌ demo_images folder not found")
    else:
        demo_files = [f for f in os.listdir(demo_folder) if f.lower().endswith(('jpg','png','jpeg'))]

        selected_files = st.multiselect(
            "Choose Demo Images",
            demo_files
        )

        if selected_files:
            if st.button("🚀 Run Demo"):
                with st.spinner("Processing Demo Images... ⏳"):

                    for file in selected_files:
                        img_path = os.path.join(demo_folder, file)
                        image = Image.open(img_path).convert('RGB')
                        img_np = np.array(image)

                        idx, score = predict(img_np)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader(f"📤 Input: {file}")
                            st.image(image, use_container_width=True)

                        with col2:
                            st.subheader("🎯 Prediction")

                            matched_name = names[idx]
                            matched_img = image_paths[idx][:, :, ::-1]

                            if score < threshold:
                                with st.container():
                                    st.markdown(
                                        f"""
                                        <div style="
                                            background-color:#fff3cd;
                                            padding:15px;
                                            border-radius:10px;
                                            border:1px solid #ffeeba;
                                            color:black;
                                        ">
                                            <h4>⚠️ Low Confidence Prediction</h4>
                                            <p><b>Name:</b> {matched_name}</p>
                                            <p><b>Confidence:</b> {score:.2f} (Below Threshold)</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.success(f"✅ {matched_name}")
                                st.write(f"Confidence: {score:.2f}")

                            st.image(matched_img, use_container_width=True)
                            st.progress(float(score))

                        st.markdown("---")