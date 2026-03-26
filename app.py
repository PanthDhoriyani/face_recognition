import streamlit as st
import requests
from PIL import Image
import base64
import io
import os

# ---------------- CONFIG ---------------- #

API_URL = "http://127.0.0.1:8000/predict_image"

st.set_page_config(
    page_title="Celebrity Face Recognition",
    page_icon="🎭",
    layout="wide"
)

st.markdown("####  @Panth-D")
st.title("🎭 Celebrity Face Recognition (API Powered)")
st.write("Upload or select demo images to detect celebrity using FastAPI backend.")

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("⚙️ Settings")

threshold = st.sidebar.slider(
    "Recognition Threshold",
    0.3, 1.0, 0.6, 0.05
)

mode = st.sidebar.radio(
    "Choose Mode",
    ["Upload Image", "Demo Images"]
)

# ---------------- API CALL FUNCTION ---------------- #

def call_api_uploaded(uploaded_file):
    try:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        response = requests.post(API_URL, files=files)

        if response.status_code != 200:
            st.error(f"❌ API Error: {response.text}")
            return None

        return response.json()

    except Exception as e:
        st.error(f"🚨 Connection Error: {e}")
        return None


def call_api_demo(file_path):
    try:
        with open(file_path, "rb") as f:
            files = {
                "file": (os.path.basename(file_path), f.read(), "image/jpeg")
            }

            response = requests.post(API_URL, files=files)

        if response.status_code != 200:
            st.error(f"❌ API Error: {response.text}")
            return None

        return response.json()

    except Exception as e:
        st.error(f"🚨 Connection Error: {e}")
        return None


# ---------------- UPLOAD MODE ---------------- #

if mode == "Upload Image":

    uploaded_file = st.file_uploader(
        "📤 Upload Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')

        if st.button("🚀 Predict"):
            with st.spinner("Calling API... ⏳"):

                result = call_api_uploaded(uploaded_file)

                if result:
                    name = result["name"]
                    confidence = result["confidence"]

                    img_bytes = base64.b64decode(result["image"])
                    matched_img = Image.open(io.BytesIO(img_bytes))

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📤 Input Image")
                        st.image(image, use_container_width=True)

                    with col2:
                        st.subheader("🎯 Prediction")

                        if confidence < threshold:
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
                                    <p><b>Name:</b> {name}</p>
                                    <p><b>Confidence:</b> {confidence:.2f}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.success(f"✅ {name}")
                            st.write(f"Confidence: {confidence:.2f}")

                        st.image(matched_img, use_container_width=True)
                        st.progress(float(confidence))


# ---------------- DEMO MODE ---------------- #

elif mode == "Demo Images":

    st.subheader("🎯 Select Demo Images")

    demo_folder = "demo_images"

    if not os.path.exists(demo_folder):
        st.error("❌ demo_images folder not found")

    else:
        demo_files = [
            f for f in os.listdir(demo_folder)
            if f.lower().endswith(('jpg','png','jpeg'))
        ]

        selected_files = st.multiselect(
            "Choose Demo Images",
            demo_files
        )

        if selected_files:
            if st.button("🚀 Run Demo"):
                with st.spinner("Processing Demo Images... ⏳"):

                    for file in selected_files:

                        img_path = os.path.join(demo_folder, file)

                        result = call_api_demo(img_path)

                        if result:
                            name = result["name"]
                            confidence = result["confidence"]

                            img_bytes = base64.b64decode(result["image"])
                            matched_img = Image.open(io.BytesIO(img_bytes))

                            col1, col2 = st.columns(2)

                            with col1:
                                st.subheader(f"📤 Input: {file}")
                                st.image(img_path, use_container_width=True)

                            with col2:
                                st.subheader("🎯 Prediction")

                                if confidence < threshold:
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
                                            <p><b>Name:</b> {name}</p>
                                            <p><b>Confidence:</b> {confidence:.2f}</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.success(f"✅ {name}")
                                    st.write(f"Confidence: {confidence:.2f}")

                                st.image(matched_img, use_container_width=True)
                                st.progress(float(confidence))

                            st.markdown("---")