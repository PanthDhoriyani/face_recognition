from fastapi import FastAPI, UploadFile, File, HTTPException
import pickle, cv2, torch, numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import base64
import os

app = FastAPI(title="Celebrity Face Recognition API")

# ---------------- LOAD DATA ---------------- #

try:
    names = pickle.load(open('names.pkl', 'rb'))
    features_list = pickle.load(open('embedding.pkl', 'rb'))
except Exception as e:
    raise RuntimeError(f"Error loading pickle files: {e}")

# build image paths
image_folder = "hf_dataset/images"
image_paths = []

for i, name in enumerate(names):
    filename = f"{name}_{i}.jpg"
    path = os.path.join(image_folder, filename)
    image_paths.append(path)

# ---------------- LOAD MODEL ---------------- #

model = InceptionResnetV1(pretrained='vggface2', classify=False)
model.eval()

mtcnn = MTCNN(image_size=160)

# ---------------- HEALTH CHECK ---------------- #

@app.get("/")
def home():
    return {"status": "API running"}

# ---------------- PREDICTION ---------------- #

@app.post("/predict_image")
def predict_image(file: UploadFile = File(...)):

    # read image
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # detect face
    face = mtcnn(img)

    if face is None:
        img_resized = cv2.resize(img, (160,160))
        face = torch.tensor(img_resized).permute(2,0,1).float()/255.0

    # embedding
    emb = model(face.unsqueeze(0)).detach().numpy()

    # similarity
    scores = [
        cosine_similarity(emb.reshape(1,-1), f.reshape(1,-1))[0][0]
        for f in features_list
    ]

    index = int(np.argmax(scores))
    confidence = float(scores[index])
    name = names[index]

    # load matched image safely
    matched_img = cv2.imread(image_paths[index])

    if matched_img is None:
        raise HTTPException(status_code=500, detail="Matched image not found")

    _, buffer = cv2.imencode('.jpg', matched_img)
    img_base64 = base64.b64encode(buffer).decode()

    return {
        "name": name,
        "confidence": confidence,
        "image": img_base64
    }