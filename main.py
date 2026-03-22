from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import pickle, io, cv2, torch, numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity

# --- FASTAPI INSTANCE ---
app = FastAPI(title="Celebrity Face Recognition API")

# --- LOAD MODEL AND DATA ON STARTUP ---
names = pickle.load(open('names.pkl', 'rb'))
features_list = pickle.load(open('embedding.pkl', 'rb'))
image_paths = pickle.load(open('images.pkl', 'rb'))

model = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes=None)
model.eval()
mtcnn = MTCNN(image_size=160)

# --- PREDICTION ENDPOINT ---
@app.post("/predict_image")
def predict_image(file: UploadFile = File(...)):

    # read file
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # detect face
    face = mtcnn(img)
    if face is None:
        img_resized = cv2.resize(img, (160,160))
        face = torch.tensor(img_resized).permute(2,0,1).float()/255.0

    # embedding
    emb = model(face.unsqueeze(0)).detach().numpy()
    result = [cosine_similarity(emb.reshape(1,-1), f.reshape(1,-1))[0][0] for f in features_list]
    index = int(np.argmax(result))

    # load matched image
    matched_img = cv2.imread(image_paths[index])
    matched_img = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', matched_img)
    io_buf = io.BytesIO(buffer)

    return StreamingResponse(io_buf, media_type="image/jpeg")