import pickle
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


####imported json file created in training
# load data
names = pickle.load(open('names.pkl', 'rb'))   # actor names
features_list = pickle.load(open('embedding.pkl', 'rb'))
image_paths = pickle.load(open('images.pkl', 'rb'))  # 🔥 ADD THIS

# model
model=InceptionResnetV1(
    pretrained='vggface2',
    classify=False,
    num_classes=None
)
model.eval()
mtcnn = MTCNN(image_size=160)

##testing
x = cv2.imread('demo images/Screenshot 2026-03-22 203446.png')
face = mtcnn(x)

if face is None:
    img_resized = cv2.resize(x, (160, 160))
    img_tensor = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0
    face = img_tensor

emb = model(face.unsqueeze(0))


####cosine similarity
emb_np = emb.detach().numpy()

result = []

for f in features_list:
    x = cosine_similarity(
        emb_np.reshape(1, -1),
        f.reshape(1, -1)
    )[0][0]

    result.append(x)


###image generation
index_pos=sorted(list(enumerate(result)),reverse=True,key=lambda x:x[1])[0][0]


img = image_paths[index_pos]
name = names[index_pos]

img = img[:, :, ::-1]

plt.imshow(img)
plt.title(name)
plt.axis('off')
plt.show()