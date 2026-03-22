import os
import zipfile
import cv2
import pickle
from facenet_pytorch import InceptionResnetV1,MTCNN
import torch

#extract zip file
with zipfile.ZipFile('data.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

##creating images folder
dataset = []
for i in os.listdir('data'):
    for j in os.listdir(os.path.join('data', i)):
        for k in os.listdir(os.path.join('data', i, j)):
            for file in os.listdir(os.path.join('data', i, j, k)):
                dataset.append(cv2.imread(os.path.join('data', i, j, k, file)))


##creating name folder

names=[]
for i in os.listdir('data'):
  for j in os.listdir(os.path.join('data',i)):
    for k in os.listdir(os.path.join('data',i,j)):
      for p in os.listdir(os.path.join('data',i,j,k)):
        names.append(p.split('.')[0])


##json file
pickle.dump(names,open('names.pkl','wb'))
pickle.dump(dataset,open('images.pkl','wb'))


##model
model=InceptionResnetV1(
    pretrained='vggface2',
    classify=False,
    num_classes=None
)
model.eval()

mtcnn = MTCNN(image_size=160)
model = InceptionResnetV1(pretrained='vggface2').eval()

##embedding of all images
embeddings = []
i = 0

for img in dataset:

    face = mtcnn(img)


    if face is None:

        img_resized = cv2.resize(img, (160, 160))
        img_tensor = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0
        face = img_tensor

    emb = model(face.unsqueeze(0))
    embeddings.append(emb.detach().numpy())

print("Total embeddings:", len(embeddings))

##craeting json file of embedding
pickle.dump(embeddings,open('embedding.pkl','wb'))