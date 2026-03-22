#  Celebrity Face Recognition App

A web-based application that identifies celebrities from uploaded images using deep learning and computer vision.

✨ This project demonstrates practical implementation of face recognition using deep learning.


##  Features

* Upload an image and detect faces automatically
* Generate facial embeddings using FaceNet (InceptionResnetV1)
* Match faces with a pre-stored celebrity dataset
* Display the closest match with confidence score
* Clean and interactive UI using Streamlit

##  Tech Stack

* Python
* Streamlit
* OpenCV
* Facenet-PyTorch (MTCNN + InceptionResnetV1)
* Scikit-learn

##  How It Works

1. User uploads an image
2. MTCNN detects the face
3. FaceNet generates a 512-d embedding
4. Cosine similarity compares it with stored embeddings
5. Best match is displayed with confidence score

##  Project Structure

```
├── app.py
├── names.pkl
├── embedding.pkl
├── images.pkl
├── requirements.txt
└── README.md
```

##  Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

##  Output

Displays:

* Uploaded image
* Matched celebrity image
* Celebrity name (highlighted)
* Confidence score




