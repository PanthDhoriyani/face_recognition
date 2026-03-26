#  Celebrity Face Recognition App


A web-based application that identifies celebrities from uploaded images using deep learning and computer vision.

✨ This project demonstrates practical implementation of face recognition using deep learning.

### DESCRIPTION OF FILE AT LAST

In this Project , I have created api using fast-api for this project , but it is not integrated with 
streamlit server. But api is working independently on local network... and can be checked on /docs endpoint
deployed->url->https://facerecognition-qufsgp5uevydmyraum3sah.streamlit.app/

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
├── testing.py
├── training.py
├── embedding.pkl
├── requirements.txt
└── data
```
##  About files

* Data -> unzip data file
* data.zip -> zip data file
* screenshot -> to generate demo output
* app.py -> streamlit file
* embedding.pkl -> embedding of image(Pickle)
* main.py(ongoing work) -> API(fastapi)
* training -> training model
* testing -> testing model
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




