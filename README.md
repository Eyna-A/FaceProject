# Face Recognition Project

A simple face recognition system using **MTCNN** and **Keras FaceNet**.  
This project detects faces from your webcam and compares them to a database of images.

---

## Features
- Detect faces using MTCNN.
- Generate embeddings with FaceNet.
- Compare live webcam feed with a database of face images.
- Display "Access Granted" or "Access Denied" based on similarity.

---

## Requirements
- Python 3.8+
- OpenCV
- MTCNN
- keras-facenet
- NumPy
- SciPy

Install dependencies using pip:

```bash
pip install opencv-python mtcnn keras-facenet numpy scipy
