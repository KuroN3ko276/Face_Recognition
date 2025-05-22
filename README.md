# Face Recognition System

This project implements a face recognition system using webcam input. It uses MTCNN for face detection, a ResNet-based model for face embedding extraction, and Qdrant for vector storage and similarity search.

## Features

- Face detection from webcam using MTCNN
- Face embedding extraction using InceptionResnetV1 (FaceNet)
- Vector database storage and retrieval with Qdrant
- Real-time face recognition with confidence scores

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- mtcnn
- tensorflow
- opencv-python
- numpy
- qdrant-client
- facenet-pytorch
- scikit-learn
- pillow

## Dataset Structure

The system expects a dataset organized as follows:

```
Dataset/
├── Person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Person2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

Each person should have their own directory containing multiple images of their face. The name of the directory will be used as the person's name in the recognition system.

## Usage

1. Organize your dataset as described above in the `Dataset` directory.

2. Register faces to the database:

```bash
# Register all faces from the dataset
python register_faces.py

# Reset the database and register all faces
python register_faces.py --reset

# Register only a specific person
python register_faces.py --person PersonName

# Reset a specific person's data and register them again
python register_faces.py --person PersonName --reset-person
```

3. Run the face recognition script:

```bash
python face_recognition.py
```

4. The system will:
   - Start the webcam for real-time face recognition
   - Display the recognized person's name and confidence score on the video feed

5. Press 'q' to quit the application.

## Testing

You can test the face recognition system without using a webcam:

```bash
# Test recognition on a single image
python test_face_recognition.py --image path/to/image.jpg

# Test recognition on all images in a directory
python test_face_recognition.py --dir path/to/person_directory
```

The test script provides:
- Visual display of detected faces and recognition results
- Detailed match information with confidence scores
- Accuracy statistics when testing multiple images

## How It Works

1. **Face Detection**: MTCNN is used to detect faces in each frame from the webcam.

2. **Face Embedding**: The detected face is cropped and processed through a pre-trained InceptionResnetV1 model to extract a 512-dimensional face embedding vector.

3. **Vector Database**: Qdrant is used to store face embedding vectors along with the corresponding person's name. During recognition, the system performs a similarity search to find the closest match.

4. **Recognition**: The system displays the name of the recognized person along with a confidence score. If the confidence is below a threshold, the face is labeled as "Unknown".

## Customization

- Adjust the recognition threshold in the `recognize_face` function to control the strictness of face matching.
- Modify the preprocessing steps in the `preprocess_face` function if needed.
- Change the webcam device index in `cv2.VideoCapture(0)` if you have multiple cameras.

## Troubleshooting

- If you encounter issues with the webcam, make sure your camera is properly connected and accessible.
- For CUDA-related errors, ensure you have the correct version of PyTorch installed for your CUDA version.
- If face detection is slow, consider reducing the resolution of the webcam feed.
