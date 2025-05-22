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
# Run with default settings
python face_recognition.py

# Run with custom performance settings to reduce lag
python face_recognition.py --width 480 --height 360 --skip 4
```

4. Performance optimization options:
   - `--width`: Webcam width resolution (default: 640)
   - `--height`: Webcam height resolution (default: 480)
   - `--skip`: Process every n frames (default: 3)

   For better performance on slower hardware:
   - Reduce resolution (e.g., 480x360 or 320x240)
   - Increase the skip value (e.g., 4 or 5)

5. The system will:
   - Start the webcam for real-time face recognition
   - Display the recognized person's name and confidence score on the video feed
   - Show the current FPS (frames per second)

6. Press 'q' to quit the application.

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
- If you experience lag or stuttering:
  - Use the performance optimization options: `--width`, `--height`, and `--skip`
  - Try lower resolutions like 480x360 or 320x240
  - Increase the `--skip` value to 4 or 5 to process fewer frames
  - Check your CPU and GPU usage while running the application
  - Close other resource-intensive applications
  - If using a laptop, ensure it's plugged in and using the high-performance power plan
- The FPS counter can help you monitor performance improvements when adjusting settings
