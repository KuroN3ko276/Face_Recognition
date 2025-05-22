# Face Recognition System

A real-time face recognition system using webcam, MTCNN for face detection, FaceNet for face embedding extraction, and Qdrant for vector similarity search.

## Features

- Real-time face detection from webcam feed
- Face recognition using vector similarity search
- User registration interface
- Visualization of detection and recognition results

## Requirements

- Python 3.7+
- Webcam
- CUDA-compatible GPU (optional, but recommended for better performance)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/face-recognition.git
   cd face-recognition
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```
     source .venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training with Your Dataset

Before using the application, you can train the system with your own dataset:

```
python train_face_recognition.py
```

The training script expects a dataset organized in the following structure:
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

Where each subfolder is named after a person and contains their face images.

#### Training Command-line Arguments

The training script accepts the following command-line arguments:

- `--dataset`: Path to the dataset directory (default: "Dataset")
- `--db`: Path to store the face database (default: "face_db")
- `--confidence`: Minimum confidence for face detection (default: 0.9)

Example:
```
python train_face_recognition.py --dataset my_custom_dataset --db my_face_database --confidence 0.85
```

### Running the Application

After training, run the main application:

```
python face_recognition_app.py
```

### Application Command-line Arguments

The application accepts the following command-line arguments:

- `--db`: Path to store the face database (default: "face_db")
- `--confidence`: Minimum confidence for face detection (default: 0.9)
- `--threshold`: Similarity threshold for face recognition (default: 0.7)

Example:
```
python face_recognition_app.py --db my_face_database --confidence 0.85 --threshold 0.75
```

### Using the Application

1. **Recognition Mode (Default)**:
   - The application starts in recognition mode
   - Detected faces will be highlighted with a green box if recognized, or red if unknown
   - Recognized faces will display the person's name and confidence score

2. **Registration Mode**:
   - Press 'r' to enter registration mode
   - Enter the name of the person to register when prompted
   - The application will collect 5 face images with different poses
   - Once complete, the application will return to recognition mode

3. **Other Controls**:
   - Press 'c' to cancel registration (when in registration mode)
   - Press 'q' to quit the application

## Project Structure

- `face_recognition_app.py`: Main application script
- `train_face_recognition.py`: Script for training the system with custom datasets
- `face_detector.py`: Face detection using MTCNN
- `face_embedder.py`: Face embedding extraction using FaceNet
- `vector_db.py`: Vector database operations using Qdrant
- `requirements.txt`: Required Python packages
- `face_db/`: Directory for storing the face database (created automatically)
- `Dataset/`: Directory containing training images organized by person

## How It Works

### Training Process

1. **Dataset Organization**: Images are organized in folders named after each person
2. **Face Detection**: MTCNN detects faces in each training image
3. **Feature Extraction**: FaceNet extracts a 512-dimensional embedding vector for each detected face
4. **Database Storage**: Embeddings are stored in Qdrant with person names as labels

### Recognition Process

1. **Face Detection**: MTCNN is used to detect faces in each frame from the webcam
2. **Feature Extraction**: FaceNet (InceptionResnetV1) extracts a 512-dimensional embedding vector for each detected face
3. **Vector Database**: Qdrant performs similarity search against stored embeddings
4. **Recognition**: Detected faces are compared with stored embeddings to identify the person

## Customization

- Adjust the confidence threshold for face detection (higher values are more strict)
- Adjust the similarity threshold for face recognition (higher values require closer matches)
- Modify the number of face images collected during registration by changing `max_registration_faces` in the code

## Troubleshooting

### Training Issues

- **No faces detected in training images**: Ensure images contain clear, front-facing faces
- **Multiple faces in training images**: The system will use only the face with highest confidence
- **Training is slow**: Consider using a GPU for faster processing
- **Poor recognition after training**: Try increasing the number of training images per person

### Recognition Issues

- **Webcam not working**: Make sure your webcam is properly connected and not being used by another application
- **Slow performance**: Consider using a GPU or reducing the resolution of the webcam feed
- **False recognitions**: Increase the similarity threshold for stricter matching
- **Failed detections**: Decrease the confidence threshold for more lenient face detection

## License

This project is licensed under the MIT License - see the LICENSE file for details.
