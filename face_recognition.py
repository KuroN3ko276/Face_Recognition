import os
import cv2
import numpy as np
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
import torchvision.transforms as transforms
from qdrant_client import QdrantClient
from qdrant_client.http import models
import matplotlib.pyplot as plt
import time

# Initialize face detector
detector = MTCNN()

# Initialize face embedding model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize Qdrant client
qdrant_client = QdrantClient(path="face_db")

# Create collection if it doesn't exist
try:
    qdrant_client.get_collection("faces")
    print("Collection 'faces' already exists")
except Exception:
    qdrant_client.create_collection(
        collection_name="faces",
        vectors_config=models.VectorParams(
            size=512,  # Dimension of the face embedding vector
            distance=models.Distance.COSINE
        )
    )
    print("Created new collection 'faces'")

# Function to preprocess face for embedding
def preprocess_face(face_img):
    # Convert to RGB if needed
    if len(face_img.shape) == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    elif face_img.shape[2] == 4:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGB)
    elif face_img.shape[2] == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    face_img = Image.fromarray(face_img)

    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return transform(face_img).unsqueeze(0).to(device)

# Function to get face embedding
def get_face_embedding(face_img):
    # Preprocess face
    face_tensor = preprocess_face(face_img)

    # Get embedding
    with torch.no_grad():
        embedding = resnet(face_tensor).cpu().numpy()[0]

    return embedding

# Check if the face database has any faces registered
def check_face_database():
    try:
        collection_info = qdrant_client.get_collection("faces")
        if collection_info.vectors_count == 0:
            print("Warning: No faces registered in the database.")
            print("Please register faces using register_faces.py before using face recognition.")
            print("Example: python register_faces.py")
            return False
        else:
            print(f"Found {collection_info.vectors_count} face embeddings in the database.")
            return True
    except Exception as e:
        print(f"Error checking face database: {e}")
        print("Please make sure to register faces using register_faces.py before using face recognition.")
        print("Example: python register_faces.py")
        return False

# Function to recognize face
def recognize_face(face_img, threshold=0.8):
    # Get embedding
    embedding = get_face_embedding(face_img)

    # Search in Qdrant
    search_result = qdrant_client.search(
        collection_name="faces",
        query_vector=embedding.tolist(),
        limit=1
    )

    if search_result and search_result[0].score > threshold:
        return search_result[0].payload["person_name"], search_result[0].score
    else:
        return "Unknown", 0.0

# Main function for webcam face recognition
def webcam_face_recognition(webcam_width=640, webcam_height=480, process_every_n_frames=3):
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Set resolution based on parameters to reduce lag
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Starting webcam face recognition. Press 'q' to quit.")

    # Variables for FPS calculation
    frame_count = 0
    fps = 0
    start_time = cv2.getTickCount()

    # Frame processing frequency (use parameter value)
    frame_counter = 0

    # Store detected faces for display between processing frames
    detected_faces = []

    # Store confidence values and timestamps for plotting
    confidence_data = []
    start_timestamp = time.time()

    while True:
        # Read frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
            break

        # Increment frame counter
        frame_counter += 1
        frame_count += 1

        # Calculate FPS every second
        if (cv2.getTickCount() - start_time) / cv2.getTickFrequency() >= 1.0:
            fps = frame_count
            frame_count = 0
            start_time = cv2.getTickCount()

        # Process only every n frames to reduce lag
        process_this_frame = frame_counter % process_every_n_frames == 0

        if process_this_frame:
            # Convert to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            faces = detector.detect_faces(rgb_frame)

            # Clear previous detected faces
            detected_faces = []

            # Process each face
            for face in faces:
                x, y, w, h = face['box']

                # Extract face
                face_img = frame[y:y+h, x:x+w]

                if face_img.size > 0:
                    # Recognize face
                    person_name, confidence = recognize_face(face_img)

                    # Print confidence to terminal
                    print(f"Detected: {person_name}, Confidence: {confidence:.4f}")

                    # Store face data for display
                    detected_faces.append({
                        'box': (x, y, w, h),
                        'name': person_name,
                        'confidence': confidence
                    })

                    # Store confidence data for plotting
                    current_time = time.time() - start_timestamp
                    confidence_data.append((current_time, person_name, confidence))

        # Display all detected faces (from current or previous frames)
        for face_data in detected_faces:
            x, y, w, h = face_data['box']

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display name and confidence
            label = f"{face_data['name']}: {face_data['confidence']:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display FPS
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Plot confidence graph if data was collected
    if confidence_data:
        print("Generating confidence graph...")

        # Create a figure and axis
        plt.figure(figsize=(12, 6))

        # Group data by person name
        person_data = {}
        for time_val, name, conf in confidence_data:
            if name not in person_data:
                person_data[name] = {'times': [], 'confidences': []}
            person_data[name]['times'].append(time_val)
            person_data[name]['confidences'].append(conf)

        # Plot data for each person with different colors
        for name, data in person_data.items():
            plt.plot(data['times'], data['confidences'], 'o-', label=f"{name}")

        # Add labels and title
        plt.xlabel('Time (seconds)')
        plt.ylabel('Confidence Score')
        plt.title('Face Recognition Confidence Over Time')
        plt.grid(True)
        plt.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()
    else:
        print("No confidence data collected for plotting.")

# Main execution
if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--width', type=int, default=640, help='Webcam width resolution')
    parser.add_argument('--height', type=int, default=480, help='Webcam height resolution')
    parser.add_argument('--skip', type=int, default=3, help='Process every n frames (higher values = less lag but less smooth recognition)')
    args = parser.parse_args()

    # Print performance settings
    print(f"Performance Settings:")
    print(f"- Resolution: {args.width}x{args.height}")
    print(f"- Processing: Every {args.skip} frames")
    print(f"- Higher 'skip' values and lower resolution will reduce lag")

    # Check if faces are registered in the database
    if check_face_database():
        # Start webcam face recognition with specified parameters
        webcam_face_recognition(
            webcam_width=args.width,
            webcam_height=args.height,
            process_every_n_frames=args.skip
        )
    else:
        print("Exiting. Please register faces first using register_faces.py")
