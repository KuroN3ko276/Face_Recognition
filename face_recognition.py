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
def recognize_face(face_img, threshold=0.6):
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
def webcam_face_recognition():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Starting webcam face recognition. Press 'q' to quit.")

    while True:
        # Read frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
            break

        # Convert to RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = detector.detect_faces(rgb_frame)

        # Process each face
        for face in faces:
            x, y, w, h = face['box']

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Extract face
            face_img = frame[y:y+h, x:x+w]

            if face_img.size > 0:
                # Recognize face
                person_name, confidence = recognize_face(face_img)

                # Display name and confidence
                label = f"{person_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # Check if faces are registered in the database
    if check_face_database():
        # Start webcam face recognition
        webcam_face_recognition()
    else:
        print("Exiting. Please register faces first using register_faces.py")
