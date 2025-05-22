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
import argparse

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

# Function to register faces from dataset
def register_faces_from_dataset(dataset_path, reset=False):
    # If reset is True, delete all points in the collection
    if reset:
        try:
            qdrant_client.delete_collection(collection_name="faces")
            print("Deleted existing collection 'faces'")
            
            qdrant_client.create_collection(
                collection_name="faces",
                vectors_config=models.VectorParams(
                    size=512,  # Dimension of the face embedding vector
                    distance=models.Distance.COSINE
                )
            )
            print("Created new collection 'faces'")
        except Exception as e:
            print(f"Error resetting collection: {e}")
    
    # Get the current count of points in the collection
    try:
        collection_info = qdrant_client.get_collection("faces")
        start_id = collection_info.vectors_count
        print(f"Starting with ID: {start_id}")
    except Exception:
        start_id = 0
    
    person_id = start_id
    total_registered = 0
    
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        
        if os.path.isdir(person_dir):
            print(f"Processing {person_name}...")
            person_registered = 0
            
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Read image
                        img = cv2.imread(img_path)
                        
                        if img is None:
                            print(f"  Error: Could not read {img_path}")
                            continue
                        
                        # Detect face
                        faces = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        
                        if faces:
                            # Get the largest face
                            face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
                            x, y, w, h = face['box']
                            
                            # Extract face
                            face_img = img[y:y+h, x:x+w]
                            
                            # Get embedding
                            embedding = get_face_embedding(face_img)
                            
                            # Add to Qdrant
                            qdrant_client.upsert(
                                collection_name="faces",
                                points=[
                                    models.PointStruct(
                                        id=person_id,
                                        vector=embedding.tolist(),
                                        payload={"person_name": person_name, "image_path": img_path}
                                    )
                                ]
                            )
                            
                            person_id += 1
                            person_registered += 1
                            total_registered += 1
                            print(f"  Registered face from {img_file}")
                        else:
                            print(f"  No face detected in {img_file}")
                    except Exception as e:
                        print(f"  Error processing {img_file}: {e}")
            
            print(f"Completed processing {person_name} - Registered {person_registered} faces")
    
    print(f"Total faces registered: {total_registered}")
    print(f"Total people in database: {len(os.listdir(dataset_path))}")

# Function to register a single person
def register_single_person(person_name, images_dir, reset_person=False):
    # Get the current count of points in the collection
    try:
        collection_info = qdrant_client.get_collection("faces")
        start_id = collection_info.vectors_count
        print(f"Starting with ID: {start_id}")
    except Exception:
        start_id = 0
    
    person_id = start_id
    total_registered = 0
    
    # If reset_person is True, delete all points for this person
    if reset_person:
        try:
            # Search for points with this person_name
            search_result = qdrant_client.scroll(
                collection_name="faces",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="person_name",
                            match=models.MatchValue(value=person_name)
                        )
                    ]
                ),
                limit=1000  # Adjust as needed
            )
            
            # Delete found points
            if search_result[0]:
                point_ids = [point.id for point in search_result[0]]
                qdrant_client.delete(
                    collection_name="faces",
                    points_selector=models.PointIdsList(
                        points=point_ids
                    )
                )
                print(f"Deleted {len(point_ids)} existing points for {person_name}")
        except Exception as e:
            print(f"Error resetting person: {e}")
    
    print(f"Processing {person_name}...")
    
    for img_file in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img_file)
        
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Read image
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"  Error: Could not read {img_path}")
                    continue
                
                # Detect face
                faces = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                if faces:
                    # Get the largest face
                    face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
                    x, y, w, h = face['box']
                    
                    # Extract face
                    face_img = img[y:y+h, x:x+w]
                    
                    # Get embedding
                    embedding = get_face_embedding(face_img)
                    
                    # Add to Qdrant
                    qdrant_client.upsert(
                        collection_name="faces",
                        points=[
                            models.PointStruct(
                                id=person_id,
                                vector=embedding.tolist(),
                                payload={"person_name": person_name, "image_path": img_path}
                            )
                        ]
                    )
                    
                    person_id += 1
                    total_registered += 1
                    print(f"  Registered face from {img_file}")
                else:
                    print(f"  No face detected in {img_file}")
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")
    
    print(f"Completed processing {person_name} - Registered {total_registered} faces")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register faces to the database')
    parser.add_argument('--dataset', type=str, default='Dataset', help='Path to the dataset directory')
    parser.add_argument('--reset', action='store_true', help='Reset the database before registering faces')
    parser.add_argument('--person', type=str, help='Register only a specific person')
    parser.add_argument('--reset-person', action='store_true', help='Reset the specific person before registering')
    
    args = parser.parse_args()
    
    if args.person:
        person_dir = os.path.join(args.dataset, args.person)
        if os.path.isdir(person_dir):
            register_single_person(args.person, person_dir, args.reset_person)
        else:
            print(f"Error: Directory for {args.person} not found at {person_dir}")
    else:
        register_faces_from_dataset(args.dataset, args.reset)