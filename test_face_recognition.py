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
import matplotlib.pyplot as plt

# Initialize face detector
detector = MTCNN()

# Initialize face embedding model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize Qdrant client
qdrant_client = QdrantClient(path="face_db")

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

# Function to recognize face
def recognize_face(face_img, threshold=0.6):
    # Get embedding
    embedding = get_face_embedding(face_img)
    
    # Search in Qdrant
    search_result = qdrant_client.search(
        collection_name="faces",
        query_vector=embedding.tolist(),
        limit=5  # Get top 5 matches for testing
    )
    
    results = []
    for match in search_result:
        if match.score > threshold:
            results.append((match.payload["person_name"], match.score, match.payload["image_path"]))
    
    return results

# Function to test recognition on a single image
def test_single_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read {image_path}")
        return
    
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = detector.detect_faces(img_rgb)
    
    if not faces:
        print(f"No faces detected in {image_path}")
        return
    
    # Create figure for display
    fig, axes = plt.subplots(1, len(faces), figsize=(15, 5))
    if len(faces) == 1:
        axes = [axes]
    
    # Process each face
    for i, face in enumerate(faces):
        x, y, w, h = face['box']
        
        # Extract face
        face_img = img[y:y+h, x:x+w]
        
        # Recognize face
        recognition_results = recognize_face(face_img)
        
        # Draw rectangle on image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display results
        if recognition_results:
            top_match = recognition_results[0]
            label = f"{top_match[0]}: {top_match[1]:.2f}"
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f"Face {i+1}: Recognized as {top_match[0]} with confidence {top_match[1]:.2f}")
            
            # Display all matches
            print(f"  Top matches for Face {i+1}:")
            for j, (name, score, path) in enumerate(recognition_results):
                print(f"  {j+1}. {name}: {score:.4f} - {path}")
        else:
            label = "Unknown"
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f"Face {i+1}: Unknown")
        
        # Display face in subplot
        axes[i].imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(label)
        axes[i].axis('off')
    
    # Show the image with detections
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Face Recognition Results')
    plt.show()

# Function to test recognition on all images in a directory
def test_directory(directory_path):
    success_count = 0
    total_count = 0
    
    for img_file in os.listdir(directory_path):
        img_path = os.path.join(directory_path, img_file)
        
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"\nTesting {img_path}...")
            
            # Read image
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Error: Could not read {img_path}")
                continue
            
            # Detect faces
            faces = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if not faces:
                print(f"No faces detected in {img_path}")
                continue
            
            # Process each face
            for i, face in enumerate(faces):
                total_count += 1
                x, y, w, h = face['box']
                
                # Extract face
                face_img = img[y:y+h, x:x+w]
                
                # Recognize face
                recognition_results = recognize_face(face_img)
                
                # Check if correctly recognized
                if recognition_results:
                    top_match = recognition_results[0]
                    expected_name = os.path.basename(directory_path)
                    
                    if top_match[0] == expected_name:
                        success_count += 1
                        print(f"Face {i+1}: Correctly recognized as {top_match[0]} with confidence {top_match[1]:.2f}")
                    else:
                        print(f"Face {i+1}: Incorrectly recognized as {top_match[0]} instead of {expected_name}")
                else:
                    print(f"Face {i+1}: Not recognized (expected {os.path.basename(directory_path)})")
    
    # Print summary
    if total_count > 0:
        accuracy = (success_count / total_count) * 100
        print(f"\nTest Summary for {os.path.basename(directory_path)}:")
        print(f"Total faces: {total_count}")
        print(f"Correctly recognized: {success_count}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print(f"\nNo faces were tested in {directory_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test face recognition')
    parser.add_argument('--image', type=str, help='Path to a single image to test')
    parser.add_argument('--dir', type=str, help='Path to a directory of images to test')
    
    args = parser.parse_args()
    
    if args.image:
        test_single_image(args.image)
    elif args.dir:
        test_directory(args.dir)
    else:
        print("Please provide either --image or --dir argument")
        print("Example: python test_face_recognition.py --image path/to/image.jpg")
        print("Example: python test_face_recognition.py --dir path/to/person_directory")