import os
import cv2
import argparse
from tqdm import tqdm

from face_detector import FaceDetector
from face_embedder import FaceEmbedder
from vector_db import FaceDatabase

def train_from_dataset(dataset_path, db_path="face_db", min_confidence=0.9):
    """
    Train face recognition model from a dataset of face images.
    
    Args:
        dataset_path (str): Path to the dataset directory
        db_path (str): Path to store the Qdrant database
        min_confidence (float): Minimum confidence for face detection
    """
    print(f"Starting training from dataset: {dataset_path}")
    
    # Create database directory if it doesn't exist
    os.makedirs(db_path, exist_ok=True)
    
    # Initialize components
    detector = FaceDetector(min_confidence=min_confidence)
    embedder = FaceEmbedder()
    database = FaceDatabase(location=db_path)
    
    # Get list of person folders
    person_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    print(f"Found {len(person_folders)} persons in the dataset")
    
    # Process each person
    for person_name in person_folders:
        person_dir = os.path.join(dataset_path, person_name)
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing {len(image_files)} images for {person_name}")
        
        # Process each image
        successful_faces = 0
        for image_file in tqdm(image_files, desc=f"Processing {person_name}"):
            image_path = os.path.join(person_dir, image_file)
            
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not load image {image_path}")
                    continue
                
                # Detect faces
                faces = detector.detect_faces(image)
                
                if not faces:
                    print(f"Warning: No face detected in {image_path}")
                    continue
                
                # Use the face with highest confidence if multiple faces detected
                if len(faces) > 1:
                    print(f"Warning: Multiple faces detected in {image_path}, using the one with highest confidence")
                    faces = [max(faces, key=lambda x: x['confidence'])]
                
                # Extract face
                face_img = detector.extract_face(image, faces[0])
                
                # Get embedding
                embedding = embedder.get_embedding(face_img)
                
                # Add to database
                metadata = {
                    "source_image": image_file,
                    "dataset": dataset_path
                }
                database.add_face(embedding, person_name, metadata)
                
                successful_faces += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        print(f"Successfully added {successful_faces} face embeddings for {person_name}")
    
    # Print summary
    print("\nTraining Summary:")
    print(f"Total persons in database: {database.get_person_count()}")
    print("Training completed successfully!")

def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(description="Train Face Recognition from Dataset")
    parser.add_argument("--dataset", type=str, default="Dataset", 
                        help="Path to the dataset directory")
    parser.add_argument("--db", type=str, default="face_db", 
                        help="Path to store the face database")
    parser.add_argument("--confidence", type=float, default=0.9, 
                        help="Minimum confidence for face detection")
    
    args = parser.parse_args()
    
    train_from_dataset(
        dataset_path=args.dataset,
        db_path=args.db,
        min_confidence=args.confidence
    )

if __name__ == "__main__":
    main()