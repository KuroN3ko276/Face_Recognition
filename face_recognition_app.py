import cv2
import numpy as np
import time
import os
from datetime import datetime
import argparse

from face_detector import FaceDetector
from face_embedder import FaceEmbedder
from vector_db import FaceDatabase

class FaceRecognitionApp:
    """
    Main application for face recognition using webcam.
    """
    
    def __init__(self, db_path="face_db", min_confidence=0.9, similarity_threshold=0.7):
        """
        Initialize the face recognition application.
        
        Args:
            db_path (str): Path to store the Qdrant database
            min_confidence (float): Minimum confidence for face detection
            similarity_threshold (float): Threshold for face recognition
        """
        print("Initializing Face Recognition System...")
        
        # Create database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize components
        self.detector = FaceDetector(min_confidence=min_confidence)
        self.embedder = FaceEmbedder()
        self.database = FaceDatabase(location=db_path)
        
        # Configuration
        self.min_confidence = min_confidence
        self.similarity_threshold = similarity_threshold
        self.required_size = (160, 160)  # Size for face images
        
        # State variables
        self.mode = "recognition"  # "recognition" or "registration"
        self.registration_name = ""
        self.registration_faces = []
        self.max_registration_faces = 5
        
        print(f"System initialized. Database contains {self.database.get_person_count()} persons.")
    
    def process_frame(self, frame):
        """
        Process a single frame from the webcam.
        
        Args:
            frame (numpy.ndarray): Input frame from webcam
            
        Returns:
            numpy.ndarray: Processed frame with annotations
        """
        # Detect faces
        faces = self.detector.detect_faces(frame)
        
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Process each detected face
        for face_info in faces:
            # Extract face
            face_img = self.detector.extract_face(frame, face_info, self.required_size)
            
            # Get face embedding
            embedding = self.embedder.get_embedding(face_img)
            
            # Get bounding box
            x, y, width, height = face_info['box']
            
            if self.mode == "recognition":
                # Search for similar faces
                results = self.database.search_similar_faces(
                    embedding, 
                    limit=1, 
                    score_threshold=self.similarity_threshold
                )
                
                # Draw bounding box
                if results:
                    # Face recognized
                    person_name = results[0]["person_name"]
                    score = results[0]["score"]
                    color = (0, 255, 0)  # Green for recognized
                    label = f"{person_name} ({score:.2f})"
                else:
                    # Unknown face
                    color = (0, 0, 255)  # Red for unknown
                    label = "Unknown"
                
                cv2.rectangle(display_frame, (x, y), (x+width, y+height), color, 2)
                cv2.putText(display_frame, label, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            elif self.mode == "registration":
                # Draw bounding box in blue for registration
                cv2.rectangle(display_frame, (x, y), (x+width, y+height), (255, 0, 0), 2)
                
                # Add count of collected faces
                label = f"Registering: {len(self.registration_faces)}/{self.max_registration_faces}"
                cv2.putText(display_frame, label, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # If we have only one face in the frame, collect it
                if len(faces) == 1 and len(self.registration_faces) < self.max_registration_faces:
                    # Add a small delay to get different poses
                    if not self.registration_faces or time.time() - self.last_registration_time > 1.0:
                        self.registration_faces.append(face_img)
                        self.last_registration_time = time.time()
                        
                        # If we have collected enough faces, register the person
                        if len(self.registration_faces) >= self.max_registration_faces:
                            self._register_person()
        
        # Add mode and instructions to the frame
        self._add_instructions(display_frame)
        
        return display_frame
    
    def _register_person(self):
        """Register a new person with the collected face images."""
        if not self.registration_name or not self.registration_faces:
            print("Cannot register: missing name or face images")
            return
        
        print(f"Registering {self.registration_name} with {len(self.registration_faces)} face images...")
        
        # Get embeddings for all collected faces
        embeddings = self.embedder.batch_get_embeddings(self.registration_faces)
        
        # Add each face to the database
        for i, embedding in enumerate(embeddings):
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "face_index": i
            }
            self.database.add_face(embedding, self.registration_name, metadata)
        
        print(f"Successfully registered {self.registration_name}")
        
        # Reset registration state
        self.mode = "recognition"
        self.registration_name = ""
        self.registration_faces = []
    
    def _add_instructions(self, frame):
        """Add instructions and status to the frame."""
        # Add mode indicator
        mode_text = f"Mode: {self.mode.capitalize()}"
        cv2.putText(frame, mode_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add instructions based on mode
        if self.mode == "recognition":
            instructions = [
                "Press 'r' to enter registration mode",
                "Press 'q' to quit"
            ]
        else:  # registration mode
            instructions = [
                f"Registering: {self.registration_name}",
                f"Collected: {len(self.registration_faces)}/{self.max_registration_faces} faces",
                "Press 'c' to cancel registration",
                "Press 'q' to quit"
            ]
        
        # Draw instructions
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, 60 + i*30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def start_registration(self, name):
        """
        Start the registration process for a new person.
        
        Args:
            name (str): Name of the person to register
        """
        self.mode = "registration"
        self.registration_name = name
        self.registration_faces = []
        self.last_registration_time = 0
        print(f"Starting registration for {name}. Please look at the camera with different poses.")
    
    def cancel_registration(self):
        """Cancel the current registration process."""
        self.mode = "recognition"
        self.registration_name = ""
        self.registration_faces = []
        print("Registration cancelled.")
    
    def run(self):
        """Run the face recognition application with webcam input."""
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        print("Starting webcam. Press 'q' to quit.")
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break
            
            # Process the frame
            display_frame = self.process_frame(frame)
            
            # Display the frame
            cv2.imshow("Face Recognition", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                break
            elif key == ord('r') and self.mode == "recognition":
                # Enter registration mode
                name = input("Enter name for registration: ")
                if name:
                    self.start_registration(name)
            elif key == ord('c') and self.mode == "registration":
                # Cancel registration
                self.cancel_registration()
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")

def main():
    """Main function to run the face recognition application."""
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument("--db", type=str, default="face_db", 
                        help="Path to store the face database")
    parser.add_argument("--confidence", type=float, default=0.9, 
                        help="Minimum confidence for face detection")
    parser.add_argument("--threshold", type=float, default=0.7, 
                        help="Similarity threshold for face recognition")
    
    args = parser.parse_args()
    
    app = FaceRecognitionApp(
        db_path=args.db,
        min_confidence=args.confidence,
        similarity_threshold=args.threshold
    )
    
    app.run()

if __name__ == "__main__":
    main()