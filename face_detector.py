import cv2
import numpy as np
from mtcnn import MTCNN

class FaceDetector:
    """
    A class for detecting faces in images using MTCNN.
    """
    
    def __init__(self, min_face_size=20, min_confidence=0.5):
        """
        Initialize the face detector.
        
        Args:
            min_face_size (int): Minimum face size to detect
            min_confidence (float): Minimum confidence threshold for face detection
        """
        self.detector = MTCNN(min_face_size=min_face_size)
        self.min_confidence = min_confidence
        
    def detect_faces(self, image):
        """
        Detect faces in an image.
        
        Args:
            image (numpy.ndarray): Input image (BGR format from OpenCV)
            
        Returns:
            list: List of dictionaries containing face information
                 (bounding box, confidence, landmarks)
        """
        # Convert BGR to RGB (MTCNN expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.detector.detect_faces(rgb_image)
        
        # Filter faces by confidence
        return [face for face in faces if face['confidence'] >= self.min_confidence]
    
    def extract_face(self, image, face_info, required_size=(160, 160)):
        """
        Extract a face from an image based on detection results.
        
        Args:
            image (numpy.ndarray): Input image (BGR format from OpenCV)
            face_info (dict): Face information from detect_faces
            required_size (tuple): Size to resize the face to
            
        Returns:
            numpy.ndarray: Extracted and aligned face image
        """
        # Get bounding box
        x, y, width, height = face_info['box']
        
        # Ensure bounding box doesn't go outside image
        x, y = max(0, x), max(0, y)
        right, bottom = min(image.shape[1], x + width), min(image.shape[0], y + height)
        width, height = right - x, bottom - y
        
        # Extract face
        face = image[y:y+height, x:x+width]
        
        # Resize to required size
        return cv2.resize(face, required_size)
    
    def draw_faces(self, image, faces):
        """
        Draw bounding boxes and landmarks on the image.
        
        Args:
            image (numpy.ndarray): Input image
            faces (list): List of face information from detect_faces
            
        Returns:
            numpy.ndarray: Image with drawn faces
        """
        img_copy = image.copy()
        
        for face in faces:
            # Draw bounding box
            x, y, width, height = face['box']
            cv2.rectangle(img_copy, (x, y), (x+width, y+height), (0, 255, 0), 2)
            
            # Draw confidence
            confidence = f"{face['confidence']:.2f}"
            cv2.putText(img_copy, confidence, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw landmarks
            for key, point in face['keypoints'].items():
                cv2.circle(img_copy, point, 2, (0, 0, 255), 2)
                
        return img_copy