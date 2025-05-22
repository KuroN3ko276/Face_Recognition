import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
import cv2

class FaceEmbedder:
    """
    A class for extracting face embeddings using a pre-trained FaceNet model.
    """
    
    def __init__(self, model_name='vggface2', device=None):
        """
        Initialize the face embedder.
        
        Args:
            model_name (str): Pre-trained model to use ('vggface2' or 'casia-webface')
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Load pre-trained model
        self.model = InceptionResnetV1(pretrained=model_name).to(self.device)
        self.model.eval()
        
        # Set embedding size
        self.embedding_size = 512
        
    def get_embedding(self, face_image):
        """
        Extract embedding from a face image.
        
        Args:
            face_image (numpy.ndarray): Face image (BGR format from OpenCV)
            
        Returns:
            numpy.ndarray: Face embedding vector (512-dimensional)
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalize
        img = rgb_image.astype(np.float32) / 255.0
        
        # Convert to PyTorch tensor and add batch dimension
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        
        # Move to device
        img = img.to(self.device)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model(img)
            
        # Convert to numpy and return
        return embedding.cpu().numpy()[0]
    
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (numpy.ndarray): First embedding
            embedding2 (numpy.ndarray): Second embedding
            
        Returns:
            float: Cosine similarity (higher means more similar)
        """
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Compute cosine similarity
        return np.dot(embedding1, embedding2)
    
    def batch_get_embeddings(self, face_images):
        """
        Extract embeddings from multiple face images.
        
        Args:
            face_images (list): List of face images
            
        Returns:
            numpy.ndarray: Array of face embeddings
        """
        if not face_images:
            return np.array([])
            
        # Process images
        processed_images = []
        for face in face_images:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            # Convert to float and normalize
            img = rgb_image.astype(np.float32) / 255.0
            
            # Convert to PyTorch tensor
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            processed_images.append(img)
            
        # Stack images into a batch
        batch = torch.stack(processed_images).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model(batch)
            
        # Convert to numpy and return
        return embeddings.cpu().numpy()