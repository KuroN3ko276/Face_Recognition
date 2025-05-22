from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import uuid
import os

class FaceDatabase:
    """
    A class for managing face embeddings in a Qdrant vector database.
    """
    
    def __init__(self, collection_name="faces", location=":memory:", embedding_size=512):
        """
        Initialize the face database.
        
        Args:
            collection_name (str): Name of the collection in Qdrant
            location (str): Location of the Qdrant database (":memory:" for in-memory, or path to directory)
            embedding_size (int): Size of face embeddings
        """
        self.collection_name = collection_name
        self.embedding_size = embedding_size
        
        # Initialize Qdrant client
        self.client = QdrantClient(location=location)
        
        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()
        
    def _create_collection_if_not_exists(self):
        """Create the collection if it doesn't already exist."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created new collection: {self.collection_name}")
    
    def add_face(self, embedding, person_name, metadata=None):
        """
        Add a face embedding to the database.
        
        Args:
            embedding (numpy.ndarray): Face embedding vector
            person_name (str): Name of the person
            metadata (dict, optional): Additional metadata
            
        Returns:
            str: ID of the added face
        """
        # Generate a unique ID
        face_id = str(uuid.uuid4())
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata["person_name"] = person_name
        
        # Add the point to the collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=face_id,
                    vector=embedding.tolist(),
                    payload=metadata
                )
            ]
        )
        
        return face_id
    
    def search_similar_faces(self, embedding, limit=5, score_threshold=0.7):
        """
        Search for similar faces in the database.
        
        Args:
            embedding (numpy.ndarray): Query face embedding
            limit (int): Maximum number of results to return
            score_threshold (float): Minimum similarity score (0-1)
            
        Returns:
            list: List of dictionaries with search results
        """
        # Search for similar faces
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=limit,
            score_threshold=score_threshold
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "person_name": result.payload.get("person_name", "Unknown"),
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items() if k != "person_name"}
            })
            
        return results
    
    def get_all_faces(self):
        """
        Get all faces in the database.
        
        Returns:
            list: List of all faces with their metadata
        """
        # Scroll through all points in the collection
        scroll_results = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000  # Adjust as needed
        )
        
        # Format results
        results = []
        for point in scroll_results[0]:
            results.append({
                "id": point.id,
                "person_name": point.payload.get("person_name", "Unknown"),
                "metadata": {k: v for k, v in point.payload.items() if k != "person_name"}
            })
            
        return results
    
    def delete_face(self, face_id):
        """
        Delete a face from the database.
        
        Args:
            face_id (str): ID of the face to delete
            
        Returns:
            bool: True if successful
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[face_id]
            )
        )
        return True
    
    def delete_person(self, person_name):
        """
        Delete all faces of a person from the database.
        
        Args:
            person_name (str): Name of the person
            
        Returns:
            int: Number of faces deleted
        """
        # Find all faces of the person
        scroll_results = self.client.scroll(
            collection_name=self.collection_name,
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="person_name",
                        match=models.MatchValue(value=person_name)
                    )
                ]
            ),
            limit=10000  # Adjust as needed
        )
        
        # Get IDs of faces to delete
        face_ids = [point.id for point in scroll_results[0]]
        
        # Delete faces if any found
        if face_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=face_ids
                )
            )
            
        return len(face_ids)
    
    def get_person_count(self):
        """
        Get the number of unique persons in the database.
        
        Returns:
            int: Number of unique persons
        """
        # Get all unique person names
        scroll_results = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,  # Adjust as needed
            with_payload=True
        )
        
        # Extract unique person names
        person_names = set()
        for point in scroll_results[0]:
            person_name = point.payload.get("person_name")
            if person_name:
                person_names.add(person_name)
                
        return len(person_names)