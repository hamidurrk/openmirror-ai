import cv2
import os
import sys
import numpy as np
import warnings
from typing import Optional, Tuple
from insightface.app import FaceAnalysis

warnings.filterwarnings('ignore', category=FutureWarning)

os.environ['ORT_LOGGING_LEVEL'] = '3' 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)   
from config.config import *


class FaceVerifier:
    """
    A thread-safe face verification class using InsightFace.
    
    This class can be instantiated multiple times for parallel processing.
    Each instance maintains its own FaceAnalysis app for thread safety.
    """
    
    def __init__(self, model_name: str = 'buffalo_l', 
                 providers: list = None,
                 det_size: Tuple[int, int] = (1024, 1024),
                 ctx_id: int = 0):
        """
        Initialize the FaceVerifier with specified configuration.
        
        Args:
            model_name: InsightFace model name (default: 'buffalo_l')
            providers: List of execution providers (default: ['CPUExecutionProvider'])
            det_size: Detection size for face analysis (default: (1024, 1024))
            ctx_id: Context ID for GPU/CPU selection (default: 0)
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
            
        self.model_name = model_name
        self.providers = providers
        self.det_size = det_size
        self.ctx_id = ctx_id
        
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
    
    def get_embedding(self, img_path: str) -> Optional[np.ndarray]:
        """
        Extract face embedding from an image file.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Normalized face embedding as numpy array, or None if no face detected
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        
        faces = self.app.get(img)
        if not faces:
            return None
        
        return faces[0].normed_embedding
    
    def get_embedding_from_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from an image array.
        
        Args:
            img: Image as numpy array (BGR format)
            
        Returns:
            Normalized face embedding as numpy array, or None if no face detected
        """
        faces = self.app.get(img)
        if not faces:
            return None
        return faces[0].normed_embedding
    
    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings and return similarity score.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score (cosine similarity, range: -1 to 1)
        """
        return float(np.dot(embedding1, embedding2))
    
    def verify_faces(self, img_path1: str, img_path2: str, 
                     threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Verify if two face images belong to the same person.
        
        Args:
            img_path1: Path to first image
            img_path2: Path to second image
            threshold: Similarity threshold for matching (default: 0.5)
            
        Returns:
            Tuple of (is_match: bool, similarity_score: float)
        """
        embedding1 = self.get_embedding(img_path1)
        embedding2 = self.get_embedding(img_path2)
        
        if embedding1 is None:
            raise ValueError(f"No face detected in: {img_path1}")
        if embedding2 is None:
            raise ValueError(f"No face detected in: {img_path2}")
        
        similarity = self.compare_embeddings(embedding1, embedding2)
        is_match = similarity > threshold
        
        return is_match, similarity


def main():
    source_file = os.path.join(IMG_DIR, 'ts_1.png')
    target_file = os.path.join(IMG_DIR, 'ts_2.jpg')
    
    verifier = FaceVerifier()
    
    try:
        is_match, similarity = verifier.verify_faces(source_file, target_file)
        
        if is_match:
            print(f"Match Found! Score: {similarity:.2f}")
        else:
            print(f"No match. Score: {similarity:.2f}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()