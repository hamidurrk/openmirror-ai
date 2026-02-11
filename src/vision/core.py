import cv2, os, sys
import numpy as np
import warnings
from insightface.app import FaceAnalysis

warnings.filterwarnings('ignore', category=FutureWarning)

os.environ['ORT_LOGGING_LEVEL'] = '3' 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)   
from config.config import *

source_file = os.path.join(IMG_DIR, 'ts_1.png')
target_file = os.path.join(IMG_DIR, 'ts_3.jpg')

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1024, 1024))

def get_embedding(img_path):
    img = cv2.imread(img_path)
    faces = app.get(img)
    if not faces:
        return None
    return faces[0].normed_embedding

source_face = get_embedding(source_file)
target_face = get_embedding(target_file)

if source_face is not None and target_face is not None:
    similarity = np.dot(source_face, target_face)
    
    if similarity > 0.5:
        print(f"Match Found! Score: {similarity:.2f}")
    else:
        print(f"No match. Score: {similarity:.2f}")