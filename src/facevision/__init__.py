"""facevision: face recognition implemented in NumPy.

    from facevision import FaceRecognizer

    rec = FaceRecognizer("mi_base")
    rec.capture("armando")                  # o rec.add_person("ana", ["a.jpg", "b.jpg"])
    for face in rec.recognize("foto.jpg"):
        print(face.name, face.distance)
    rec.live()

Pipeline: YuNet finds each face and five landmarks (the only pre-trained
part); everything after is implemented here: alignment by the landmarks,
Tan-Triggs illumination normalization, LBP histograms, histogram
equalization, PCA, LDA, distances, calibration and evaluation.

Modules
-------
detection      YuNet face detector (OpenCV's cv2.FaceDetectorYN)
alignment      similarity transform from 5 landmarks, bilinear warp
imaging        grayscale, resize, equalization, blur, Tan-Triggs
lbp            local binary patterns
features       pixel and LBP feature vectors of aligned crops
preprocessing  frames and files -> aligned crops; datasets
models         eigenfaces, fisherfaces, lbp and combined methods
calibration    rejection threshold from the registered photos
evaluation     rank-1 and verification error on a dataset
recognizer     FaceRecognizer: register people and recognize them
camera         camera loops (capture and live recognition)
cli            the ``facevision`` command
"""

from .alignment import align_face
from .detection import detect, detect_face, detect_faces
from .imaging import tan_triggs
from .lbp import lbp_histogram
from .models import DEFAULT_THRESHOLDS, METHODS, FaceModel, cosine_distances, lda, pca, train_model
from .preprocessing import FACE_SIZE, extract_face, extract_faces, load_dataset
from .recognizer import Face, FaceRecognizer

__version__ = "2.1.0"

__all__ = [
    "FaceRecognizer", "Face", "FaceModel", "train_model", "METHODS", "DEFAULT_THRESHOLDS",
    "detect", "detect_face", "detect_faces", "align_face", "extract_face", "extract_faces",
    "load_dataset", "FACE_SIZE", "lbp_histogram", "tan_triggs", "pca", "lda", "cosine_distances",
]
