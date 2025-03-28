from skimage.feature import hog
import numpy as np

from Extractores_con_clasificador.ann_hog import ann_hog
from Extractores_con_clasificador.svm_hog import svm_hog

def extract_hog_features(images):
    hog_features = []
    for img in images:
        feature = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True, channel_axis=-1)
        hog_features.append(feature)
    return np.array(hog_features)

def main_hog(X_train, y_train, X_test, y_test, class_names, plot):
    num_classes = len(class_names)
    # Extraer características HOG
    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)

    # Entrenar y evaluar ANN con características HOG
    print("Entrenando y evaluando ANN con características HOG...")
    ann_model, ann_scores = ann_hog(X_train_hog, y_train, X_test_hog, y_test, num_classes, class_names, plot)

    # Entrenar y evaluar SVM con características HOG
    print("\nEntrenando y evaluando SVM con características HOG...")
    svm_model, svm_scores = svm_hog(X_train_hog, y_train, X_test_hog, y_test, num_classes, class_names, plot)

    return ann_model, ann_scores, svm_model, svm_scores
