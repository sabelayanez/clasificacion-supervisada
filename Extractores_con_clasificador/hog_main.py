from skimage.feature import hog
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report
from Extractores_con_clasificador import ann_hog, svm_hog

def extract_hog_features(images):
    hog_features = []
    for img in images:
        feature = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(feature)
    return np.array(hog_features)

def main_hog(X_train, X_test, y_train, y_test, num_classes, class_names):
    # Extraer características HOG
    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)

    # Entrenar y evaluar ANN con características HOG
    print("Entrenando y evaluando ANN con características HOG...")
    ann_model, scores_ann = ann_hog(X_train_hog, y_train, X_test_hog, y_test, num_classes, class_names)

    # Entrenar y evaluar SVM con características HOG
    print("\nEntrenando y evaluando SVM con características HOG...")
    svm_model, scores_svm = svm_hog(X_train_hog, y_train, X_test_hog, y_test, num_classes, class_names)

    return ann_model, scores_ann, svm_model, scores_svmS