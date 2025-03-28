from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from utils import validacion, evaluar_rendimiento
from constants import scoring, CV
import matplotlib.pyplot as plt
import random
import numpy as np

def random_forest(X_train, y_train_encoded, X_test, y_test_encoded, class_names, plot):
    # Aplanar imágenes
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Crear y entrenar el modelo Random Forest
    modelRF = RandomForestClassifier(n_estimators=100, random_state=42)
    modelRF.fit(X_train_flat, y_train_encoded)
    if plot == True:
        y_pred = modelRF.predict(X_test_flat)
        y_pred_prob = modelRF.predict_proba(X_test_flat)  # Probabilidades para ROC

        # Validación cruzada
        scoresRF = cross_validate(modelRF, X_train_flat, y_train_encoded, cv=5, scoring=scoring)
        
        validacion(X_test, y_test_encoded, y_pred, class_names)
        # Visualizar imágenes y predicciones
        evaluar_rendimiento(y_test_encoded, y_pred_prob, y_pred, "Random Forest")

    return modelRF, scoresRF 

def rforest_vgg16_pca(X_train, y_train_encoded, X_test, y_test_encoded, class_names, plot, input_shape=(256, 256, 3), n_components=500):
    
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Extraer características con VGG16
    X_train_features = feature_extractor.predict(X_train)
    X_test_features = feature_extractor.predict(X_test)

    # Aplanar las características extraídas
    X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
    X_test_features_flat = X_test_features.reshape(X_test_features.shape[0], -1)
    
    pca = PCA(n_components=n_components, svd_solver='randomized')  # Elegimos 200 características más relevantes
    X_train_pca = pca.fit_transform(X_train_features_flat)
    X_test_pca = pca.transform(X_test_features_flat)

    modelRF = RandomForestClassifier(
        n_estimators=200,  # Más árboles = mejor generalización
        max_depth=30,  # Mayor profundidad
        min_samples_split=3,  # Menos datos necesarios para dividir
        min_samples_leaf=2,  # Evita ramas muy pequeñas
        random_state=42
    )

    # Validación cruzada
    scoresRF = cross_validate(modelRF, X_train_pca, y_train_encoded, cv=CV, scoring=scoring)

    modelRF.fit(X_train_pca, y_train_encoded)

    if plot == True:
        y_pred = modelRF.predict(X_test_pca)
        y_pred_prob = modelRF.predict_proba(X_test_pca)  # Probabilidades para ROC

        # Validación cruzada
        scoresRF = cross_validate(modelRF, X_train_pca, y_train_encoded, cv=5, scoring=scoring)

        validacion(X_test, y_test_encoded, y_pred, class_names)

        # Visualizar imágenes y predicciones
        evaluar_rendimiento(y_test_encoded, y_pred_prob, y_pred, "Random Forest vgg16 PCA")
    
    # Precisión con VGG16 + PCA
    return modelRF, scoresRF

def extract_hog_features(images):

    hog_features = []
    for img in images:
        feature = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(feature)

    return np.array(hog_features)

def rforest_vgg16_pca_hog(X_train, y_train, X_test, y_test_encoded, X_train_gray, X_test_gray, class_names, plot, input_shape=(256, 256, 3), n_components=500):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Extraer características con VGG16
    X_train_features = feature_extractor.predict(X_train)
    X_test_features = feature_extractor.predict(X_test)

    # Aplanar las características extraídas
    X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
    X_test_features_flat = X_test_features.reshape(X_test_features.shape[0], -1)
    
    pca = PCA(n_components=n_components, svd_solver='randomized')  # Elegimos 200 características más relevantes
    X_train_pca = pca.fit_transform(X_train_features_flat)
    X_test_pca = pca.transform(X_test_features_flat)

    # Extraer características HOG
    X_train_hog = extract_hog_features(X_train_gray)
    X_test_hog = extract_hog_features(X_test_gray)
    # Normalizar características
    scaler = StandardScaler()
    X_train_hog_scaled = scaler.fit_transform(X_train_hog)
    X_test_hog_scaled = scaler.transform(X_test_hog)

    # Concatenar VGG16 + HOG
    X_train_combined = np.hstack((X_train_pca, X_train_hog_scaled))
    X_test_combined = np.hstack((X_test_pca, X_test_hog_scaled))

    # Entrenar Random Forest con características combinadas
    model_rf_combined = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42)
    
    cv_scores = cross_validate(model_rf_combined, X_train_combined, y_train, cv=CV, scoring=scoring)

    model_rf_combined.fit(X_train_combined, y_train)
    if plot == True:
        y_pred = model_rf_combined.predict(X_test_combined)
        y_pred_prob = model_rf_combined.predict_proba(X_test_combined)  # Probabilidades para ROC

        # Visualizar imágenes y predicciones
        validacion(X_test, y_test_encoded, y_pred, class_names)
        evaluar_rendimiento(y_test_encoded, y_pred_prob, y_pred, "Random Forest vgg16 PCA HOG")

    return model_rf_combined, cv_scores
