import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

from constants import CV, scoring
from utils import evaluar_rendimiento, validacion

def knn(X_train_rgb, y_train_encoded, X_test_rgb):
    X_train_flattened = X_train_rgb.reshape(X_train_rgb.shape[0], -1)
    X_test_flattened = X_test_rgb.reshape(X_test_rgb.shape[0], -1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flattened)
    
    pca = PCA(n_components=50)  # You can adjust n_components as needed
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    knn_model = KNeighborsClassifier(n_neighbors=3)

    scoresKNN = cross_validate(knn_model, X_train_pca, y_train_encoded, cv=CV, scoring=scoring)
    
    knn_model.fit(X_train_pca, y_train_encoded)

    # Return both the trained KNN model and PCA model as a tuple
    return knn_model, pca, scoresKNN


def knn_with_gridsearch(X_train_rgb, y_train_encoded, X_test_rgb, y_test_encoded, class_names):
    # Aplanar las imágenes
    X_train_flattened = X_train_rgb.reshape(X_train_rgb.shape[0], -1)
    X_test_flattened = X_test_rgb.reshape(X_test_rgb.shape[0], -1)

    # Escalar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flattened)
    X_test_scaled = scaler.transform(X_test_flattened)

    # Definir el espacio de búsqueda de hiperparámetros
    param_grid = {
        'pca__n_components': [30, 50],
        'knn__n_neighbors': [3, 7]
    }

    # Crear un pipeline
    pipeline = Pipeline([
        ('pca', PCA()),
        ('knn', KNeighborsClassifier())
    ])

    # Aplicar GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=CV, scoring=scoring, refit='accuracy')
    grid_search.fit(X_train_scaled, y_train_encoded)

    print(f"Mejores parámetros: {grid_search.best_params_}")

    # Obtener el mejor modelo
    best_model = grid_search.best_estimator_
    best_pca = best_model.named_steps['pca']
    best_knn = best_model.named_steps['knn']

    results = pd.DataFrame(grid_search.cv_results_)[[
        'param_pca__n_components', 'param_knn__n_neighbors', 
        'mean_test_accuracy', 'mean_test_precision', 'mean_test_recall', 'mean_test_f1', 'mean_test_roc_auc'
    ]]

    # Obtener las predicciones del modelo
    y_pred_prob = grid_search.predict(X_test_scaled)  # Probabilidades de las clases
    y_pred = (y_pred_prob > 0.5).astype(int)  # Convertir las probabilidades en etiquetas de clase (binario)

    # Si es clasificación multiclase
    y_pred = np.argmax(y_pred_prob, axis=1)


    evaluar_rendimiento(
        y_test_encoded,
        y_pred_prob,
        y_pred,
        "KNN"
    )
    validacion(X_test_rgb, y_test_encoded, y_pred_prob, class_names)
    return best_knn, best_pca, results 

