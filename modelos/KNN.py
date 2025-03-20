import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

from constants import CV, scoring

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


def knn_with_gridsearch(X_train_rgb, y_train_encoded, X_test_rgb, y_test_encoded):
    # Aplanar las imágenes para convertirlas en una matriz 2D
    X_train_flattened = X_train_rgb.reshape(X_train_rgb.shape[0], -1)
    X_test_flattened = X_test_rgb.reshape(X_test_rgb.shape[0], -1)

    # Escalar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flattened)
    X_test_scaled = scaler.transform(X_test_flattened)

    # Definir el espacio de búsqueda de los hiperparámetros
    param_grid = {
        'pca__n_components': [30, 50],  # Número de componentes principales
        'knn__n_neighbors': [3, 7]  # Número de vecinos para el clasificador KNN
    }

    # Crear un pipeline para aplicar PCA seguido de KNN
    pipeline = Pipeline([
        ('pca', PCA()),  # PCA en el pipeline
        ('knn', KNeighborsClassifier())  # KNN en el pipeline
    ])

    # Usar GridSearchCV para buscar los mejores hiperparámetros
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)  # Validación cruzada
    grid_search.fit(X_train_scaled, y_train_encoded)

    # Imprimir los mejores parámetros encontrados
    print(f"Mejores parámetros: {grid_search.best_params_}")

    # Obtener el mejor modelo entrenado con los mejores hiperparámetros
    best_model = grid_search.best_estimator_

    # Extraer la mejor instancia de PCA y KNN
    best_pca = best_model.named_steps['pca']
    best_knn = best_model.named_steps['knn']
    results = pd.DataFrame(grid_search.cv_results_)[['param_pca__n_components', 'param_knn__n_neighbors', 'mean_test_score']]
    print(results.sort_values(by='mean_test_score', ascending=False))  # Ordenar de mejor a peor

    # Devolver el modelo KNN y la instancia de PCA como una tupla (igual que la función knn)
    return best_knn, best_pca
