from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

import numpy as np

from sklearn.model_selection import cross_validate
from constants import scoring
from utils import evaluar_rendimiento

def arbol_decision(X_train, y_train):

    # Aplanar las imágenes para el Árbol de Decisión
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # De 4D a 2D

    param_grid = {
        'max_depth': [10, 20],  # Profundidad máxima del árbol
        'min_samples_split': [2, 10],  # Mínimo número de muestras para dividir un nodo
        'min_samples_leaf': [1, 2],  # Mínimo número de muestras en un nodo hoja
        'criterion': ['gini'],  # Función de división
    }

    # Búsqueda de los mejores parámetros utilizando GridSearchCV
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5) #Validación cruzada
    grid_search.fit(X_train_flat, y_train)

    # Obtener el mejor modelo
    model_tree = grid_search.best_estimator_

    return model_tree, grid_search  # Devolver el modelo entrenado

def arbol_decision_vgg16(X_train, y_train, X_test, y_test_encoded, input_shape=(256,256,3)):
    # Cargar VGG16 preentrenado SIN la capa superior
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Extraer características con VGG16
    X_train_features = feature_extractor.predict(X_train)
    X_test_features = feature_extractor.predict(X_test)

    # Aplanar las características extraídas
    X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
    X_test_features_flat = X_test_features.reshape(X_test_features.shape[0], -1)

    # Validación cruzada con las características extraídas
    model_tree_vgg = DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=42)

    scores_tree_vgg = cross_validate(model_tree_vgg, X_train_features_flat, y_train, cv=5, scoring=scoring)
    #print(f"Precisión media de validación cruzada con VGG16: {np.mean(scores_tree_vgg):.4f}")
    # Mostrar los resultados promedio de cada métrica
    print("Resultados de Validación Cruzada:")
    for metric in scoring:
        mean_score = np.mean(scores_tree_vgg[f'test_{metric}'])
        print(f"{metric}: {mean_score:.4f}")
    # Entrenar el modelo con todo el conjunto de entrenamiento
    model_tree_vgg.fit(X_train_features_flat, y_train)

    # Obtener las predicciones del modelo
    y_pred_prob = model_tree_vgg.predict_proba(X_test_features_flat)
    y_pred = np.argmax(y_pred_prob, axis=1)

    evaluar_rendimiento(
        y_test_encoded, y_pred_prob, y_pred, "Arbol decisión VGG16"
    )

    return model_tree_vgg, scores_tree_vgg

def arbol_vgg16_pca(X_train, y_train, X_test, y_test_encoded, input_shape=(256,256,3), n_components=500):
    # Cargar VGG16 preentrenado SIN la capa superior
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Extraer características con VGG16
    X_train_features = feature_extractor.predict(X_train)
    X_test_features = feature_extractor.predict(X_test)

    # Aplanar las características extraídas
    X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
    X_test_features_flat = X_test_features.reshape(X_test_features.shape[0], -1)

    # Reducción de dimensionalidad con PCA
    pca = PCA(n_components=n_components, svd_solver='randomized')
    X_train_pca = pca.fit_transform(X_train_features_flat)
    X_test_pca = pca.transform(X_test_features_flat)

    # Validación cruzada con PCA y VGG16
    model_tree_pca = DecisionTreeClassifier(criterion="gini", max_depth=20, random_state=42)
    
    scores_tree_pca = cross_validate(model_tree_pca, X_train_pca, y_train, cv=5, scoring=scoring)
    #print(f"Precisión media de validación cruzada con VGG16 y PCA: {np.mean(scores_tree_pca):.4f}")
    print("Resultados de Validación Cruzada:")
    for metric in scoring:
        mean_score = np.mean(scores_tree_pca[f'test_{metric}'])
        print(f"{metric}: {mean_score:.4f}")

    # Entrenar el modelo con todo el conjunto de entrenamiento
    model_tree_pca.fit(X_train_pca, y_train)

    # Obtener las predicciones del modelo
    y_pred_prob = model_tree_pca.predict_proba(X_test_pca)
    y_pred = np.argmax(y_pred_prob, axis=1)

    evaluar_rendimiento(
        y_test_encoded, y_pred_prob, y_pred, "Arbol decisión VGG16 PCA"
    )


    return model_tree_pca, scores_tree_pca