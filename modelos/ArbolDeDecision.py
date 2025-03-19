from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

def arbol_decision(X_train, y_train):

    # Aplanar las imágenes para el Árbol de Decisión
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # De 4D a 2D

    param_grid = {
        'max_depth': [10, 20, None],  # Profundidad máxima del árbol
        'min_samples_split': [2, 5, 10],  # Mínimo número de muestras para dividir un nodo
        'min_samples_leaf': [1, 2, 4],  # Mínimo número de muestras en un nodo hoja
        'criterion': ['gini', 'entropy'],  # Función de división
    }


    # Búsqueda de los mejores parámetros utilizando GridSearchCV
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5) #Validación cruzada
    grid_search.fit(X_train_flat, y_train)

    print(f"Mejores parámetros: {grid_search.best_params_}")

    # Obtener el mejor modelo
    model_tree = grid_search.best_estimator_

    return model_tree  # Devolver el modelo entrenado

def arbol_decision_vgg16(X_train, y_train, X_test, input_shape=(256,256,3)):
    # Cargar VGG16 preentrenado SIN la capa superior
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Extraer características con VGG16
    X_train_features = feature_extractor.predict(X_train)
    X_test_features = feature_extractor.predict(X_test)

    # Aplanar las características extraídas
    X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
    X_test_features_flat = X_test_features.reshape(X_test_features.shape[0], -1)

    # Entrenar un Árbol de Decisión con estas características
    model_tree_vgg = DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=42)
    model_tree_vgg.fit(X_train_features_flat, y_train)
    # Precisión usando VGG16 como extractor
    return model_tree_vgg

def arbol_vgg16_pca(X_train, y_train, X_test, input_shape=(256,256,3), n_components=500):
    
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Extraer características con VGG16
    X_train_features = feature_extractor.predict(X_train)
    X_test_features = feature_extractor.predict(X_test)

    # Aplanar las características extraídas
    X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
    X_test_features_flat = X_test_features.reshape(X_test_features.shape[0], -1)
    
    pca = PCA(n_components=500, svd_solver='randomized')  # Elegimos 200 características más relevantes
    X_train_pca = pca.fit_transform(X_train_features_flat)
    X_test_pca = pca.transform(X_test_features_flat)

    # Entrenar Árbol de Decisión con características reducidas
    model_tree_pca = DecisionTreeClassifier(criterion="gini", max_depth=20, random_state=42)
    model_tree_pca.fit(X_train_pca, y_train)

    # Precisión con VGG16 + PCA
    return model_tree_pca