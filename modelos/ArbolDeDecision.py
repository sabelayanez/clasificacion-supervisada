from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

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