from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
from utils import validacion
from constants import CV, scoring

def regresion_logistica(X_train, y_train_encoded, X_test, y_test_encoded):
    # Aplanar las imágenes de 4D a 2D
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Aplanar a 2D: [n_samples, n_features]
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    # Entrenar modelo de Regresión Logística
    modelLR = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)

    # Validación cruzada
    scoresLR = cross_validate(modelLR, X_train_flat, y_train_encoded, cv=CV, scoring=scoring)

    # Entren
    modelLR.fit(X_train_flat, y_train_encoded)
    # Realizar las predicciones sobre el conjunto de prueba
    y_pred_lr = modelLR.predict(X_test_flat)

    validacion(X_test, y_test_encoded, y_pred_lr)

    return modelLR, scoresLR