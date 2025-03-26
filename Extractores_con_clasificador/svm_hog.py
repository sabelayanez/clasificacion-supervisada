from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_validate
import numpy as np
from utils import validacion
from constants import CV, scoring


def svm_hog(X_train_hog, y_train_encoded, X_test_hog, y_test_encoded, y_test, class_names):
    # Crear y entrenar el clasificador SVM
    svm_model = SVC(kernel="linear", random_state=42)
    svm_model.fit(X_train_hog, y_train_encoded)

    # Validación cruzada para obtener métricas
    scores_svm = cross_validate(svm_model, X_train_hog, y_train_encoded, cv=CV, scoring=scoring)

    # Hacer predicciones en el conjunto de prueba
    y_pred_svm = svm_model.predict(X_test_hog)

    # Evaluar precisión
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"Precisión SVM con HOG: {accuracy_svm:.2f}")

    # Generar reporte de clasificación
    print("\nReporte de SVM con HOG:")
    print(classification_report(y_test, y_pred_svm))
    y_pred_svm = svm_model.predict(X_test_hog)

    validacion(X_test_hog, y_test_encoded, y_pred_svm, class_names)

    return svm_model, scores_svm
