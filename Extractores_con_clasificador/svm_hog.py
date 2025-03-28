from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_validate
import numpy as np
from utils import validacion, evaluar_rendimiento
from constants import CV, scoring


def svm_hog(X_train_hog, y_train_encoded, X_test_hog, y_test_encoded, y_test, class_names, plot):
    # Crear y entrenar el clasificador SVM
    svm_model = SVC(kernel="linear", random_state=42, probability=True)
    svm_model.fit(X_train_hog, y_train_encoded)

    # Validación cruzada para obtener métricas
    scores_svm = cross_validate(svm_model, X_train_hog, y_train_encoded, cv=2, scoring=scoring)

    # Hacer predicciones en el conjunto de prueba
    y_pred_probs = svm_model.predict(X_test_hog)
    y_pred_prob = svm_model.predict_proba(X_test_hog)
    y_pred = np.argmax(y_pred_prob, axis=1)  # Obtener clases predichas

    # Evaluar precisión
    accuracy_svm = accuracy_score(y_test_encoded, y_pred_probs)
    classification_rep = classification_report(y_test_encoded, y_pred, output_dict=True)

    print(f"Precisión SVM con HOG: {accuracy_svm:.2f}")
    print("\nReporte de SVM con HOG:")
    print(classification_rep)
    # Generar reporte de clasificación
    if plot == True:
        y_pred = np.argmax(y_pred_prob, axis=1)  # Obtener clases predichas
        evaluar_rendimiento(
            y_test_encoded,
            y_pred_prob,
            y_pred,
            "SVM"
        )
        #validacion(X_test_hog, y_test_encoded, y_pred_svm, class_names)

    return svm_model, scores_svm
