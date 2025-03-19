from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def svm_hog(X_train_hog, y_train, X_test_hog, y_test):
    # Crear y entrenar el clasificador SVM
    svm_model = SVC(kernel="linear", random_state=42)
    svm_model.fit(X_train_hog, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred_svm = svm_model.predict(X_test_hog)

    # Evaluar precisión
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"Precisión SVM con HOG: {accuracy_svm:.2f}")

    # Generar reporte de clasificación
    print("\nReporte de SVM con HOG:")
    print(classification_report(y_test, y_pred_svm))

    return svm_model
