from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.utils import to_categorical

def ann_hog(X_train_hog, y_train, X_test_hog, y_test, num_classes):
    # Convertir las etiquetas a one-hot encoding
    y_train_onehot = to_categorical(y_train, num_classes)
    y_test_onehot = to_categorical(y_test, num_classes)

    # Crear el modelo ANN
    ann_model = Sequential([
        Dense(512, activation='relu', input_dim=X_train_hog.shape[1]),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # num_classes = cantidad de clases
    ])

    # Compilar el modelo
    ann_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo ANN
    ann_model.fit(X_train_hog, y_train_onehot, epochs=5, batch_size=32, validation_data=(X_test_hog, y_test_onehot))

    # Obtener predicciones
    y_pred_ann = ann_model.predict(X_test_hog)
    y_pred_ann_classes = np.argmax(y_pred_ann, axis=1)

    # Calcular precisión
    accuracy_ann = accuracy_score(y_test, y_pred_ann_classes)
    print(f"Precisión ANN con HOG: {accuracy_ann:.2f}")

    # Generar reporte de clasificación
    print("\nReporte de ANN con HOG:")
    print(classification_report(y_test, y_pred_ann_classes))

    return ann_model